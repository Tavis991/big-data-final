import pandas as pd 
import gower 
from sklearn.neighbors import DistanceMetric
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
assitant = {}
data = pd.read_csv('Portuguese.csv')

i=0
for label in data.isna().sum() : 
    if label : 
        print (f'missing data at label {label}, please correct dataset') 
        i+=1
if not i : 
    print ('no missing data, proceeding')

target = data['G3'] 
thresh = (target.max() - target.min()) / 3
lam = lambda x : 0 if x < thresh else  1 if x < 2*thresh else 2 
 
#dividing grades into 3 groups
target = target.apply(lam)

def apply_zerone(col, vals):
  
    for i in range(len(vals)) :   
        if "no" in vals[i]:   #no = 0, yes = 1
            return data[col].replace({ vals[i] : 0, vals[len(vals)-1-i] : 1} )  

    for i in range(len(vals)) :  # < 3  = 0  > 3 = 1
        if "LE" in vals[i]:
            return data[col].replace({ vals[i] : 0, vals[len(vals)-1-i] : 1})
      # every other case, 1st = 0, 2nd =  1. save in dict for visual aid 
    assitant[col] = {vals[0] : 0, vals[1] :1}
    return data[col].replace({ vals[0] : 0, vals[1] : 1})

data.drop(['G2','G1', 'G3'], inplace=True, axis=1)
for col in data.columns: 
    vals = data[col].unique()
    if len(vals) == 2 :
        data[col] = apply_zerone(col,vals)

data_dummy = pd.get_dummies(data, drop_first=True)

for i in range(3):
    data_std = StandardScaler().fit_transform(data_dummy)
    data_pca = PCA(n_components=2).fit_transform(data_std)

    data_pca = np.vstack((data_pca.T, target)).T

    df_pca = pd.DataFrame(data_pca, columns=['First_Component',
                                        'Second_Component',
                                        'class'])
    df_pca['class'].apply(int)
    # sns.FacetGrid(data=df_pca, hue='class')\
    #    .map(plt.scatter, 'First_Component', 'Second_Component')\
    #    .add_legend();

    plt.figure()
    plt.scatter(df_pca['First_Component'][df_pca['class'] == 0], df_pca['Second_Component'][df_pca['class'] == 0], color='blue', label='fail')
    plt.scatter(df_pca['First_Component'][df_pca['class'] == 1], df_pca['Second_Component'][df_pca['class'] == 1],  color='red', label='medium')
    plt.scatter(df_pca['First_Component'][df_pca['class'] == 2], df_pca['Second_Component'][df_pca['class'] == 2], color='yellow', label='top')
    plt.legend()
    plt.show()


    pca = PCA(n_components=len(data_std[0]))
    X_pca = pca.fit_transform(data_std)

    percent_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)

    cum_var_explained = np.cumsum(percent_var_explained)

    # Plot the PCA Spectrum
    plt.figure(1, figsize=(12, 6))
    plt.clf()
    plt.plot(cum_var_explained, linewidth=2)
    plt.axis('tight')
    plt.grid()
    plt.xlabel('n_components')
    plt.ylabel('Cumulative Explained Variance');
    plt.show()

    tsne = TSNE(n_components=2, random_state=0)

    data_tsne = tsne.fit_transform(data_std)

    X_tsne_data = np.vstack((data_tsne.T, target)).T
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
    df_tsne['class'].apply(int)
    #Plot the 2 components from t-SNE
    # sns.FacetGrid(df_tsne, hue='class')\
    #    .map(plt.scatter, 'Dim1', 'Dim2')\
    #    .add_legend();
    # plt.show()

    plt.figure()
    plt.scatter(df_tsne['Dim1'][df_tsne['class'] == 0], df_tsne['Dim2'][df_tsne['class'] == 0], color='cyan', label='fail')
    plt.scatter(df_tsne['Dim1'][df_tsne['class'] == 1], df_tsne['Dim2'][df_tsne['class'] == 1],  color='black', label='medium')
    plt.scatter(df_tsne['Dim1'][df_tsne['class'] == 2], df_tsne['Dim2'][df_tsne['class'] == 2], color='green', label='top')
    plt.legend()
    plt.show()

    pred = LogisticRegression(random_state=0, multi_class="multinomial").fit(data_std, target)
    pred2 = np.copy(pred) 
    importance = pred.coef_
    ez = np.abs(pred.coef_)
    to_remove_log = np.argsort(ez)

    # for i in importance :
    #     plt.bar([x for x in range(len(i))], i)
    #     plt.title('significance of parameters for logistic regression')
    #     plt.show()

    model = DecisionTreeRegressor()
    model.fit(data_std, target)
    importance = model.feature_importances_
    to_remove_tree = np.argsort(importance)
    # summarize feature importance
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) # I actually have many subplots, but this is not related to a problem
    ax.bar([x for x in data_dummy.columns], importance, align='center')
    ax.set_title('significance of parameters for Random Trees predictor')
    ax.set_ylabel('significance')
    ax.set_xticklabels([x for x in data_dummy.columns], rotation=70, ha='center')

    # plt.bar(, )
    # plt.title()
    plt.show()
    remove_list = np.copy(data_dummy.columns)

    for x in range (len(to_remove_tree)//3): # by trees
        data_dummy.drop(remove_list[to_remove_tree[x]], inplace=True, axis=1)

    # for row in to_remove_log :  #by logistic regression 
    #     for i in range(len(row)//3) : 
    #         try:
    #              data_dummy.drop(data.columns[row[i]], inplace=True, axis=1)
    #         except:
    #             pass
    print ([remove_list[s] for s in np.argsort(importance)[-10:]], "top 10 parameters")

print('End')