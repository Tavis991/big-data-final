import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from copy import deepcopy


LEGENDS = {'fails': {'fails': 'orange', 'other': 'blue'}, 'top': {'top': 'orange', 'other': 'blue'}}
ASSITANT = {}
TOP_TEN = {}
PASS, GRAD_A = 7.5, 13 

lam = lambda x : 0 if x < PASS else  1 if x < GRAD_A else 2  #3 class   
lam_fail = lambda x : 1 if x < PASS else  0  # 2 class seperator, fail and other
lam_top = lambda x : 0 if x < GRAD_A else  1  # 2 class, top and other 

def apply_zerone(data, col, vals):#for replacing categorical data to binary data 
  
    for i in range(len(vals)) :   
        if "no" in vals[i]:   #no = 0, yes = 1
            return data[col].replace({ vals[i] : 0, vals[len(vals)-1-i] : 1} )  

    for i in range(len(vals)) :  # famsize < 3  = 0  > 3 = 1
        if "LE" in vals[i]:
            return data[col].replace({ vals[i] : 0, vals[len(vals)-1-i] : 1})
      # every other case, 1st = 0, 2nd =  1. save in dict for visual aid 
    ASSITANT[col] = {vals[0] : 0, vals[1] :1}
    return data[col].replace({ vals[0] : 0, vals[1] : 1})

def analyze(data_dummy, target, param): #main analysis function - PCA, sTNE, decision tree, 

    data_std = StandardScaler().fit_transform(data_dummy)
    data_pca = PCA(n_components=2).fit_transform(data_std)

    data_pca = np.vstack((data_pca.T, target)).T

    df_pca = pd.DataFrame(data_pca, columns=['First_Component',
                                        'Second_Component',
                                        'class'])
    df_pca['class'].apply(int)
    sns.FacetGrid(data=df_pca, hue='class')\
       .map(plt.scatter, 'First_Component', 'Second_Component')\
       .add_legend(LEGENDS[param]);

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
    # plt.show()

    tsne = TSNE(n_components=2, random_state=0)

    data_tsne = tsne.fit_transform(data_std)

    X_tsne_data = np.vstack((data_tsne.T, target)).T
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
    df_tsne['class'].apply(int)
    #Plot the 2 components from t-SNE
    sns.FacetGrid(df_tsne, hue='class')\
       .map(plt.scatter, 'Dim1', 'Dim2')\
       .add_legend(LEGENDS[param]);
    # plt.show()

    model = DecisionTreeRegressor()
    model.fit(data_std, target)
    importance = model.feature_importances_ 
    to_remove_tree = np.argsort(importance)
    # summarize feature importance
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) # I actually have many subplots, but this is not related to a problem
    ax.bar([x for x in data_dummy.columns], importance, align='center')
    ax.set_title('significance of parameters for Decision Tree predictor')
    ax.set_ylabel('significance')
    ax.set_xticklabels([x for x in data_dummy.columns], rotation=70, ha='center')
    fig.legend(LEGENDS[param])
    # plt.show()

    remove_list = np.copy(data_dummy.columns)
    for x in range (len(to_remove_tree)//3): # by decision trees
        data_dummy.drop(remove_list[to_remove_tree[x]], inplace=True, axis=1)

    top_ten = [remove_list[s] for s in np.argsort(importance)[-10:]]
    print (top_ten, f"top 10 parameters for {param}")
    return top_ten

def logistic(data_dummy, target, param): #logistic regression, with 10 top parameters of prior analysis  
    data_std = StandardScaler().fit_transform(data_dummy[TOP_TEN[param]])
    pred = LogisticRegression(random_state=0, multi_class="multinomial").fit(data_std, target)
    importance = pred.coef_

    fig = plt.figure()
    for i in range(len(importance)) :
        ax = fig.add_subplot(1, 1, 1) 
        ax.bar([x for x in TOP_TEN[param]], importance[i], align='center')
        ax.set_xticklabels([x for x in TOP_TEN[param]], rotation=70, ha='center')
        ax.set_title(f'significance of parameters for {param}, logistic regression ')
       # plt.legend(LEGENDS[param])

    print(f'score for prediction of {param}: ', pred.score(data_std, target))
    plt.show()

def read_data(file): #main function 
    data = pd.read_csv(file)
    i=0
    for label in data.isna().sum() :  #data verification, checking for null values 
        if label : 
            print (f'missing data at label {label}, please correct dataset') 
            i+=1
    if not i : 
        print ('no missing data, proceeding')
    # t_data, v_data = data.randomSplit([0.8, 0.2], 1)

    target = data['G3'] 
    plt.hist(target)
    data.drop(['G2','G1', 'G3'], inplace=True, axis=1)
    for col in data.columns: 
        vals = data[col].unique()
        if len(vals) == 2 :
            data[col] = apply_zerone(data, col, vals)

    data_dummy = pd.get_dummies(data, drop_first=True)

    target_b = target.apply(lam_fail)
    data = deepcopy(data_dummy)

    for i in range (3): 
        TOP_TEN_b= analyze(data, target_b, 'fails')
    TOP_TEN['fails'] = TOP_TEN_b

    target_c = target.apply(lam_top)
    data = deepcopy(data_dummy)

    for i in range (3): 
        TOP_TEN_c= analyze(data, target_c, 'top')
    TOP_TEN['top'] = TOP_TEN_c

    logistic(data_dummy, target_b, 'fails')
    logistic(data_dummy, target_c, 'top')

for file in ['Maths.csv', 'Portuguese.csv'] :
    read_data(file)