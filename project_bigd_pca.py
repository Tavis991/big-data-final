import pandas as pd 
import gower 
from sklearn.neighbors import DistanceMetric
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

assitant = {}
data = pd.read_csv('Portuguese.csv')
print (data.columns)

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


# def remove_shekr():
#     data.drop(, inplace=True, axis=1)
print (len(data_dummy.columns))
print (data_dummy.columns)
print (data_dummy.head(50))
print (target)
print(assitant)

