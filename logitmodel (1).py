#!/usr/bin/env python
# coding: utf-8

# In[1]:
#Load libraries for data processing
import matplotlib.pyplot as plt
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from scipy.stats import norm
import pickle
##import dataset
data = pd.read_csv('I:\\Working directory\\clean-data.csv', index_col=False)
data.drop('Unnamed: 0',axis=1, inplace=True)

#Assign predictors to a variable of ndarray (matrix) type
array = data.values
X = array[:,1:31]
y = array[:,0]

#transform the class labels from their original string representation (M and B) into integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.preprocessing import StandardScaler
# Normalize the  data (center around 0 and scale to remove the variance).
scaler =StandardScaler()
Xs = scaler.fit_transform(X)

from sklearn.decomposition import PCA
# feature extraction
pca = PCA(n_components=10)
fit = pca.fit(Xs)

X_pca = pca.transform(Xs)
# # Applying logistic regression using the Prin Comp.
# > *First 2 components*
##Converting the model into dataframe
X_pca = pd.DataFrame(X_pca) 

##Selecting PCs with significant variances, first 3 PCs
X_PC1= X_pca.iloc[:,0:2]

##Model Building
import statsmodels.formula.api as sm
from sklearn.linear_model import LogisticRegression##To apply logistic regression

logit_model= LogisticRegression()
logit_model.fit(X, y)

# Saving model to disk
pickle.dump(logit_model, open('model2.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model2.pkl','rb'))
print(model.predict([[17.99,10.38, 122.80,1001.0,0.11840,0.27760,0.3001,0.14710,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193, 25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.11890]]))

