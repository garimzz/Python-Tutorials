#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os # to change/set working directory

# set working directory
os.chdir('/Users/garima/Downloads/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing')
os.getcwd()

#import csv file
dataset = pd.read_csv('Data.csv')
dataset.head()
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values


#handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
X

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

#Hot encoding
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, y_train, X_test, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)
X_test
Y
#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_test


# In[ ]:




