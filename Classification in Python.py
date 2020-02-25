#!/usr/bin/env python
# coding: utf-8

# In[3]:



# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os # to change/set working directory

# set working directory
os.chdir('/Users/garima/Downloads/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 14 - Logistic Regression')
os.getcwd()

#import csv file
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset.head()
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values

#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,Y,test_size = 0.25, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

################################# Logistic Regression

#Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#Predicting on test set
y_pred = classifier.predict(X_test)

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Calculate Precision, recall and F1-score
from sklearn.metrics import precision_score, recall_score, f1_score
precision_lr = precision_score(y_test, y_pred)
recall_lr = recall_score(y_test, y_pred)
f1_score_lr = f1_score(y_test, y_pred)

############################### Random Forest classification
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier_rf.fit(X_train, y_train)

#predicting on test test
y_pred_rf = classifier_rf.predict(X_test)

#Making confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

#Calculate Precision, recall and F1-score
from sklearn.metrics import precision_score, recall_score , f1_score
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_score_rf = f1_score(y_test, y_pred_rf)




# In[ ]:





# In[ ]:





# In[ ]:




