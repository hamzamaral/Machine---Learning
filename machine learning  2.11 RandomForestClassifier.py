# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 18:15:46 2022

@author: HAMZA
"""



import pandas as pd
import numpy as np
#%%  import data

data = pd.read_csv("data.csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace = True)

# %%
data.diagnosis = [ 1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)
#%% normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.15,random_state = 42)

#%% decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("decision tree score: ", dt.score(x_test,y_test))

#%%  random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100,random_state = 1)#n_estimators = 100 => 100 tree oluşur
rf.fit(x_train,y_train)
print("random forest algo result: ",rf.score(x_test,y_test))

#%%
"""from sklern.model_selection import train_test_split
xtrain,ytrain
from sklearn.tree import DecisionTreeClassifier
dt =DecisionTreeClassifier()
dt.fit(xtrain,ytrain)

print(dt.score(xtest,ytest))

from sklearn.random_projection import RandomForestClassifier

rd = RandomForestClassifier()
rd.fit(xtrain,ytrain)

print(rd.score(xtest,ytest))"""
