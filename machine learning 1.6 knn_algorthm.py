# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 21:45:03 2022

@author: HAMZA
"""
import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt 

data = pd.read_csv("data.csv")

data.columns
data.head()

data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
# %%
B = data[data.diagnosis == "B"]

M = data[data.diagnosis == "M"]

plt.scatter(M.radius_mean, M.texture_mean, color="blue",label="kötü ",alpha=0.5)

plt.scatter(B.radius_mean, B.texture_mean, color="red",label="iyi ",alpha=0.5)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.show()
plt.legend()

data.diagnosis = [ 1  if i == "M" else 0  for i in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis = 1)

# %%
# normalization 

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%
# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

# %%
# knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))

scor_listm = []

for i  in range(1,15) :
    knn2 = KNeighborsClassifier(n_neighbors= i)
    knn2.fit(x_train,y_train)
    scor_listm.append(knn2.score(x_test,y_test))
    
    
plt.plot(range(1,15),scor_listm)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


