# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 23:45:01 2022

@author: HAMZA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv("random+forest+regression+dataset.csv",sep=";",header= None)

x =df.iloc[:,0].values.reshape(-1,1)
y =df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor

kl =RandomForestRegressor(n_estimators=100,random_state=42)
# n_estimators=100=split 100
# random_state = we use 42. stutaion
kl.fit(x,y)

print("7.8 seviyesinde fiyatın ne kadar olduğu: ",kl.predict(7.8))

p_ =np.arange(min(x),max(x),0.01).reshape(-1,1)

y_ = kl.predict(p_)

plt.scatter(x,y,color="red")
plt.plot(p_,y_,color="green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()

# Calculation of Evaluation Regression Model 

from sklearn.metrics import r2_score

y2_ = kl .predict(x)

print("r_score :" , r2_score(y, y2_) )


























