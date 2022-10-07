# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 23:06:22 2022

@author: HAMZA
"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df = pd.read_csv("decision+tree+regression+dataset.csv",sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)

y = df.iloc[:,1].values.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor

tre_ = DecisionTreeRegressor()

tre_.fit(x,y)

tre_.predict([[5.5]])

x_ =np.arange(min(x),max(x),0.01).reshape(-1,1)

y_ = tre_.predict(x_)

plt.scatter(x,y, color="red")

plt.plot(x_,y_,color="green")

plt.xlabel("tribün level")
plt.ylabel("ücret")
plt.show()













