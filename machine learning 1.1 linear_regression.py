# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 23:46:48 2022

@author: HAMZA
"""
"""pip install pip
pip install seaborn"""
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
"""   Linear Regression """
df = pd.read_csv("linear_regression_dataset.csv",sep = ";")

plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()
"""
pip install pip
pip install LinearRegression"""
from sklearn.linear_model import LinearRegression

linear_ = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_.fit(x,y)

#%% prediction
import numpy as np

b0 = linear_.predict([[0]])
print("b0: ",b0)

b0_ = linear_.intercept_
print("b0_: ",b0_)

b1 = linear_.coef_
print("b1: ",b1)

maa = 1663 + 1138*11
print(maa)

print(linear_.predict([[11]]))
 
array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)
#y= np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70]).reshape(-1,1)
plt.scatter(x,y)
plt.show()

y_head = linear_.predict(array)



plt.plot(array, y_head,color = "red")
plt.legend()
plt.show()

linear_reg.predict([[100]])




                          