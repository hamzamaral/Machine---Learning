# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:57:13 2022

@author: HAMZA
"""

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("polynomial+regression.csv",sep =";")


x= df.araba_fiyat.values.reshape(-1,1)

y= df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x, y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
plt.show()

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(x,y)
#%% predict

y_  = lm.predict(x)

plt.plot(x, y_,color="red",label="linear")
plt.show()

print("10 milyon tl lik araba hizi : ",lm.predict([[10000]]))




# %%
# polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures

polynomial = PolynomialFeatures(degree=5)

xpolynomial = polynomial.fit_transform(x)

linear_regression = LinearRegression()
linear_regression.fit(xpolynomial,y)

y_head = linear_regression.predict(xpolynomial)

plt.plot(x,y_head,color= "green",label = "poly")
plt.legend()
plt.show()

print(linear_regression.coef_)

print("10 milyon tl lik araba hizi : ",linear_regression.predict([[]])





