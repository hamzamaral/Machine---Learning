# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:54:01 2022

@author: HAMZA
"""

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple_linear_regression_dataset.csv", sep=";")

x= df.iloc[:,[0,2]].values


y = df.maas.values.reshape(-1,1)


multiple_linear = LinearRegression()

multiple_linear.fit(x,y)

print("bo",multiple_linear.intercept_)

print("b1,b2",multiple_linear.coef_)

multiple_linear.predict(np.array([[10,35],[5,35]]))