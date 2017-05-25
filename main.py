# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:37:00 2017

@author: abu.chowdhury
"""

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2]

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


    

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
    
    
# Visualising the Linear Regression Results
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:,0],X[:,1], y[:], c='r')
ax.set_xlabel('Number of Students')
ax.set_ylabel('Number of Teachers')
ax.set_zlabel('Number of A+')

#ax.plot(X[:,0], X[:,1], lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')

# Trying surface plot
x = np.linspace(min(X[:,0]), max(X[:,0]), 100 )
y = np.linspace( min(X[:,1]), max(X[:,1]), 100 )
xx = np.transpose( np.array([x, y]) )
z = lin_reg_2.predict(poly_reg.fit_transform(xx) )
x,y = np.meshgrid(x,y)
ax.plot_surface(x,y,z, alpha=.8)

plt.show()



















