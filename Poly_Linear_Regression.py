#polynomial regression
#comparing linear to polynomial linear

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)

#if poly is selected, then you can use this as a config
#degree=2 is quadratic, 3 = cubic, etc...
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

print(X_poly)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("truth or bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg_2.predict(X_poly), color = "blue")
plt.title("truth or bluff (poly Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#predicts using linear regression at an x point:6.5
lin_reg.predict([[6.5]])

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))