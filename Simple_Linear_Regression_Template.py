#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#import dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#print(X_train, X)

#Regression is used when you have to predict a continuous real value
#like a salary. Classification is wen you have to predict a category
regressor = LinearRegression()
#Train the model
regressor.fit(X_train, y_train)
#this .fit gives us the equation of the best fit line and later when we plot,
#we plug the X_train x components into the x of y = b +bx, to get y_pred. Which
# point on the best fit line. To view how good the line is, view it on the test
#set

y_pred = regressor.predict(X_test)

print(y_pred)

plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")

plt.title('Salary vs Exp (Training Set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")

plt.title('Salary vs Exp (Training Set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()