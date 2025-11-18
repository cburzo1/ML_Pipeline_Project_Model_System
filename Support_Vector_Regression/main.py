'''
SVR
---
- for linear svr, instead of a line going through a
scatter plot like linear regression, its a tube, which
looks like a line in the middel and then its increased in
thickness. like a goth mommy. the distance from the edge
the tube to the center is called the epsipilon- insensitive
tube. What this provides is an established margin of error
that you can disregard. the points on the outside of the
tube still perform as ordinary least squares, but stop at
the edge of the tube. THey are called slack variables.

formula = 1/2*abs(w)^2 + C * SUM(epssubi + epssubi^*) ->min

th reason its called a support vector regression, is
because we draw the vector line from the origin to the
support vector points on the outside of the tube.

for non-linear svr,

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

'''
feature scaling is important with this data set because the salary
will be way larger and out of range. 
'''

print(X)
print(y)

y = y.reshape(len(y), 1)

print(y)

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

print(X)
print(y)

regressor = SVR(kernel = 'rbf')

regressor.fit(X,y)

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'blue')
plt.title("truth or bluff (svr)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()