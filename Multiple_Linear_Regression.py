'''
Multiple Linear Regression
--------------------------

ways to build a model:
All - in: take all columns in the multiple linear regression
Backward Elimination:
    - Step 1: Select a significance level to stay in the model
    - Step 2: Fit the full model with all possible predictors
    - Step 3: Consider the predictor with the highest P-value.
    if P > SL, got to STEP4, otherwise go to FIN
    - Step 4: Remove the predictor
    - Step 5: Fit model without this variable
    - go to step 3 and repeat

    FIN: Your Model is Ready

Forward Selection:
    - Step 1: Select a significance level to stay in the model
    - Step 2: Fit all simple regression models y -xsubn
    - Step 3: Keep this variable and fit all possible modles with
    one extra predictor added to the one(s) you already have
    - Step 4: Consider the predictor with the lowest P-value. If
    P < SL, go to step 3, otherwise FIN

    FIN: Keep the previous Model

Bidirectional Elimination:
    - Step 1 Select a sig level to enter and to stay in the
    model e.g.: SLENTER = 0.05, SLSTAY = 0.05
    - Step 2: Perform the next step of Forward Selection (new
    variables must have: P < SLENTER to enter)
    - Step 3: perform all steps of Backward Elimination (old
    variables must have P < SLSTAY to stay)
    - Step 4: No new variables can enter and no old variables
    can exit

    FIN: Your Model is Ready

All possible Models:
    - Step 1: Select a criterion of goodness of fit
    - Step 2: Construct All possible Regression Models: 2^n-1
    total combinations
    - Step 3: Select the one with the best criterion

    FIN: Your Model is Ready

'''

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

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

