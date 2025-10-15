#data processing template
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#import dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, -1].values

print(X)
print()
print(y)
print()

#taking care of missing data
# We can have a function to take the percentage of missing data
# so if its a large dataset, you can have the option to just remove
# the missing data. but if you have a lot of missing data, you have
#other ways to configure the missing data handling
# way cn also choose to replace missing data by the average of all data
# exists. you can replace by median, and most frequent

imputer = SimpleImputer(missing_values = np.nan, strategy="mean")

imputer.fit(X[:, 1:3])
print(X[:, 1:3])
print()
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

# Encoding Categorical Data.
'''
Categorical data is data that represents a category. In our case its
spain france and germany. But Anything that represents a category like 
red, blue, white is also a category. The computer cannot process 
strings in relation to numbers so we have to transform categorical data
represented by strings into numbers. but numbering them 1,2,3 implies an
order, or a ranking "order matters" with integers, so we have to encode
in a way that the computer knows not to rank them. So with vectors:
spain = [1,0,0], france = [0,1,0], germany = [0,0,1] or what ever order
the point is that order doesnt matter when represented as a vector.
a 3d vector of binary integers can help you represent up to 2^3 categories.
'''

# we can have a config where order matters: 1,2,3, or order doesnt matter
# [vectors, ...]
print()
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# np.array turns this to a specific numpy array which is an expected
#input for the machine learning models. its like a special type cast

print(X)

#if you have binary categorical data like yes and no, you can represent
# it simply with 1 and 0

le = LabelEncoder()
y = le.fit_transform(y)
#machine learning models dont expect an np_array type for dependant variables

print()
print(y)

#apply feature scaling after splitting data
#Splitting the dataset into training and test sets
# for config, we can select what percentage the training set is from the
# testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#random state = 1 allows the random state to stay fixed so you get the same result
print()
print(X_train)
print()
print(X_test)
print()
print(y_train)
print()
print(y_test)

#feature scaling
#have all features in the same range
#Standardization will be mostly used, but normalization will be
#used when you have normal data
# we can config standardization or normalization

sc = StandardScaler()

print(X_train[:, 3:5])

#fit computes values. here it computes the mean and the standardization
X_train[:, 3:5] = sc.fit_transform(X_train[:, 3:5])
X_test[:, 3:5] = sc.transform(X_test[:, 3:5])

print()
print(X_train)
print()
print(X_test)

