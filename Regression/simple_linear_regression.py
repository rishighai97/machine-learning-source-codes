# Simple Linear Regression

# Data Preprocessing

# importing the libraries
import numpy as np #mathematics
import matplotlib.pyplot as plt #plot charts
import pandas as pd #handle csv's, import datasets

# importing the dataset
dataset = pd.read_csv('salary_Data.csv')
X = dataset.iloc[ : , :-1].values # first arguement means all rows, second arguement means ignore last column
y = dataset.iloc[ : ,1].values # first arguement means all rows, second arguement means only last column

# splitting the data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) #out of 30, 10 for testing

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set results
y_pred = regressor.predict(X_test)

# visualizing Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs. Years of Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualizing Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs. Years of Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()