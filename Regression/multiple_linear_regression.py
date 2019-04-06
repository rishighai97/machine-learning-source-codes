# Data Preprocessing

# importing the libraries
import numpy as np #mathematics
import matplotlib.pyplot as plt #plot charts
import pandas as pd #handle csv's, import datasets

# importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , :-1].values # first arguement means all rows, second arguement means ignore last column
y = dataset.iloc[ : , 4].values # first arguement means all rows, second arguement means only last column

# ecnoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_x = LabelEncoder()
X[ : , 3] = labelencoder_x.fit_transform(X[ : , 3]) # 0,1,2 for France, Spain, Germany.0,1,2 suggests weight but countries are independent, hence dummy encoding, separate columns for all three countries and 1, for the country present 
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


# avoiding dummy variable trap
X = X[ : ,  1 : ]  

# splitting the data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting multiple linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)


# Predicting the test set results
y_pred = regressor.predict(X_test)


# Building the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones(shape = (50 , 1)).astype(int) , values = X, axis = 1)

X_opt = X[ : , [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[ : , [0, 1, 3, 4, 5]] # x2 > 0.05 (highest)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[ : , [0, 3, 4, 5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[ : , [0, 3, 5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[ : , [0, 3]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
