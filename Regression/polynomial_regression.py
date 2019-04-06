# Polynomial Regression


# Data Preprocessing

# importing the libraries
import numpy as np #mathematics
import matplotlib.pyplot as plt #plot charts
import pandas as pd #handle csv's, import datasets

# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[ : , 1 : 2].values # first column not needed since level is there. We specify 1:2 as only 1 gives us vector and we need matrix of features
y = dataset.iloc[ : , 2].values # first arguement means all rows, second arguement means only last column


"""
# splitting the data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
"""
# feature scaling: standardization(we use this [(x-mean)/standard dev] )  normalization( [(x-max)/(max-min)])
# for euclidean distance, in other also feature scaling is used for faster convergance (training of model)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #first fit and transform X_train
X_test = sc_X.transform(X_test) #X_test is feature scaled based on the scale of X_test hence only transform
"""

# Fit Linear Regression model to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fit polynomial regression model to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visulaize the linear regresssion results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

# Visulaize the polynomial regresssion results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

# Predicting new results using Linear Regression
lin_reg.predict(6.5)

# Predicting new results using Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))

