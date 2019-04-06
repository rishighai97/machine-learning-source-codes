# Data Preprocessing

# importing the libraries
import numpy as np #mathematics
import matplotlib.pyplot as plt #plot charts
import pandas as pd #handle csv's, import datasets

# importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[ : , :-1].values # first arguement means all rows, second arguement means ignore last column
y = dataset.iloc[ : , 3].values # first arguement means all rows, second arguement means only last column

# missing data
# using mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[ : , 1:3]) # only columns at index 1 and 2 have missing values, 3 since upper bound is neglected 
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

# ecnoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_x = LabelEncoder()
X[ : , 0] = labelencoder_x.fit_transform(X[ : , 0]) # 0,1,2 for France, Spain, Germany.0,1,2 suggests weight but countries are independent, hence dummy encoding, separate columns for all three countries and 1, for the country present 
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# splitting the data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling: standardization(we use this [(x-mean)/standard dev] )  normalization( [(x-max)/(max-min)])
# for euclidean distance, in other also feature scaling is used for faster convergance (training of model)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #first fit and transform X_train
X_test = sc_X.transform(X_test) #X_test is feature scaled based on the scale of X_test hence only transform



