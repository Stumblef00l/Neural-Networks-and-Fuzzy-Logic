"""

Created on 29/8/19
@author: Kunal Verma

"""

#Importing the libraries

import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset

dataset = pd.read_excel('data.xlsx', header  = None)

#Adding a dummy column

dataset[3] = np.array([1.0]*349)

#Splitting into features and output variables

dataset_temp = pd.concat([dataset.iloc[:, :2],dataset.iloc[:, 3:]], axis = 1)
X = dataset_temp.iloc[:, :].values
y = dataset.iloc[:, 2:3].values

#Splitting the dataset to training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)

# Feature scaling

from sklearn.preprocessing import StandardScaler
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

#Fixing the mess from normalizing the last dummy column

for record in X_train:
    record[-1] = 1.0

for i in X_test:
    record[-1] = 1.0

#Creating a class for Vectorized Regression

class VectorizedRegressor:
    def __init__(self, X_train, y_train):
        self.__X_train = X_train
        self.__y_train = y_train
        self.__w = []
        self.__train()
        
    def __train(self):
        current_matrix = np.matrix(self.__X_train)
        next_matrix = np.matrix(self.__X_train)
        current_matrix = np.transpose(current_matrix)
        current_matrix = np.dot(current_matrix, next_matrix)
        current_matrix = inv(current_matrix)
        next_matrix = np.transpose(np.matrix(self.__X_train))
        current_matrix = np.dot(current_matrix, next_matrix)
        current_matrix = np.dot(current_matrix, np.matrix(y_train))
        self.__w = np.asarray(current_matrix)
    
    def returnWeights(self):
        return self.__w
    
    def predict(self, X_test):
        results = []
        for record in X_test:
            val = 0
            for j in range(0,len(self.__w)):
                val = val + record[j] * self.__w[j]
            results.append(val)
        return results
    
#Making predicions on test set
        
regressor = VectorizedRegressor(X_train, y_train)
predictions = regressor.predict(X_test)