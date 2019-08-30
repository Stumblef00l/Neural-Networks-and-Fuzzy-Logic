"""

Created on 29/8/19
@author: Kunal Verma

"""

#Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset

dataset = pd.read_excel('data.xlsx', header  = None)
X = dataset.iloc[:, :2].values
y = dataset.iloc[:, 2:].values

#Splitting the dataset to training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)

# Feature scaling

from sklearn.preprocessing import StandardScaler
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

#Creating a Linear Regressor class

class StochasticLinearRegressor:
    """Linear Regression class that takes
    the training set as input """

    def __init__(self,X_train,y_train,epochs):
        self.__X_train = X_train
        self.__y_train = y_train
        self.__epochs = epochs
        self.__output_log = []
        self.__w = [0.3]*(X_train.shape[1]+1)
        self.__learning_rate = 0.003
        self.__cost = 0.0
        self.__train()

    def __calculateCost(self):
        """Method to calculate the current
         objective function value"""

        self.__cost = 0.0
        for i in range(0,len(self.__X_train)):
            current_cost = self.__w[0]
            for j in range(1,len(self.__w)):
                current_cost = current_cost + (self.__w[j])*(self.__X_train[i][j-1])
            current_cost = current_cost - self.__y_train[i]
            current_cost = (current_cost)*(current_cost)
            self.__cost = (self.__cost) + current_cost
        self.__cost = 0.5*(self.__cost)
        return self.__cost

    def __train(self):
        """Method to train the model
         on the training set"""

        new_cost = self.__calculateCost()
        current_values = []
        current_values.append(new_cost)
        current_values.append(self.__w)
        self.__output_log.append(current_values)
        for t in range(1,self.__epochs+1):
            current_values = []
            for i in range(0,len(self.__X_train)):
                current_cost = self.__w[0]
                for j in range(1,len(self.__w)):
                    current_cost = current_cost + (self.__w[j])*(self.__X_train[i][j-1])
                current_cost = current_cost - self.__y_train[i]
                self.__w[0] = self.__w[0] - (current_cost * self.__learning_rate)
                for j in range(1,len(self.__w)):
                    self.__w[j] = self.__w[j] - (current_cost * self.__learning_rate * self.__X_train[i][j-1])
            new_cost = self.__calculateCost()
            current_values.append(new_cost)
            current_values.append(self.__w)
            self.__output_log.append(current_values)
            print('Epoch: ' + str(t) + ' Cost: ' + str(new_cost))

    def predict(self, X_test):
        """Method to predict the values"""

        predictions = []
        for x in X_test:
            pred = self.__w[0]
            for j in range(0,len(x)):
                pred = pred + (self.__w[j+1] * x[j])
            predictions.append(pred)
        return predictions

    def getLog(self):
        return self.__output_log

regressor = StochasticLinearRegressor(X_train, y_train, 100)
predictions = regressor.predict(X_test)
