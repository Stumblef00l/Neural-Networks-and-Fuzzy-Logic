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

class LinearRegressor:
    """Linear Regression class that takes
    the training set as input """
    
    def __init__(self,X_train,y_train,epochs):
        self.__X_train = X_train
        self.__y_train = y_train
        self.__epochs = epochs
        self.__output_log = []
        self.__train()
        self.__w = []
        for i in range(0,X_train.shape[1]+1):
            self.__w.append(0.3)
        self.__learning_rate = 0.3
        self.__cost = 0
    
    def __calculateCost(self):
        """Method to calculate the current objective function value"""
        
        self.__cost = 0
        for i in range(0,len(self.__X_train)):
            current_cost = self.__w[0]
            for j in range(1,len(self.__w)):
                current_cost = current_cost + (self.__w[j])*(self.__X_train[i][j-1])
            current_cost = current_cost - self.__y_train[i]
            current_cost = (current_cost)*(current_cost)
            self.__cost = self.__cost + current_cost
        self.__cost = 0.5*(self.__cost)
        return self.__cost

    def __train(self):
        """Method to train the model on the training set"""
        
        new_cost = self.__calculateCost()
        current_values = []
        current_values.append(new_cost)
        current_values.append(self.__w)
        self.__output_log.append(current_values)
        for t in range(1,self.__epochs+1):
            current_values = []
            sum_of_values = [0]*(len(self.__w))
            for i in range(0,len(self.__X_train)):
                current_cost = self.__w[0]
                for j in range(1,len(self.__w)):
                    current_cost = current_cost + (self.__w[j])*(self.__X_train[i][j-1])
                current_cost = current_cost - self.__y_train[i]
                sum_of_values[0] = sum_of_values[0] + current_cost
                for j in range(1,len(self.__w)):
                    sum_of_values[j] = sum_of_values[j] + (current_cost * self.__X_train[i][j-1]) 
            for j in range(0,len(self.__w)):
                self.__w[j] = self.__w[j] - (sum_of_values[j] * self.__learning_rate)
            new_cost = self.__calculateCost()
            current_values.append(new_cost)
            current_values.append(self.__w)
            self.__output_log.append(current_values)
    
    
    
                
            
        
        
    
    



