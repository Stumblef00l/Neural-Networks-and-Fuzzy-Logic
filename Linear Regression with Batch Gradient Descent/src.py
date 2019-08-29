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
