import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from arff import Arff
from math import e
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("magic04.csv")
X = data.iloc[:,:len(data.columns)-1]
Y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1, test_size = 0.2)
clf = MLPClassifier(hidden_layer_sizes = (7,7), alpha = 1e-4, solver = 'adam')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))