from __future__ import division
from sklearn.datasets import load_iris
from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures


import numpy as np
import pandas as pd
import tools

'''
Sigmoid function
Input: Parameter value
Output: Sigmoid of given paramter
'''
def sigmoid(X):
      return 1.0 / (1 + np.exp(-X))

'''
Read and process input data for processing
Input: null
Output: Features X and Labels Y of all examples
'''
def readData():
      iris = load_iris()
      X = iris.data[:100, :]
      Y = iris.target[:100]
      
      return X,Y

'''
Train for parameter Theta recursively using graident decent
Input: Training feature matrix and Training label matrix
Output: Parameter matrix for the given training data
'''
def trainModel(X, Y, eta, iterations):
      m, n = np.shape(X)
      theta = np.ones((n, 1))

      #Iterative Calcualtion of Theta
      for i in range(iterations):
            h = sigmoid(np.dot(X, theta))
            Y = Y.reshape((X.shape[0], 1))
            theta_i = eta * (np.sum((h-Y) * X, axis=0))
            theta_i = theta_i.reshape(theta.shape)
            theta = theta - theta_i

      return theta

'''
Predictor Function to predict the labels of the test data based on 
values of theta calculated above
Input: Test feature data, Test label data
Output: Predicted labels of the Test data
'''
def predict(X, theta):
      Y = np.dot(X, theta)
      Y_label = []
      for i in Y:
            if i[0] > 0:
                  Y_label.append(1)
            else:
                  Y_label.append(0)
      return Y_label

'''
Main Function. Performs KFolds cross validation and uses tertairy methods
to classify
Input: void
Output: void. Prints the final classification errors
'''
if __name__ == "__main__":
      X, Y = readData()
      kf = KFold(X.shape[0], n_folds=10, shuffle=True)
      poly = PolynomialFeatures(degree=1)
      X = poly.fit_transform(X)
      precision, recall = 0.0, 0.0
      f_measure, accuracy = 0.0, 0.0
      for test, train in kf:
            X_Train, X_Test = X[train], X[test]
            Y_Train, Y_Test = Y[train], Y[test]
            theta = trainModel(X_Train, Y_Train, 0.001, 500)
            Y_prediction = predict(X_Test, theta)
            p, r ,f, a = tools.createConfusion(\
                        Y_prediction, Y_Test.tolist(), [0,1])
            precision += p
            recall += r
            f_measure += f
            accuracy += a
      precision /= 10
      recall /= 10
      f_measure /= 10
      accuracy /= 10
      print "Two Class:\n \
                  Precision:\t%s\n \
                  Recall:\t%s\n \
                  F-Measure:\t%s\n \
                  Accuracy:\t%s\n" %\
                  (precision, recall, f_measure, accuracy)


