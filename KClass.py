from __future__ import division
from sklearn.datasets import load_iris
from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd
import tools

'''
Read Data from iris data set for the processing 3 class classification
Input: null
Output: np array of features and labels
'''
def readData():
      iris = load_iris()
      X = iris.data
      Y = iris.target
      
      return X, Y

'''
Helper function to generate indicator random variables
Input: Label matrix and class
Returns: Indicator random vector
'''
def indicator(Y, c):
      indicator_v = []
      for label in Y:
            if label == c:
                  indicator_v.append(1)
            else:
                  indicator_v.append(0)
      return np.asarray(indicator_v)

'''
SoftMax function to calculate the descriminant function of a given class
Input: theta of given class, X, thetas of all other classes
Output: SoftMax of dot products
'''
def softMax(theta, X, thetas):
      priorSum = getPriorSum(thetas, X)
      softmax = np.exp(np.dot(X, theta))
      softmax /= priorSum
      return softmax

'''
Sum of exp(theta.X) for all classes. Changes with every class iteration
Input: thetas of all classes, X
Output: Sum of all priors
'''
def getPriorSum(thetas, X):
      sum = 0
      m, n = np.shape(thetas)
      for i in range(n):
            sum += np.exp(np.dot(X, thetas[:,i]))
      return sum

'''
Train for parameter theta for K-Classes using gradient decent
Input: Features X and Labels Y, learning rate and no of iterations
Output: Parameter vector theta
'''
def trainModel(X, Y, eta, iterations, classList):
      m, n = np.shape(X)
      thetas = np.ones((n, len(classList)))
      for j in range(iterations):
            for i, c in enumerate(classList):
                  theta = thetas[:,i]
                  h = softMax(theta, X, thetas)
                  ind = indicator(Y, c)
                  theta_i = eta * (np.sum((h-ind).reshape(len(h),1)*X, axis=0))
                  theta_i = theta_i.reshape(theta.shape)
                  theta = theta - theta_i
                  thetas[:,i] = theta
      return thetas

'''
Function to predict the labels of y-test
Input: Test feature data, Test labels
Output: Predcition of the Y labels
'''
def predict(X_Test, thetas):
      Y_Prediction = []
      thetas = thetas.T
      for x in X_Test:
            h = -np.inf
            label = None
            for i, theta in enumerate(thetas):
                  h_dash = np.dot(x, theta)
                  if h_dash > h:
                        h = h_dash
                        label = i
            Y_Prediction.append(label)
      return Y_Prediction

'''
Main Function. Performs KFold cross validation and reports the errors
Input: null
Output: null
'''
if __name__ == "__main__":
      X, Y = readData()
      kf = KFold(X.shape[0], n_folds = 10, shuffle = True)
      poly = PolynomialFeatures(degree = 1)
      X = poly.fit_transform(X)
      precision , recall = 0, 0
      f_measure, accuracy = 0, 0
      for test, train in kf:
            X_Train, X_Test = X[train], X[test]
            Y_Train, Y_Test = Y[train], Y[test]
            thetas = trainModel(X_Train, Y_Train, 0.001, 10000, [0,1,2])
            Y_Prediction = predict(X_Test, thetas)
            p, r ,f, a = tools.createConfusion(\
                        Y_Prediction, Y_Test.tolist(), [0,1,2])
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


            
