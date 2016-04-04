from __future__ import division
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures
# from sklearn.neural_network import MLPClassifier
from pylab import *

import numpy as np
import os
import tools

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
Sigmoid function
Input: Parameter value
Output: Sigmoid of given paramter
'''
def sigmoid(X):
      return 1.0 / (1 + np.exp(-X))

'''
Read Data from MINST character image data from sklearn.dataset
Input: null
Output: numpy matrx X with normalized samples in rows. Label matrix Y
'''
def readData():
      mnist = fetch_mldata('MNIST Original')
      m,n = mnist.data.shape
      np.unique(mnist.target)
      X, Y = mnist.data/255.0 , mnist.target
      return X, Y

'''
Indicator Function
Input: Y labels, class
Output: Indicator Vector
'''
def indicator(Y, cls):
      Y_ind = []
      for l in Y:
            if  l == cls:
                  Y_ind.append(1)
            else:
                  Y_ind.append(0)
      return np.asarray(Y_ind)


'''
Use gradient decent algorithm to converge to the most suitable weights
Input: Feature matrix X, Label vecotr Y, Learning rate eta, No of iterations
Output: Weights of hidden layer w and output layer v
'''
def trainModel(X, Y, eta, iterations, clasList):
      poly = PolynomialFeatures(degree = 1)
      X = poly.fit_transform(X)
      m, n = np.shape(X)
      h = 2*n  + 1 # No of hidden layer paramaeters      
      parameter_w = np.ones((n, h))
      parameter_v = np.ones((h, len(classList)))
      for j in range(iterations):
            Z = []
            for i in range(h-1):
                  w = parameter_w[:,i]
                  Z_i = sigmoid(np.dot(X, w))
                  Z.append(Z_i)
            Z = np.transpose(np.asarray(Z))
            Z = poly.fit_transform(Z)
            
            Y_predict = []
            Y_ind_all = []
            for idx, cls in enumerate(classList):
                  v = parameter_v[:,idx]
                  Y_i = softMax(v, Z, parameter_v)
                  Y_predict.append(Y_i)
                  Y_ind = indicator(Y, cls)
                  Y_ind_all.append(Y_ind)
                  v_i = eta * (np.sum((Y_i - Y_ind).reshape(m,1) * Z, axis=0))
                  v_i = v_i.reshape(v.shape)
                  parameter_v[:,idx] = v - v_i
            Y_predict = np.transpose(np.asarray(Y_predict))
            Y_ind_all = np.transpose(np.asarray(Y_ind_all))

            for i in range(h):
                  sum_over_k = np.sum((Y_predict - Y_ind_all) * \
                        parameter_v[i,:].reshape(classList.shape), axis = 1)
                  w_i = sum_over_k * Z[:,i] * (1 - Z[:,i])
                  print w_i.shape
                  w_i = np.sum(w_i.reshape((m,1)) * X, axis = 0)
                  parameter_w[:,i] -= eta * w_i
      return parameter_w, parameter_v

'''
Prediction function uses the v and w weights to
calculate the class for given features
Input: parameter_w, parameter_v, X_test
Output: Y_test
'''
def predict(parameter_w, paramater_v, X):
      m, n = np.shape(X)
      h = 2*n
      poly = PolynomialFeatures(degree = 2)
      Z = []
      for i in range(h):
            w = parameter_w[:,i]
            Z_i = sigmoid(np.dot(X, w))
            Z.append(Z_i)
      Z = np.transpose(np.asarray(Z))
      Z = poly.fit_transform(Z)

      Y_class_val = np.dot(Z, parameter_v)
      Y_predict = []
      for i in range(m):
            Y_predict.append(np.argmax(Y_class_val[i,:]))
      return Y_predict

     

if __name__ == "__main__":
      X, Y = readData()
      classList = np.unique(Y)
      kf = KFold(X.shape[0], n_folds = 10, shuffle = True)
      precision, recall = 0, 0
      f_measure, accuracy = 0, 0
      for test, train in kf:
            X_train, X_test = X[train], X[test]
            Y_train, Y_test = Y[train], Y[test]
            param_w, param_v = \
                        trainModel(X_train, Y_train, 0.001, 10000, classList)
            Y_prediction = predict(param_w, param_v, X_test)
            p, r, f, a = tools.createConfusion(\
                        Y_prediction, Y_test.toList(), classList)
            precision += p
            recall += r
            f_measure += f
            accuracy += a
      precision /= 10
      recall /= 10
      f_measure /= 10
      accuracy /= 10
      print "MLP:\nPrecision:\t%s\nRecall:\t%s\nF-Measure\t%s\nAccuracy:\t%s\n"\
                  % (precision, recall, f-measure, accuracy)
            


