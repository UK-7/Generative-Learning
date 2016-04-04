from __future__ import division
import numpy as np
import math

'''
Compute the confusion matrix for a general class of classification
Hope you appreciate the pun ;)
Input: Prediction, Label, Class Set as vector
Output: Precision, Recall, F-Measure, Accuracy
'''

def createConfusion(y_hat, y, classSet):
      confusion = np.zeros((len(classSet), len(classSet)))
      
      '''
      for i in range(len(y_hat)):
            print "Prediction: %s\tActual:%s" % (y_hat[i], y[i])
      '''
      

      for i in range(len(y_hat)):
            confusion[y_hat[i],y[i]] += 1
      precision = np.zeros(len(classSet))
      recall = np.zeros(len(classSet))
      f_measure = np.zeros(len(classSet))
      for _class in classSet:
            precision[_class] = \
                        confusion[_class, _class] / \
                        np.sum(confusion, axis=1, dtype='float')[_class]
            if math.isnan(precision[_class]):
                  precision[_class] = 0
            recall[_class] = \
                        confusion[_class, _class] / \
                        np.sum(confusion, axis=0, dtype='float')[_class]
            if math.isnan(recall[_class]):
                  recall[_class] = 0
            f_measure[_class] = 2*(precision[_class] * recall[_class])/\
                        (precision[_class] + recall[_class])
            if math.isnan(f_measure[_class]):
                  f_measure[_class] = 0
      accuracy = np.trace(confusion)/np.sum(confusion)
      return precision, recall, f_measure, accuracy

