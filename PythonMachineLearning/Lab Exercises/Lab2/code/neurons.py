"""
Created on Tue Oct  2 14:43:09 2018

@author: cameron
"""

import numpy as np

class Perceptron(object):
    """
    Perceptron classifier based on the standard perceptron learning rule
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.weights = None
        self.errors_ = None
        
    def fit(self, X, y):
        '''
        DOCSTRING
        '''
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self 
    
    def net_input(self, X):
        '''
        DOCSTRING
        '''
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        '''
        DOCSTRING
        '''
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
class AdalineGD(Perceptron):
    '''
    Here we implement an adaptive linear neuron which will update weights based
    on a linear activation function instead of a step function. We apply a 
    quantizer on to this linear output to get our predicted class, inherits the 
    Perceptron class functions
    '''
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        super().__init__(eta, n_iter, random_state)

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.cost = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()

            cost = (errors ** 2).sum() / 2.0
            self.cost.append(cost)
        return self

    def activation(self, X):
        return self.net_input(X)