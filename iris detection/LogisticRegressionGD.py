import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from tools import plot_decision_regions
from tools import sigmoid

class LogiscticRegressionGD(object):
    """Logistic Regression Gradiant Deacreas.

    Parameters:

    eta : float
        Learning rate (between 0.0 and 1.0) 
    n_iter : int
        Passes over the training dataset
    random_state : int.
        Random number generator seed for random weight intialization.

    shuffle : bool
        Shuffle training data if True to prevent cycles


    Attributes:

    w_ : id-array
        Weights after fitting
    errors_ : list
        Number of misclassifications in every epoch.
     
      
        """
    

    def __init__(self, eta=0.05, n_iter=100 ,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_initialized = False

    def fit(self, X, y):
        """ Fit training data.

        Parameters:

        X : (arrau), shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_featurs is the number of features.
        Y : array, shape = [n_samples]
            Target values.

        Returns:



        self : object
           """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0 , scale = 0.01 , size=1 +X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta* X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            cost = (-y.dot(np.log(output)) -  ((1-y).dot(np.log(1-output))))
            self.cost_.append(cost)
        return self
    

    def net_input(self, X):
        """ Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0] 

    def activation(self, z):
        return 1. /(1. +np.exp(-np.clip(z , -250 , 250)))
    def predict(self, X):
        """Return class Label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)


    

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
X_train_1_subset = X_train[(y_train==0) | (y_train ==1)]
y_train_1_subset = y_train[(y_train==0) | (y_train ==1)]
lrgd = LogiscticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_1_subset,y_train_1_subset)
plot_decision_regions(X=X_train_1_subset,y=y_train_1_subset,classifier=lrgd)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
