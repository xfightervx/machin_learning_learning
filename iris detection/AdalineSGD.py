import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

class AdalineSGD:
    """ADAsgdptive LInear NEuron with Stochastic Gradiant Diasandant classifier.

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
    

    def __init__(self, eta=0.01, n_iter=50,shuffle=True ,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle
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
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X , y)
            cost = []
            for xi,target in zip(X,y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self.w_initialize_weights(X.shape[1])
        if y.ravel().shape[0] >1:
            for xi ,target in zip(X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        return self

    def _shuffle(self, X,y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0, scale=0.01, size =1+m)
        self.w_initialized = True

    def _update_weights(self, xi , target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta *xi.dot(error)
        self.w_[0] +=self.eta *error
        cost = 0.5 * error**2
        return cost
    

    def net_input(self, X):
        """ Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0] 

    def activation(self, X):
        return X
    def predict(self, X):
        """Return class Label after unit step"""
        return np.where(self.net_input(X) >= 0.0 , 1 , -1)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s','x','o','^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min , x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,c=colors[idx],marker=markers[idx], label = cl, edgecolors='black')
    

df = pd.read_csv("iris detection\data\iris.data", header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1 , 1)
X = df.iloc[0:100, [0,2]].values

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()