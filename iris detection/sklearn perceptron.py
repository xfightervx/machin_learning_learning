from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from tools import load_data_iris
from tools import plot_decision_regions

import numpy as np


X_train_std, X_test_std, y_train, y_test = load_data_iris()
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
X_combined_std = np.vstack((X_train_std, X_test_std))
Y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=Y_combined, classifier=ppn, test_idx=range(105,150))
plt.xlabel('petal lenght[stenderazed]')
plt.ylabel('petsl width[stenderized]')
plt.legend(loc='upper left')
plt.show()

