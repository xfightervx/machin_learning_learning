from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from LogisticRegressionGD import plot_decision_regions
from tools import load_data_iris

X_train_std, X_test_std, y_train, y_test = load_data_iris()

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_train_std, y_train, classifier=lr)
plt.xlabel('petal Lenght[standarisation]')
plt.ylabel('petal width[standarisation]')
plt.legend(loc='upper left')
plt.show()


