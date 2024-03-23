from tools import plot_decision_regions
from tools import XOR_dataset
from sklearn.svm import SVC
import matplotlib.pyplot as plt


svm = SVC(kernel='rbf', random_state=1 ,gamma = 0.1 , C=10000)
X_xor, y_xor = XOR_dataset()
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.show()