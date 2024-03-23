from sklearn.neighbors import KNeighborsClassifier
from tools import plot_decision_regions, load_data_iris
import matplotlib.pyplot as plt

X_train_std, X_test_std, y_train, y_test, X_combined_std , y_combined = load_data_iris()
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(106,150))
plt.xlabel('petal lenght')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()