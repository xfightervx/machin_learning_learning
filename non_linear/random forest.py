from sklearn.ensemble import RandomForestClassifier
from tools import plot_decision_regions, load_data_iris
import matplotlib.pyplot as plt

forest = RandomForestClassifier(criterion='gini',n_estimators=25,random_state=1,n_jobs=2)
X_train_std, X_test_std, y_train, y_test, X_combined_std , y_combined = load_data_iris()
forest.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=forest, test_idx=range(105,150))
plt.xlabel('petal lenght')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()