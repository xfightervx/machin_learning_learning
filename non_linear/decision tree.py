from sklearn.tree import DecisionTreeClassifier
from tools import load_data_iris
from tools import plot_decision_regions
import matplotlib.pyplot as plot
tree = DecisionTreeClassifier(criterion='gini',max_depth=3, random_state=0)
X_train_std, X_test_std, y_train, y_test, X_combined_std , y_combined = load_data_iris()
tree.fit(X_combined_std, y_combined)
plot_decision_regions(X_combined_std, y_combined, classifier=tree, test_idx=range(105,150))
plot.legend(loc='upper left')
plot.show()
