from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from tools import load_data_iris
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
tree = DecisionTreeClassifier(criterion='entropy',max_depth=4, random_state=1)
X_train_std, X_test_std, y_train, y_test, X_combined_std , y_combined = load_data_iris()
tree.fit(X_combined_std, y_combined)
dot_data = export_graphviz(tree,filled=True,rounded=True,class_names=['Setosa','Versicolor','Virginica'],feature_names=['petal length','petal width'], out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('treedecision classifier.png')
