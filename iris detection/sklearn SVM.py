from tools import load_data_iris
from tools import plot_decision_regions
from sklearn.svm import SVC
import matplotlib.pyplot as plot

X_train_std, X_test_std, y_train, y_test, X_combined_std , y_combined = load_data_iris()

svm =SVC(kernel='linear' , C=1.0, random_state= 1)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm , test_idx=range(105,150))
plot.xlabel("petal lenght")
plot.ylabel("petal width")
plot.legend(loc='upper left')
plot.show()