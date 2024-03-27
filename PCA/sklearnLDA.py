from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from LDA import get_std_data
from sklearnPCA import plot_decision_region
import matplotlib.pyplot as plt

X_train_std, X_test_std, y_train, y_test = get_std_data()
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
plot_decision_region(X_test_lda, y_test, classifier=lda)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()