from SBS import SBS
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from wine_dataset import standardscalewine

knn = KNeighborsClassifier(n_neighbors=5)
X_train_std,X_test_std,y_train,y_test = standardscalewine()
sbs = SBS(knn , k_features=1)
sbs.fit(X_train_std, y_train)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()