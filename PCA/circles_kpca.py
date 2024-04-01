from Kpca import get_circles_dataset
from sklearn.decomposition import KernelPCA
import matplotlib.pylab as plt
import numpy as np

X , y = get_circles_dataset()
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
fig , ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
X_kpca = kpca.fit_transform(X)
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((500, 1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((500, 1))-0.02, color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()