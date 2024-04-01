from Pca import load_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles


def rbf_kernel_pca(X, gamma, n_componants):
    """
    RBF kernel PCA implementation

    Parameters

    X: Numpy array , shape =[n_sampples, n_features]

    gamma: float , Tuning parameter of the RBF kernel

    n_componants:int  Number of principale components to return

    Returns

    X_pc: Numpy array , shape[n_samples, k_features] Projected dataset
    """

    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # Collect the top k eigenvectors (projected samples)
    alpha = np.column_stack((eigvecs[:, i] for i in range(n_componants)))
    lambdas = [eigvals[i] for i in range(n_componants)]

    return alpha, lambdas

def get_moons_dataset(n = 100, rs=1):
    return make_moons(n_samples=n, random_state=rs)

def get_circles_dataset(n = 1000, rs=1):
    return make_circles(n_samples=n, random_state=rs, noise=0.1, factor=0.2)
if __name__ == '__main__':
    X , y = get_circles_dataset()
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
    plt.show()