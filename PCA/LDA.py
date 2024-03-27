from Pca import load_data
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def get_std_data():
    # load data and standardize it
    X_train , X_test , y_train , y_test = load_data()
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std , X_test_std , y_train , y_test

def lda():
    #implement LDA
    X_train_std , X_train_std, y_train , y_train = get_std_data()
    # calculate mean vectors
    np.set_printoptions(precision=4)
    mean_vecs = []
    for label in range(1, 4):
        mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    # calculate within-class scatter matrix
    d= 13 # dimention of the dataset
    S_w = np.zeros((d, d))
    for label, mv in zip(range(1,4), mean_vecs):
        class_scatter = np.zeros((d, d))
        for row in X_train_std[y_train == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1)
            class_scatter += (row - mv).dot((row - mv).T)
        S_w += class_scatter
    
    # calculate between-class scatter matrix
    mean_overall = np.mean(X_train_std, axis=0)
    S_b = np.zeros((d, d))
    for i, mean_vec in enumerate(mean_vecs):
        n = X_train_std[y_train == i + 1, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1)
        mean_overall = mean_overall.reshape(d, 1)
        S_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    return S_w, S_b

def lda_eigen():
    # calculate eigen values and eigen vectora
    S_w, S_b = lda()
    eigen_vals, eigen_vecs = np.linalg.eigh(np.linalg.inv(S_w).dot(S_b))
    
    # get eigen pairs
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

    # sort eigen pairs in descending order
    eigen_pairs = sorted(eigen_pairs, key = lambda k : k[0], reverse=True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    return eigen_pairs, eigen_vals, eigen_vecs, w

def visuliaze_lda():
    # plot lda
    eigen_pairs, eigen_vals, eigen_vecs = lda_eigen()
    total = sum(eigen_vals)
    discr = [(i/total) for i in sorted(eigen_vals, reverse=True)]
    cum_discr = np.cumsum(discr)
    plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='individual explained variance')
    plt.step(range(1, 14), cum_discr, where='mid', label='cumulative explained variance')
    plt.xlabel('Linear Discriminants')
    plt.ylabel('Explained variance ratio')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='best')
    plt.show()

def projecting_lda_visualization():
    # project data feutures onto eigen vector
    _, _, _, w = lda_eigen()
    X_train_std , _, y_train , y_train = get_std_data()
    X_train_lda = X_train_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for l, c , m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_lda[y_train==l, 0],X_train_lda[y_train==l, 1] * (-1),c=c, label=l, marker=m)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='upper left')
    plt.show()

