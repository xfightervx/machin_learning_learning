import pandas as pd
from tools import load_wine_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
def load_data():
    # loading dataset as pandas dataframe
    df = load_wine_dataset()
    # splitting dataset
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return X_train , X_test , y_train , y_test
def pca(k:int):
    X_train , _ , _, _ = load_data()
    # standardizing dataset
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    # get eigenpairs
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
    return eigen_vals, eigen_vecs

def viualize_variation():
    # plot explained variance ratio
    eigen_vals , _ = pca(13)
    total = sum(eigen_vals)
    var_exp = [(i/total) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='individual explained variance')
    plt.step(range(1, 14), cum_var_exp, where='mid', label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.show()

def feuture_transformation(k:int):
    # make  eigen paire
    eigen_vals , eigen_vecs = pca(k)
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    # sort eigen pairs in descending order
    eigen_pairs.sort(key=lambda k : k[0], reverse=True)

    # make transformation matrix W
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    return w


def transform_matrix_pca(k:int):
    X_train , _ , _,_ = load_data()
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    w = feuture_transformation(k)
    X_train_pca = X_train_std.dot(w)
    return X_train_pca

def visualiz_transformation(k:int):
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    X_train , _ , y_train ,_ = load_data()
    X_train_pca = transform_matrix_pca(k)
    for l,c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1], c=c, label=l, marker=m)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='upper left')
    plt.show()



if __name__ == '__main__':
    visualiz_transformation(13)



