import numpy as np

def XOR_dataset(n=1):
    np.random.seed(n)
    X_xor = np.random.randn(200,2)
    y_xor = np.logical_xor(X_xor[:,0]>0 , X_xor[:,1]>0)
    y_xor = np.where(y_xor,1,-1)
    return X_xor, y_xor