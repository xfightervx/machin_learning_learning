import matplotlib.pylab as plt
import numpy as np
from tools import XOR_dataset

X_xor, y_xor = XOR_dataset()
plt.scatter(X_xor[y_xor==1,0],X_xor[y_xor==1,1], c='b', marker='x', label= '1')
plt.scatter(X_xor[y_xor==-1,0],X_xor[y_xor==-1,1], c='r', marker='s', label= '-1')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc='best')
plt.show()
