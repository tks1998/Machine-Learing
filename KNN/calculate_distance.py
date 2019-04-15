from __future__ import print_function
import numpy as np
from time import time # for comparing runing time

d, N = 1000, 10000 # dimension, number of training points
X = np.random.randn(N, d) # N d-dimensional points

M = 100
Z = np.random.randn(M, d)

def dist_ps_fast(z, X):
    X2 = np.sum(X*X, 1) # square of l2 norm of each ROW of X
    z2 = np.sum(z*z) # square of l2 norm of z
    return X2 + z2 - 2*X.dot(z) # z2 can be ignored
def dist_ss_0(Z, X):
    M = Z.shape[0]
    N = X.shape[0]
    res = np.zeros((M, N))
    for i in range(M):
        res[i] = dist_ps_fast(Z[i], X)
    return res
# from each point in one set to each point in another set, fast
def dist_ss_fast(Z, X):
    X2 = np.sum(X*X, 1)     
    Z2 = np.sum(Z*Z, 1) 
    return Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2*Z.dot(X.T)
t1 = time()
D3 = dist_ss_0(Z, X)
print("half fast set2set running time:", time() - t1, "s")
t1 = time()
D4 = dist_ss_fast(Z, X)
print("fast set2set running time", time() - t1, "s")
print("Result difference:", np.linalg.norm(D3 - D4))