import numpy as np

A = np.genfromtxt('data.txt', delimiter=',')

X = A[:60, 0:5]

Y = A[60:80, 0:5]

X2 = np.sum(X*X,1)
print(X2)
print(X2.reshape(-1,1))