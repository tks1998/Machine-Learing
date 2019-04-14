import numpy as np
#compute distance between vector in X_test and set X_train
def dist_ps_fast(z, X):
    X2 = np.sum(X*X, 1) 
    z2 = np.sum(z*z) 
    return X2 + z2 - 2*X.dot(z) 
# sort array by first element 
def sortFirst(val):
    return val[0]
#choose label 
#
def choose(k,z,X,Y):
    p = dist_ps_fast(z,X)
    distance = []
    dem0=0
    dem1=0
    #compute distance vector and training set X
    for i in range(0,len(p)):
        distance.append((p[i],i))
    #sort array distance by first element 
    distance.sort(key=sortFirst)
    #choose k label nearest z and set label z
    for i in range(0,k):
        if Y[distance[i][1]] == 1:
            dem1 = dem1 + 1
        else:
            dem0 = dem0 +1 
    if dem0>dem1:
        return 0
    else:
       return 1

