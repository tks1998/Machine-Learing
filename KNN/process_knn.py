import numpy as np
#compute distance between set X_test and set X_train
from sklearn import metrics
def dist_ss_fast(Z, X):
    X2 = np.sum(X*X, 1) # square of l2 norm of each ROW of X
    Z2 = np.sum(Z*Z, 1) # square of l2 norm of each ROW of Z
    return Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2*Z.dot(X.T)

# sort array by first element 
def sortFirst(val):
    return val[0]
#choose label 

def findlabel(p,k,Y):
    dem0 = 0 
    dem1 = 0
    distance = [] 
    for i in range(0,len(p)):
        distance.append((p[i],i))
            #sort array distance by first element 
    distance.sort(key=sortFirst)
    for i in range(0,k):  
        if Y[distance[i][1]] == 1:
           dem1 = dem1 + 1
        else:
           dem0 = dem0 +1
    if dem1>dem0:
        return 1 
    else:
        return 0
def choose(k,distance_of_set,Y):
    label_train = []
    for i in range(0,len(distance_of_set)):
        label_train.append(findlabel(distance_of_set[i],k,Y))
    return label_train
def output(result,label):
    count = 0
    for i in range(0,len(label)):
        if result[i] == label[i]: 
            count = count + 1 
    print ("truth data  ",label)
    print ("traning data",result)
    print ("Accuracy " ,100*count/len(label))     


def Precision_f1_recall(traingdata ,  realdata ): 
    print('Precision          Recall       F1 score ')
    print('%.2f'% (metrics.precision_score(realdata, traingdata)) ,"              ",
    '%.2f'% (metrics.recall_score(realdata, traingdata)) ,"       ",
    '%2f'% (metrics.f1_score(realdata,traingdata)) )