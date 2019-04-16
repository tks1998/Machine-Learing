from process_knn import *
from scaling import *
import numpy as np
import time 
A = np.genfromtxt('data.txt', delimiter=',')

#split data from data.txt

X_train = A[:60, 0:5]
y_train = A[:60, 5]


X_test = A[60:80, 0:5]
y_test = A[60:80, 5]

#choose k
result = []
k=18
print("RESULT " , k , "NN : ") 
print("calculate none scaling")
# calculate distance between X_test and X_train    
st= time.time()
distance = dist_ss_fast(X_test,X_train)
#result choose k point neart 
result = choose(k,distance,y_train)
# calculate accuracy && print result
print("------eluvation system--------")
output(result,y_test)
Precision_f1_recall(result,y_test)
# pre process with scaling data #
ed = time.time()
print("/*********************/", ed-st)
print("result scaling data with standardization" )

Xmean = np.mean(X_train,axis = 0 ) 
st = np.std(X_train,axis = 0)
X_test_stadar = standardization(X_test,Xmean,st)
X_train_stadar = standardization(X_train,Xmean,st)

distance = dist_ss_fast(X_test_stadar,X_train_stadar)
result = choose(k,distance,y_train)

print("------eluvation system--------")
output(result,y_test)
Precision_f1_recall(result,y_test)

print("result scaling data with Scaling to unit length " )
# scaling with unit length
# calculate norm2 each other vector in A
p_train = []
p_test =  []
for i in range (0,60):
    p_train.append(np.sqrt((np.sum(X_train[i]*X_train[i]))))
for i in range(0,20):
    p_test.append(np.sqrt((np.sum(X_test[i]*X_test[i]))))

X_test_unit_length = unit_length(X_test,p_test)
X_train_unit_length = unit_length(X_train,p_train)
distance = dist_ss_fast(X_test_unit_length,X_train_unit_length)
result = choose(k,distance,y_train)

print("------eluvation system--------")
output(result,y_test)
Precision_f1_recall(result,y_test)
#scaling with rescaling
print("scaling data with rescaling")
X_test_rescaling = rescaling (X_test)
X_train_rescaling = rescaling(X_train)

distance = dist_ss_fast(X_test_rescaling,X_train_rescaling)
result = choose(k,distance,y_train)
output(result,y_test)
Precision_f1_recall(result,y_test)
