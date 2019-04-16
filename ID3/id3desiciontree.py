import numpy as np
from id3 import Id3Estimator
from sklearn.metrics import accuracy_score
from writeresult import *
from scalingdata import *
import time

A = np.genfromtxt('data.txt', delimiter=',') 
# split data 
Xtrain = A[:60,0:5]
Ytrain = A[:60,5]

Xtest = A[60:80,0:5]
Ytest = A[60:80,5]

print("---------result id3 build decision tree non scaling data--------")

t= time.time()

clf = Id3Estimator()

clf.fit(Xtrain,Ytrain)

h = clf.predict(Xtest)
ouput(h,Ytest)
ed =time.time()
print("Time excution non scaling data",ed-t) 


print("---------result scaling data with standardization------------" )
t= time.time()

Xmean = np.mean(Xtrain,axis = 0 ) 
st = np.std(Xtrain,axis = 0)
X_test_stadar = standardization(Xtest,Xmean,st)
X_train_stadar = standardization(Xtrain,Xmean,st)

clf1 = Id3Estimator()

clf1.fit(X_train_stadar,Ytrain)

h = clf1.predict(X_test_stadar)

print("------eluvation system--------")

ouput(h,Ytest)

ed =time.time()

print("TIME EXCUTE WITH STANDARDIZATION" ,ed-t)


print("---------result scaling data with Scaling to unit length -----------" )
# scaling with unit length
# calculate norm2 each other vector in A

t= time.time()

p_train = []
p_test =  []
for i in range (0,60):
    p_train.append(np.sqrt((np.sum(Xtrain[i]*Xtrain[i]))))
for i in range(0,20):
    p_test.append(np.sqrt((np.sum(Xtest[i]*Xtest[i]))))

X_test_unit_length = unit_length(Xtest,p_test)
X_train_unit_length = unit_length(Xtrain,p_train)

clf2 = Id3Estimator()

clf2.fit(X_train_unit_length,Ytrain)

h = clf2.predict(X_test_unit_length)

print("------eluvation system--------")

ouput(h,Ytest)

ed =time.time()

print("TIME EXCUTE WITH STANDARDIZATION" ,ed-t)




#scaling with rescaling
print("-------scaling data with rescaling-------")

t= time.time()

X_test_rescaling = rescaling (Xtest)
X_train_rescaling = rescaling(Xtrain)

clf3 = Id3Estimator()

clf3.fit(X_train_rescaling,Ytrain)

h = clf3.predict(X_test_rescaling)

print("------eluvation system--------")

ouput(h,Ytest)

ed =time.time()

print("TIME EXCUTE WITH STANDARDIZATION" ,ed-t)
