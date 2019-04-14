import numpy as np
from  training_entropy import *
from sklearn.tree import DecisionTreeClassifier 

A = np.genfromtxt('data.txt', delimiter=',') 
# split data 
Xtrain = A[:60,0:5]
Ytrain = A[:60,5]

Xtest = A[60:80,0:5]
Ytest = A[60:80,5]

clf = train(Xtrain,Ytrain)
y_pred_entropy = prediction(Xtest,clf) 
ouput(Ytest, y_pred_entropy) 