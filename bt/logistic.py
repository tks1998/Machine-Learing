import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from numba import jit
from scipy.sparse import hstack
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes  import BernoulliNB
from prepare import *
"""
    read data with pandas
"""
data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)

""" get train set  """
trainData = data[:20000]
trainX = trainData['headline']
trainY = trainData['is_sarcastic']
""" get test set """ 

testData = data[20000:]
testX = testData['headline']
testY = testData['is_sarcastic']
#prepare data with tf idf 
vectorizer = TfidfVectorizer().fit(trainX)
vectorX = vectorizer.transform(trainX)
vectortestX = vectorizer.transform(testX)
@jit
def sigmoid(S):
    return 1/(1 + np.exp(-S))

@jit
def loss(p, y):
    return np.sum(-y * np.log(p) - (1 - y) * np.log(1 - p))

@jit
# add 1 columum
def add1Col(x):
    intercept = np.ones((x.shape[0], 1))
    return hstack((intercept, x))

# def batch gradient descent
@jit
def BGD(x, y,alpha ,numIter=10000):
    x = add1Col(x)
    theta = np.zeros(x.shape[1])
    m = x.shape[0]
    for i in range(numIter):
        z = x.dot(theta)
        h = sigmoid(z)
        lossf = loss(h,y)
        theta = theta - (alpha / m) * x.transpose().dot((h - y))

    return lossf , theta
alpha = 5
lossfuntion , theta = BGD(vectorX,trainY,alpha)
vectortestX = add1Col(vectortestX)
vectortestXX = vectortestX.tocsr()
# p save label 
# calculate label with sigmoid 
p=[]

for i in range(0,vectortestX.shape[0]):
    if  sigmoid(vectortestXX[i].dot(theta))>=0.5:
        p.append(1)
    else:
        p.append(0)
output('logistic  ' , score(p,testY))
print(lossfuntion)