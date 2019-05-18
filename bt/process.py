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

from sklearn import svm
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


"""
    process data with module TfidfVectorizer
    Tfidf :Convert a collection of raw documents to a matrix of TF-IDF features.
"""

vectorizer = TfidfVectorizer().fit(trainX)
vectorX = vectorizer.transform(trainX)
vectortestX = vectorizer.transform(testX)


"""
    processing data with batch gradient descent
"""


"""
    Knn
    process knn with KNeighborsClassifier    
    choose number loop training 
"""
number_point = 7 
KNNmodel = KNeighborsClassifier(n_neighbors = number_point, metric='euclidean').fit(vectorX, trainY)
y_traning  = KNNmodel.predict(vectortestX)
output('KNN model: ' , score(y_traning,testY))

"""
    processing data with id3
    model sklearn support DecisionTreeClassifier process data with desicion tree 
"""
desicion_tree = DecisionTreeClassifier().fit(vectorX, trainY)
y_traning = desicion_tree.predict(vectortestX)
output('Decision Tree model: ',score(y_traning,testY))



"""
    build model with SVM 
"""
from sklearn import svm
model = svm.SVC(kernel='linear')
model.fit(vectorX,trainY)

y_pred = model.predict(vectortestX)
output('SVM   ',score(y_pred , testY))

