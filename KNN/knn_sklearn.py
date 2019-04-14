import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


A = np.genfromtxt('data.txt', delimiter=',')

X_train = A[:60, 0:5]
y_train = A[:60, 5]

X_test = A[60:80, 0:5]
y_test = A[60:80, 5]

clf = neighbors.KNeighborsClassifier(n_neighbors=17, p=2)  

clf.fit(X_train, y_train)

h = clf.predict(X_test)

print ("training labels: " , h)
print ("truth label    : " , y_test)

print ("Accuracy of 17KNN: %.2f %%" % (100*accuracy_score(y_test, h)))