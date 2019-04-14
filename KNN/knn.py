from process_knn import *

A = np.genfromtxt('data.txt', delimiter=',')

#split data from data.txt

X_train = A[:60, 0:5]
y_train = A[:60, 5]


X_test = A[60:80, 0:5]
y_test = A[60:80, 5]

#choose k
result = []
k=17
count = 0
# calculate distance between vector in X_test and vector in X_train    
for i in range(0,len(X_test)):
    result.append(choose(k,X_test[i],X_train,y_train))

for i in range(0,len(y_test)):
    if result[i] == y_test[i]: 
        count = count + 1 
print ("truth data  ",y_test)
print ("traning data",result)
print ("Accuracy " , k , "nn" , 100*count/len(X_test))     

