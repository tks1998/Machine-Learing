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
# calculate distance between X_test and X_train    
distance = dist_ss_fast(X_test,X_train)

#result choose k point neart 
result = choose(k,distance,y_train)

count = 0

for i in range(0,len(y_test)):
    if result[i] == y_test[i]: 
        count = count + 1 

print ("truth data  ",y_test)
print ("traning data",result)
print ("Accuracy " , k , "nn" , 100*count/len(X_test))     

print("scaling data")

t = np.mean(X_train,axis = 0 ) 
st = np.std(X_train,axis = 0)