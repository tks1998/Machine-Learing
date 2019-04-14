from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

def train(X_train , y_train):
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy")#, random_state = 100, 
         #   max_depth = 50, min_samples_leaf = 50) 
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy

def prediction(X_test, clf_object): 
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    return y_pred 
def ouput(traingdata ,  realdata ):
    #real data 
    print ("real data")
    print(realdata)

    print ("traing data")
    print(traingdata)

    print('accuracy ')
    print(100*metrics.accuracy_score(realdata,traingdata))
    
    print('evaluation system')
    
    print('Precision of ')
    print('%.2f'% (metrics.precision_score(realdata, traingdata)) )
    
    print('Recall')
    print('%.2f'% (metrics.recall_score(realdata, traingdata)))

    print('F1 score')
    print('%2f'% (metrics.f1_score(realdata,traingdata)))