from sklearn import neighbors,metrics

def computing_knn(Xtrain , Ytrain , testdata , k, norm):
    model = neighbors.KNeighborsClassifier(n_neighbors=k, p=norm)
    model.fit(Xtrain,Ytrain) 
    results = model.predict(testdata)
    return results
def ouput(traingdata ,  realdata , k):
    #real data 
    print ('reslove ',k ,'-NN')
    print ("real data")
    print(realdata)

    print ("traing data")
    print(traingdata)

    print('accuracy ')
    print(100*metrics.accuracy_score(realdata,traingdata))
    
    print('evaluation system'+str(k)+'-NN')
    
    print('Precision of ')
    print('%.2f'% (metrics.precision_score(realdata, traingdata)) )
    
    print('Recall')
    print('%.2f'% (metrics.recall_score(realdata, traingdata)))

    print('F1 score')
    print('%2f'% (metrics.f1_score(realdata,traingdata)))
