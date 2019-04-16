from sklearn import metrics

def ouput(traingdata ,  realdata ):
    #real data 
    print ("real data")
    print(realdata)

    print ("traing data")
    print(traingdata)
    
    print('Accuracy       Precision          Recall       F1 score ')

    print('%.2f'% (100*metrics.accuracy_score(realdata,traingdata)),'         '
          '%.2f'% (metrics.precision_score(realdata, traingdata)),'               '
          '%.2f'% (metrics.recall_score(realdata, traingdata)), '       '
          '%.2f'% (metrics.f1_score(realdata,traingdata)))