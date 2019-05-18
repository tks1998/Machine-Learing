"""
    def score : calculate accuracy , F1-score , precision , recall
    def output :  print kq
"""
from sklearn import metrics

def score(train_label , real_label ):
    accuracy    =  round(metrics.accuracy_score(real_label, train_label),2)
    precision   =  round(metrics.precision_score(real_label, train_label),2)
    recall      =  round(metrics.recall_score(real_label, train_label),2)
    f1score     =  round(metrics.f1_score(real_label, train_label),2)
    return {'accuracy' : accuracy ,
            'precision': precision,
            'recall'   : recall,
            'f1score'  : f1score }

def output(type,result):
    print('processing data with ' , type )
    print('accuracy    ','precision    ','recall    ','f1score    ')
    print(result['accuracy'],'        ',result['precision'],'         ',result['recall'],'       ',result['f1score'])    