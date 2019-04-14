import numpy as np
def scaling(data , t):
    for i in range(0,len(data)):
        data[i] = data[i]/t[i] ; 
    return data
def scaling2(data,t,st):
    for i in range(data.shape[1]-1):
        data[:,i] = (data[:,i] - t[i] ) / st[i] ;  
    
    return data

