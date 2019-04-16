import numpy as np
def rescaling(data):
    result = np.random.randn(len(data),5)
    for i in range(0,len(data)):
        maxn = np.max(data[i])
        minn = np.min(data[i])
        result[i] = (data[i]-minn)/(maxn-minn)
    return result
def unit_length(data , t):
    for i in range(0,len(data)):
        data[i] = data[i]/t[i] 
    return data
def standardization(data,t,st):
    for i in range(data.shape[1]-1):
        data[:,i] = (data[:,i] - t[i] ) / st[i] 
    return data
