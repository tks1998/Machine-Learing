import os
from PIL import Image
import numpy as np
from numba import jit
import pandas as pd
import csv
path = "/home/sen/Desktop/hoc tap/Machine-Learing/Kclustering/Folio Leaf Dataset/Folio"
def ResizeAndConvertToBitmap():
    dem = 0
    with open('data.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow("0")
        for f in os.listdir(path):
            for element in os.listdir(os.path.join(path,f)):
                path1 = path+"/"+f+"/"+element
                new_img = Image.open(path1)
                img = new_img.resize((520,520))
                arr = np.array(img).reshape(-1)
                dem =  dem + 1 
                print(dem)
                writer.writerow([arr])
    return 

def prepare(p):
    A = pd.read_csv('data.csv')
    #print(A.iloc[0]) 
    dem = 0
    vector_label = []
    for i in range(0,len(A)):
        if i in p:
            dem = dem + 1 
            vector_label.append(i)
    return vector_label
def distance(point1 , point2):
    return np.sum((point1-point2)*(point1-point2))