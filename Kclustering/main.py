import process
import os
from PIL import Image
import numpy as np

#process.ResizeAndConvertToBitmap()
p=[]
process.prepare(p) 
# path = "/home/sen/Desktop/hoc tap/Machine-Learing/Kclustering/Folio Leaf Dataset/Folio"
# dem = 0
# number = 0
# label = []
# distance_all = []
# distance = []
# while (number<1):
#     dem = 0 
#     key = 0
#     number = number + 1
#     p = np.random.randint(low=1, high=637, size=32)
#     vector_label = process.prepare(p)
#     for f in os.listdir(path):
#         key = key + 1 
#         for element in os.listdir(os.path.join(path,f)):
#             dem = dem + 1
#             print 
#             label.append(key) 
#             path1 = path+"/"+f+"/"+element
#             new_img = Image.open(path1)
#             img = new_img.resize((520,520))
#             arr = np.array(img).reshape(-1)
#             for i in range(0,31):
#                 if dem !=i:
#                     distance.append(process.distance(arr,vector_label[i]))
#             distance_all.append(distance)



# print(distance_all[1])
    
        