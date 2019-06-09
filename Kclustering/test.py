import numpy as np 

p = np.random.randint(low=1, high=637, size=32)
for i in range(0,300):
    if (i in p):
        print(i)