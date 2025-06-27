import numpy as np

def numpy_sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s 

print(numpy_sigmoid(2))
