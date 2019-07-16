import numpy as np
def eclidean_distance(x,y):
    return  np.sqrt( (x[0] - y[0])**2 + (x[1] - y[1])**2 )