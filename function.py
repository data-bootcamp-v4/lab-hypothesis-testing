import numpy as np

def euclidean_distance(lon1, lat1, lon2, lat2):
    return np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)