import numpy as np

def sigmoid(X, weights):
    return (1 / (1 + np.exp(-X.dot(weights))))
