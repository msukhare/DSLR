import numpy as np
import pickle
import os

from .optimizers import gradient_descent, compute_dweights
from .activation_functions import sigmoid
from .cost_functions import binary_cross_entropy

def transform_dummy_var(Y):
    classes = np.unique(Y)
    to_ret = np.zeros((Y.shape[0], classes.shape[0]))
    for i, ele in enumerate(classes):
        idx_set_to_one = np.where(Y == ele)[0]
        to_ret[idx_set_to_one, i] = 1
    return to_ret, classes

class LogisticReg:

    def __init__(self,\
                lr=0.01,\
                epochs=100,\
                batch=None,\
                show_training=False):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.batch = batch
        self.classes = None

    def fit(self, X, Y):
        Y, self.classes = transform_dummy_var(Y)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        if self.weights is None:
            self.weights = np.zeros((X.shape[1], Y.shape[1]))
        for epoch in range(self.epochs):
            forward_res = sigmoid(X, self.weights)
            dweights = compute_dweights(X, forward_res, Y)
            self.weights = gradient_descent(self.weights, dweights, self.lr)
            train_loss = 0
            for i in range(self.classes.shape[0]):
                train_loss += binary_cross_entropy(Y[:, i], forward_res[:, i])
            train_loss /= self.classes.shape[0]
        print("training 2:", train_loss)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return sigmoid(X, self.weights)

    def eval(self, X, Y):
        pass

    def features_importance(self, columns):
        pass

    def save_weights(self, path_to_where_save):
        pass

    def load_weights(self, path_to_weights):
        pass
