import numpy as np
import pickle
import os

from .kernels import KERNELS
#from .metrics import accuracy_score,\
#                        precision_score,\
#                        recall_score,\
#                        f1_score

class LogisticReg:

    def __init__(self,\
                kernel='OVR',\
                optimizer='gradient_descent',\
                regularization=None,\
                lr=0.01,\
                epochs=100,\
                batch_size=None,\
                early_stopping=False,\
                validation_fraction=0.10,\
                n_epochs_no_change=5,\
                tol=1e-3,\
                validate=False,\
                accuracy=False,\
                precision=False,\
                recall=False,\
                f1=False):
        if kernel not in KERNELS.keys():
            raise Exception("%s is not a valide kernel")
        self.kernel = KERNELS[kernel]
        self.optimizer = optimizer
        self.regularization = regularization
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_epochs_no_change = n_epochs_no_change
        self.tol = tol
        self.validate = validate
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.weights = None
        self.classes = None

    def predict_proba(self, X):
        pass

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return self.kernel.predict(X, self.classes, self.weights)

    def eval(self, X, Y):
        pass

    def get_batch(self, X, Y):
        i = 0
        while (i < X.shape[0]):
            yield X[i: i + self.batch_size], Y[i: i + self.batch_size]
            i += self.batch_size
        if i < X.shape[0]:
            yield X[i:], Y[i:]

    def fit(self, X, Y):
        self.classes, Y = self.kernel.transform_y(Y)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        if self.batch_size is None:
            self.batch_size = X.shape[0]
        self.weights = self.kernel.init_weights(self.weights, self.classes, X)
        for epoch in range(self.epochs):
            global_loss = 0
            training_process = ""
            iter_ = 0
            for X_batch, Y_batch in self.get_batch(X, Y):
                self.weights, loss = self.kernel.infer_on_batch(X_batch,\
                                                                Y_batch,\
                                                                self.classes,\
                                                                self.weights,\
                                                                self.lr,\
                                                                self.regularization,\
                                                                self.optimizer)
                global_loss += loss
                iter_ += 1
            training_process += "%d/%d loss train is equal to %f" %(epoch, self.epochs, global_loss / iter_)
            if self.validate is True:
                training_process += "; loss val is equal to %f" %(loss_val)
                if self.accuracy is True:
                    training_process += "; val accuracy is equal to %f" %(accuracy_score(predictions_val, Y_val))
                if self.precision is True:
                    training_process += "; val precision is equal to %f" %(precision_score(predictions_val, Y_val))
                if self.recall is True:
                    training_process += "; val recall is equal to %f" %(recall_score(predictions_val, Y_val))
                if self.f1_score is True:
                    training_process += "; val f1_score is equal to %f" %(f1_score(predictions_val, Y_val))
            print(training_process)

    def features_importance(self, columns):
        pass

    def save_weights(self, path_to_where_save):
        pass

    def load_weights(self, path_to_weights):
        pass
