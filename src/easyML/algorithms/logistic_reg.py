import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from .kernels import KERNELS
from ..data_managment import split_data
from .metrics import accuracy_score,\
                        precision_score,\
                        recall_score,\
                        f1_score

class LogisticReg:

    def __init__(self,\
                kernel='OVR',\
                optimizer='gradient_descent',\
                l2=False,\
                lambda_value=0.01,\
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
                f1_score=False,\
                average='macro'):
        if kernel not in KERNELS.keys():
            raise Exception("%s is not a valide kernel" %kernel)
        self.kernel = KERNELS[kernel]
        self.optimizer = optimizer
        self.l2 = l2
        self.lambda_value = lambda_value
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
        self.f1_score = f1_score
        self.average = average
        self.weights = None
        self.classes = None

    def predict_proba(self, X):
        pass

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return self.kernel.predict(X, self.classes, self.weights)

    def get_batch(self, X, Y, batch_size):
        i = 0
        while (i < X.shape[0]):
            yield X[i: i + batch_size], Y[i: i + batch_size]
            i += batch_size
        if i < X.shape[0]:
            yield X[i:], Y[i:]

    def evaluate_training(self, X, Y):
        global_loss = 0
        iter_ = 0
        confusion_matrix = np.zeros((self.classes.shape[0], self.classes.shape[0]))
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = X.shape[0]
        for X_batch, Y_batch in self.get_batch(X, Y, batch_size):
            predicted_y, loss = self.kernel.eval_on_batch(X_batch,\
                                                        Y_batch,\
                                                        self.classes,\
                                                        self.weights,\
                                                        self.l2,\
                                                        self.lambda_value)
            global_loss += loss
            iter_ += 1
            for i, ele in enumerate(predicted_y):
                confusion_matrix[np.argmax(Y_batch[i])][ele] += 1
        return confusion_matrix, global_loss / iter_

    def fit(self, X, Y):
        best_loss_val = 0
        batch_size_train = self.batch_size
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.classes, Y = self.kernel.transform_y(Y)
        self.weights = self.kernel.init_weights(self.weights, self.classes, X)
        if self.validate is True or self.early_stopping is True:
            X_train, X_val, Y_train, Y_val = split_data(X, Y, self.validation_fraction)
            #confusion_matrix, best_loss_val = self.evaluate_training(X_val, Y_val)
        else:
            X_train, Y_train = X, Y
        if self.batch_size is None:
            batch_size_train = X.shape[0]
        nb_epochs_waiting = 0
        for epoch in range(self.epochs):
            global_loss = 0
            training_process = ""
            iter_ = 0
            for X_batch, Y_batch in self.get_batch(X_train, Y_train, batch_size_train):
                self.weights, loss = self.kernel.infer_on_batch(X_batch,\
                                                                Y_batch,\
                                                                self.classes,\
                                                                self.weights,\
                                                                self.lr,\
                                                                self.optimizer,\
                                                                self.l2,\
                                                                self.lambda_value)
                global_loss += loss
                iter_ += 1
            training_process += "epoch %d/%d epochs loss train is equal to %f" %(epoch, self.epochs, global_loss / iter_)
            if self.validate is True or self.early_stopping is True:
                confusion_matrix, loss_val = self.evaluate_training(X_val, Y_val)
                if self.early_stopping is True:
                    if loss_val > best_loss_val - self.tol:
                        nb_epochs_waiting += 1
                    else:
                        nb_epochs_waiting = 0
                    if nb_epochs_waiting >= self.n_epochs_no_change:
                        return
                    best_loss_val = min(loss_val, best_loss_val)
                training_process += "; loss val is equal to %f" %(loss_val)
                if self.accuracy is True:
                    training_process += "; val accuracy is equal to %f" %(accuracy_score(confusion_matrix))
                if self.precision is True:
                    training_process += "; val precision is equal to %f" %(precision_score(confusion_matrix,\
                                                                                            self.average))
                if self.recall is True:
                    training_process += "; val recall is equal to %f" %(recall_score(confusion_matrix,\
                                                                                    self.average))
                if self.f1_score is True:
                    training_process += "; val f1_score is equal to %f" %(f1_score(confusion_matrix,\
                                                                                    self.average))
            print(training_process)

    def features_importance(self, labels_names, features_names, number_features_max):
        weights = np.abs(self.weights)
        for key in labels_names.keys():
            values, feats = [], []
            for value, feat in sorted(list(zip(weights[labels_names[key]], features_names)), key=lambda tup: tup[0], reverse=True)[0: number_features_max]:
                values.append(value)
                feats.append(feat)
            plt.bar(feats, values, width=0.5)
            plt.ylabel("magnitude")
            plt.xlabel("features")
            plt.title("features importance for %s" %key)
            plt.show()

    def save_weights(self, path_to_where_save, params_scaling, labels):
        pass

    def load_weights(self, path_to_weights):
        pass