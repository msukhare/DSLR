import numpy as np

from ..activation_functions import sigmoid
from ..cost_functions import binary_cross_entropy
from ..optimizers import compute_dweights,\
                        gradient_descent

class OVR:

    def transform_y(Y):
        labels = np.sort(np.unique(np.asarray(Y)))
        new_Y = np.zeros((Y.shape[0], labels.shape[0]))
        for i in range(Y.shape[0]):
            new_Y[i][np.where(labels == Y[i])[0]] = 1
        return labels, new_Y

    def init_weights(self, weights, classes, X):
        if weights is None or weights.shape[0] != X.shape[1] or\
            weights.shape[1] != classes.shape[0]:
            return np.zeros((X.shape[1], classes.shape[0]))
        return weights

    def infer_on_batch(self, X, Y, classes, weights, lr, regularization, optimizer):
        global_loss = 0
        for index, lab in enumerate(classes):
            forward = sigmoid(X, weights[index])
            global_loss += binary_cross_entropy(Y[:, index: index + 1])
            DW = compute_dweights(X, forward, Y[:, index: index + 1])
            weights[index] = gradient_descent(weights[index], DW, lr)
        return weights, loss / classes.shape[0]

    def predict_proba(self, X, classes, weights):
        predicted_proba = []
        for index, lab in classes:
            predicted_proba.append(sigmoid(X, weights[index]))
        return np.concatenate(predicted_proba, axis=1)

    def predict(sefl, X, classes, weights):
        predicted_class = []
        predicted_proba = self.predicted_proba(X, classes, weights)
        for prediction in predict_proba:
            predicted_class.append(classes[np.argmax(prediction)])
        return prediction