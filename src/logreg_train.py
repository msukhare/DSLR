import os
import sys
import argparse

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from math import floor
#from read_and_complete_data import read_file
import csv

from easyML import LogisticReg

def hypo(X, i, thetas, th):
    return (1 / (1 + np.exp(-thetas[th].dot(X[i]))))

def g(X, thetas, th):
    tmp = np.reshape(thetas[th], (X.shape[1], 1))
    return (1 / (1 + np.exp(-X.dot(tmp))))

def get_new_y(Y, th):
    new_Y = np.zeros((Y.shape[0], Y.shape[1]), dtype=float)
    for i in range(int(Y.shape[0])):
        if (Y[i] == (th + 1)):
            new_Y[i][0] = 1
    return (new_Y)

def get_cost(X, thetas, hs, Y):
    return (-Y.transpose().dot(np.log(g(X, thetas, hs))) - (1 - Y).transpose().dot(np.log(1 -\
            g(X, thetas, hs))))

def cost_function(X, Y, thetas, nb_theta):
    row = X.shape[0]
    ret = []
    for hs in range(int(nb_theta)):
        new_y = get_new_y(Y, hs)
        tmp = ((1 / row) * get_cost(X, thetas, hs, new_y))
        ret.append(tmp[0][0])
    return (ret)

def gradient_descent(X, Y, thetas, nb_theta):
    row = X.shape[0]
    for th in range(int(nb_theta)):
        tmp_t = np.reshape(thetas[th], (X.shape[1], 1))
        tmp_Y = get_new_y(Y, th)
        tmp = tmp_t - (0.01 / row) * (X.transpose().dot((g(X, thetas, th) - tmp_Y)))
        size = tmp.shape[0]
        for i in range(int(size)):
            thetas[th][i] = tmp[i][0]

def get_index_max(all_res):
    max = all_res[0]
    ind = 0
    for i in range(len(all_res)):
        if (all_res[i] >= 0.5 and max < all_res[i]):
            max = all_res[i]
            ind = i
    return (ind)

def predict_Y(X, thetas, nb_theta):
    prediction = np.zeros((X.shape[0], 1), dtype=int)
    for i in range(X.shape[0]):
        all_res = []
        for hs in range(int(nb_theta)):
            all_res.append(float(hypo(X, i, thetas, hs)))
        prediction[i][0] = (get_index_max(all_res) + 1)
    return (prediction)

def get_accuracy(pred_Y, Y):
    nb_true = 0
    for i in range(pred_Y.shape[0]):
        if (pred_Y[i][0] == Y[i][0]):
            nb_true += 1
    return (float(nb_true / Y.shape[0]))

def get_quality_classifier(X, Y, thetas, nb_theta):
    pred_Y = predict_Y(X, thetas, nb_theta)
    precision = 0
    recall = 0
    for hs in range(int(nb_theta)):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        tmp_y = get_new_y(Y, hs)
        tmp_pred_y = get_new_y(pred_Y, hs)
        for i in range(tmp_y.shape[0]):
            if (tmp_y[i][0] == 1 and tmp_pred_y[i][0] == 1):
                tp += 1
            elif (tmp_y[i][0] == 0 and tmp_pred_y[i][0] == 1):
                fp += 1
            elif (tmp_y[i][0] == 1 and tmp_pred_y[i][0] == 0):
                fn += 1
            elif (tmp_y[i][0] == 0 and tmp_pred_y[i][0] == 0):
                tn += 1
        print("For the class:", (hs + 1))
        print("tp:", tp, ", fp:", fp, ", fn:", fn, ", tn:", tn)
        precision += (tp / (tp + fp))
        recall += (tp / (tp + fn))
    precision = precision / nb_theta
    recall = recall / nb_theta
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", ((2 * (precision * recall)) / (precision + recall)))
    print("accuracy:", get_accuracy(pred_Y, Y))

def train_thetas(X, Y, thetas, nb_theta):
    cost_res = []
    cost_res2 = []
    index = []
    row = X.shape[0]
    #X_train, X_test, X_val = X[ : floor(row * 0.70)], X[floor(row * 0.70) : floor(row * 0.90)],\
    #        X[floor(row * 0.90) :]
    #Y_train, Y_test, Y_val = Y[ : floor(row * 0.70)], Y[floor(row * 0.70) : floor(row * 0.90)],\
    #        Y[floor(row * 0.90) :]
    X_train = X
    Y_train = Y
    for i in range(100):#815
        #tmp = cost_function(X_test, Y_test, thetas, nb_theta)
        #cost_res.append((tmp[0] + tmp[1] + tmp[2] + tmp[3]) / 4)
        gradient_descent(X_train, Y_train, thetas, nb_theta)
    #tmp = cost_function(X_train, Y_train, thetas, nb_theta)
    #print("training: ", np.mean(np.asarray(tmp)))
    #cost_res2.append((tmp[0] + tmp[1] + tmp[2] + tmp[3]) / 4)
    #index.append(i)
    #plt.plot(index, cost_res, color='red')
    #plt.plot(index, cost_res2, color='green')
    #plt.show()
    #get_quality_classifier(X_val, Y_val, thetas, nb_theta)

def write_thetas_in_file(thetas):
    try:
        file = csv.writer(open("thetas.csv", "w"), delimiter=',',\
                quoting=csv.QUOTE_MINIMAL)
    except:
        sys.exit("fail to create thetas.csv")
    for i in range(thetas.shape[0]):
        file.writerow(thetas[i])

def check_argv():
    if (len(sys.argv) <= 1):
        sys.exit("Need file name with data")
    if (len(sys.argv) >= 3):
        sys.exit("too much arguments")
    if (sys.argv[1] != "dataset_train.csv"):
        sys.exit("file must be dataset_train.csv")


def main_2(cc):
    #check_argv()
    #X, Y = read_file("train", sys.argv[1], "dataset_test.csv")
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    Y = np.array([[1], [2], [3], [4]])
    nb_theta = max(Y)
    X = np.c_[np.ones((X.shape[0], 1), dtype=float), X]
    thetas = np.zeros((int(nb_theta), X.shape[1]), dtype=float)
    train_thetas(X, Y, thetas, nb_theta)
    write_thetas_in_file(thetas)

import os
import sys
import argparse

from easyML import LogisticReg,\
                    scaling_features,\
                    split_data
from utils import read_data_csv_cls

def main(args):
    try:
        X, Y = read_data_csv_cls(args.data_path, train=True)
    except Exception as error:
        sys.exit('Error: ' + str(error))
    X, params_to_save = scaling_features(X, None, args.type_of_features_scaling)
    classificator = LogisticReg(args.kernel,\
                                args.optimizer,\
                                args.regularization,\
                                args.learning_rate,\
                                args.epochs,\
                                args.batch_size,\
                                args.early_stopping,\
                                args.validation_fraction,\
                                args.n_epochs_no_change,\
                                args.tol,\
                                args.validate,\
                                args.accuracy,\
                                args.precision,\
                                args.recall,\
                                args.f1_score)

    classificator.fit(X, Y)

    try:
        classificator.fit(X, Y)
    except Exception as error:
        sys.exit('Error:' + str(error))
    if args.features_importance is True:
        pass
    classificator.save_weights(args.file_where_store_weights, params_to_save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',\
            nargs='?',\
            type=str,\
            help="""correspond to path of csv file""")
    parser.add_argument('--file_where_store_weights',\
            nargs='?',\
            type=str,\
            help="""correspond to path where store weights after training and
                    informations about pipeline""")
    parser.add_argument('--kernel',\
            nargs='?',\
            type=str,\
            default="OVR",\
            const="OVR",\
            choices=['OVR', 'OVO', 'MULTINOMIAL'],\
            help="""correspond to kernel to use during training.
                    By default OVR""")
    parser.add_argument('--optimizer',\
            nargs='?',\
            type=str,\
            default="gradient_descent",\
            const="gradient_descent",\
            choices=['gradient_descent', 'adam', 'momentum', 'RMSP'],\
            help="""correspond to optimizer to use.
                    By default gradient descent""")
    parser.add_argument('--regularization',\
            nargs='?',\
            type=str,\
            default=None,\
            const=None,\
            choices=['l1', 'l2', 'weight_decay'],\
            help="""correspond to regularization to use.
                    By default none regularization is done""")
    parser.add_argument('--type_of_features_scaling',\
            nargs='?',\
            type=str,\
            default="standardization",\
            const="standardization",\
            choices=['standardization', 'rescaling', 'normalization'],\
            help="""correspond to technic use for features scaling.
                    By default standardization""")
    parser.add_argument('--learning_rate',\
            nargs='?',\
            type=float,\
            default=0.1,\
            const=0.1,\
            help="""correspond to learning rate used during training.
                By default 0.1""")
    parser.add_argument('--epochs',\
            nargs='?',\
            type=int,\
            default=100,\
            const=100,\
            help="""correspond to numbers of epochs to do during training.
                By default 100""")
    parser.add_argument('--batch_size',\
            nargs='?',\
            type=int,\
            default=None,\
            const=None,\
            help="""correspond to numbers of sample to use for one iteration.
                By default None all samples are used during one iteration""")
    parser.add_argument('--early_stopping',\
            dest='early_stopping',\
            action='store_true',
            help="""if pass as params will do early stopping on val set, base on tol and
                n_epochs_no_change in gradient descent""")
    parser.add_argument('--validation_fraction',\
            nargs='?',\
            type=float,\
            default=0.10,\
            const=0.10,\
            help="""correspond to percentage data use during training as val set in gradient descent.
                Used if early_stopping is True or validate is set True.
                By default 0.10 percentage of data""")
    parser.add_argument('--n_epochs_no_change',\
            nargs='?',\
            type=int,\
            default=5,\
            const=5,\
            help="""correspond to numbers of epochs wait until cost function don't change.
                Only used in gradient descent and if --early_stoping is set at True.
                By default 5 epochs""")
    parser.add_argument('--tol',\
            nargs='?',\
            type=float,\
            default=1e-3,\
            const=1e-3,\
            help="""correspond to stopping criteron in early stopping.
                Only used in gradient descent and if --early_stopping is set at True.
                By default 1e-3""")
    parser.add_argument('--features_importance',\
            dest='features_importance',\
            action='store_true',
            help="""if pass as params will show features importance at the end of training""")
    parser.add_argument('--validate',\
            dest='validate',\
            action='store_true',
            help="""if pass as params will do evaluation on validation set during training,
                By default show only loss function, you can add other metrics""")
    parser.add_argument('--accuracy',\
            dest='accuracy',\
            action='store_true',
            help="""if pass as params will compute accuracy on validation set
                    validate must be pass as params to show accuracy""")
    parser.add_argument('--precision',\
            dest='precision',\
            action='store_true',
            help="""if pass as params will compute precision on validation set
                    validate must be pass as params to show precision""")
    parser.add_argument('--recall',\
            dest='recall',\
            action='store_true',
            help="""if pass as params will compute recall on validation set
                    validate must be pass as params to show recall""")
    parser.add_argument('--f1_score',\
            dest='f1_score',\
            action='store_true',
            help="""if pass as params will compute f1_score on vaslidation set
                    validate must be pass as params to show f1_score""")
    parsed_args = parser.parse_args()
    if parsed_args.data_path is None:
        sys.exit("Error: missing name of CSV data to use")
    if os.path.exists(parsed_args.data_path) is False:
        sys.exit("Error: %s doesn't exists" %parsed_args.data_path)
    if os.path.isfile(parsed_args.data_path) is False:
        sys.exit("Error: %s must be a file" %parsed_args.data_path)
    main(parsed_args)

""" Algo without Vectorization
def gradient_descent(X, Y, thetas, tmp_thetas, nb_theta):
    th = 0
    for th in range(int(nb_theta)):
        col = thetas.shape[1]
        for j in range(int(col)):
            tmp_thetas[th][j] = thetas[th][j] - get_somme(X, Y, j, thetas, th)
    for th in range(int(nb_theta)):
        col = thetas.shape[1]
        for j in range(int(col)):
            thetas[th][j] = tmp_thetas[th][j]

def cost_function(X, Y, thetas, nb_theta):
    th = 0
    row = X.shape[0]
    results = []
    for th in range(int(nb_theta)):
        i = 0
        res = 0
        while (i < row):
            if (Y[i] == (th + 1)):
                res += -np.log(hypo(X, i, thetas, th))
            else:
                res += -np.log((1 - hypo(X, i, thetas, th)))
            i += 1
        results.append(-(1 / row) * res)
    return (results)

def get_somme(X, Y, j, thetas, th, bias):
    res = 0
    row = X.shape[0]
    for i in range(int(row)):
        if (Y[i][0] == (th + 1)):
            res += ((hypo(X, i, thetas, th, bias) - 1) * X[i][j])
        else:
            res += ((hypo(X, i, thetas, bias) - 0) * X[i][j])
    return ((-(0.03 / row) * res))
"""
