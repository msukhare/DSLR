# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logreg_train.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/07/10 09:51:06 by msukhare          #+#    #+#              #
#    Updated: 2018/11/01 19:49:05 by kemar            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from math import floor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from read_and_complete_data import read_file

def hypo(X, i, thetas, th):
    return (1 / (1 + np.exp(-thetas[th].dot(X[i]))))

def g(X, thetas, th):
    tmp = np.reshape(thetas[th], (X.shape[1], 1))
    return (1 / (1 + np.exp(-X.dot(tmp))))

def get_new_y(Y, th, row):
    new_Y = np.zeros((row, Y.shape[1]), dtype=float)
    for i in range(int(row)):
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
        new_y = get_new_y(Y, hs, row)
        tmp = ((1 / row) * get_cost(X, thetas, hs, new_y))
        ret.append(tmp[0][0])
    return (ret)

def gradient_descent(X, Y, thetas, nb_theta):
    th = 0
    row = X.shape[0]
    for th in range(int(nb_theta)):
        tmp_t = np.reshape(thetas[th], (X.shape[1], 1))
        tmp_Y = get_new_y(Y, th, row)
        tmp = tmp_t - (0.06 / row) * (X.transpose().dot((g(X, thetas, th) - tmp_Y)))
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

def get_quality_classifier(X, Y, thetas, nb_theta):
    row = X.shape[0]
    precision = 0
    recall = 0
    tm = np.zeros((Y.shape[0], Y.shape[1]), dtype=float)
    print(Y.shape[0], tm.shape[0], Y.shape[1], tm.shape[1])
    for i in range(int(row)):
        all_res = []
        for hs in range(int(nb_theta)):
            all_res.append(float(hypo(X, i, thetas, hs)))
        tm[i] = (get_index_max(all_res) + 1)
    pre = 0
    accu = 0
    for hs in range(int(nb_theta)):
        vp = 0
        fp = 0
        fn = 0
        vn = 0
        true_y = get_new_y(Y, hs, row)
        pred_y = get_new_y(tm, hs, row)
        pre += precision_score(true_y, pred_y)
        accu += accuracy_score(true_y, pred_y)
        print(accuracy_score(true_y, pred_y))
        for i in range(int(row)):
            if (true_y[i] == 1 and pred_y[i] == 1):
                vp += 1
            elif (true_y[i] == 0 and pred_y[i] == 1):
                fp += 1
            elif (true_y[i] == 1 and pred_y[i] == 0):
                fn += 1
            elif (true_y[i] == 0 and pred_y[i] == 0):
                vn += 1
        tn, fp1, fn1, tp = confusion_matrix(true_y, pred_y).ravel()
        print(tn, fp1, fn1, tp)
        print(vn, fp, fn, vp)
        precision += (vp / (vp + fp))
        recall += (vp / (vp + fn))
  #  print(tm, len(tm), Y, Y.shape[0])
    precision = ((1 / nb_theta) * precision)
    recall = ((1 / nb_theta) * recall)
    print("precision :", precision)
    print("recall :", recall)
    print("f1 :", ((2 * (precision * recall)) / (precision + recall)))
    print(((1 / nb_theta) * pre))
    print(((1 / nb_theta) * accu))
    print(accuracy_score(Y, tm))  

def train_thetas(X, Y, thetas, nb_theta):
    cost_res = []
    cost_res2 = []
    index = []
    row = X.shape[0]
    X_train, X_test, X_val = X[ : floor(row * 0.70)], X[floor(row * 0.70) : floor(row * 0.90)],\
            X[floor(row * 0.90) :]
    Y_train, Y_test, Y_val = Y[ : floor(row * 0.70)], Y[floor(row * 0.70) : floor(row * 0.90)],\
            Y[floor(row * 0.90) :]
    for i in range(2000):#815
        tmp = cost_function(X_test, Y_test, thetas, nb_theta)
        cost_res.append((tmp[0] + tmp[1] + tmp[2] + tmp[3]) / 4)
        tmp = cost_function(X_train, Y_train, thetas, nb_theta)
        cost_res2.append((tmp[0] + tmp[1] + tmp[2] + tmp[3]) / 4)
        index.append(i)
        gradient_descent(X_train, Y_train, thetas, nb_theta)
    plt.plot(index, cost_res, color='red')
    plt.plot(index, cost_res2, color='green')
    plt.show()
    get_quality_classifier(X_val, Y_val, thetas, nb_theta)

def main():
    X, Y = read_file("train")
    nb_theta = max(Y)
    X = np.c_[np.ones((X.shape[0], 1), dtype=float), X]
    thetas = np.zeros((int(nb_theta), X.shape[1]), dtype=float)
    train_thetas(X, Y, thetas, nb_theta)

if __name__ == "__main__":
    main()

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
