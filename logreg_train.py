# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logreg_train.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/07/10 09:51:06 by msukhare          #+#    #+#              #
#    Updated: 2018/07/22 15:51:52 by kemar            ###   ########.fr        #
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

# #---> Algo WITHOUT VECTORIZATION

def read_file():
    try:
        data = pd.read_csv(sys.argv[1])
    except:
        sys.exit("File doesn't exist")
    data.drop(['First Name'], axis = 1, inplace = True)
    data.drop(['Last Name'], axis = 1, inplace = True)
    data.drop(['Birthday'], axis = 1, inplace = True)
    data['Best Hand'] = data['Best Hand'].map({'Left' : 0, 'Right': 1})
    data.drop(['Index'], axis = 1, inplace = True)
    data['Hogwarts House'] = data['Hogwarts House'].map({'Ravenclaw' : 3, 'Slytherin': 2, 'Gryffindor' : 1, 'Hufflepuff' : 4})
  #  for key in data:
   #     values = {key : data[key].quantile(0.50)}
   #     data.fillna(value=values, inplace=True)
    data.dropna(inplace=True)
    data = data.reset_index()
    data.drop(['index'], axis = 1, inplace = True)
    Y = data.iloc[:, 0:1]
    Y = np.array(Y.values, dtype=float)
    data.drop(['Hogwarts House'], axis=1, inplace=True)
    return (data, Y)

def scale_feature(data):
    desc = data.describe()
    X_scale = np.zeros((data.shape[1], data.shape[0]), dtype=float)
    i = 0
    for key in data:
        for j in range(int(desc[key]['count'])):
            X_scale[i][j] = (data[key][j] - desc[key]['mean']) / desc[key]['std']#(desc[key]['max'] - desc[key]['min'])
        i += 1
    return (X_scale)

def hypo(X, i, thetas, th, bias):
    return (1 / (1 + np.exp(-thetas[th].dot(X[i]) + bias)))

def g(X, thetas, th, bias):
    tmp = np.reshape(thetas[th], (14, 1))
    return (1 / (1 + np.exp(-X.dot(tmp) + bias)))

def get_new_y(Y, th, row):
    new_Y = np.zeros((row, Y.shape[1]), dtype=float)
    for i in range(int(row)):
        if (Y[i] == (th + 1)):
            new_Y[i][0] = 1
    return (new_Y)

def get_cost(X, thetas, hs, Y, bias):
    return (-Y.transpose().dot(np.log(g(X, thetas, hs, bias))) - (1 - Y).transpose().dot(np.log(1 - g(X, thetas, hs, bias))))

def cost_function(X, Y, thetas, nb_theta, bias):
    row = X.shape[0]
    ret = []
    for hs in range(int(nb_theta)):
        new_y = get_new_y(Y, hs, row)
        tmp = ((1 / row) * get_cost(X, thetas, hs, new_y, bias))
        ret.append(tmp[0][0])
    return (ret)

#def cost_function(X, Y, thetas, nb_theta):
#    th = 0
#    row = X.shape[0]
#    results = []
#    for th in range(int(nb_theta)):
#        i = 0
#        res = 0
#        while (i < row):
#            if (Y[i] == (th + 1)):
#                res += -np.log(hypo(X, i, thetas, th))
#            else:
#                res += -np.log((1 - hypo(X, i, thetas, th)))
#            i += 1
#        results.append(-(1 / row) * res)
#    return (results)

#def get_somme(X, Y, j, thetas, th, bias):
#    res = 0
 #   row = X.shape[0]
  #  for i in range(int(row)):
   #     if (Y[i][0] == (th + 1)):
   #         res += ((hypo(X, i, thetas, th, bias) - 1) * X[i][j])
    #    else:
     #       res += ((hypo(X, i, thetas, bias) - 0) * X[i][j])
   # return (((0.03 / row) * res))

def get_somme(X, Y, thetas, th, bias):
    res = 0
    row = X.shape[0]
    for i in range(int(row)):
        if (Y[i][0] == (th + 1)):
            res += ((hypo(X, i, thetas, th, bias) - 1))
        else:
            res += ((hypo(X, i, thetas, th, bias) - 0))
    return (((0.06 / row) * res))

def gradient_descent(X, Y, thetas, nb_theta, bias):
    th = 0
    row = X.shape[0]
    for th in range(int(nb_theta)):
        tmp_t = np.reshape(thetas[th], (14, 1))
        tmp_Y = get_new_y(Y, th, row)
        tmp = tmp_t - (0.06 / row) * (X.transpose().dot((g(X, thetas, th, bias) - tmp_Y)))
        bias = bias - get_somme(X, Y, thetas, th, bias)
        size = tmp.shape[0]
        for i in range(int(size)):
            thetas[th][i] = tmp[i][0]

#def gradient_descent(X, Y, thetas, tmp_thetas, nb_theta):
   # th = 0
   # for th in range(int(nb_theta)):
   #     col = thetas.shape[1]
   #     for j in range(int(col)):
   #         tmp_thetas[th][j] = thetas[th][j] - get_somme(X, Y, j, thetas, th)
   # for th in range(int(nb_theta)):
   #     col = thetas.shape[1]
   #     for j in range(int(col)):
   #         thetas[th][j] = tmp_thetas[th][j]

def get_quality_theta(X, Y, thetas, nb_theta, bias):
    row = X.shape[0]
    precision = 0
    recall = 0
    tm = np.zeros((Y.shape[0], Y.shape[1]), dtype=float)
    print(Y.shape[0], tm.shape[0], Y.shape[1], tm.shape[1])
    for i in range(int(row)):
        for hs in range(int(nb_theta)):
            t = hypo(X, i, thetas, hs, bias)
            if (t >= 0.5):
                tm[i] = (hs + 1)
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

def make_predi(X, Y, thetas, nb_theta):
    #cost_res = []
    #cost_res2 = []
    #cost_res1 = []
    #cost_res2 = []
    #cost_res3 = []
    #cost_res4 = []
    i = 0
    #index = []
    row = X.shape[0]
    bias = 0.0
    X_train, X_cost = X[ : floor(row * 0.85)], X[floor(row * 0.85) :]
    Y_train, Y_cost = Y[ : floor(row * 0.85)], Y[floor(row * 0.85) :]
    #X_train, X_cost, X_test = X[ : floor(row * 0.85)], X[floor(row * 0.85) :]
    #Y_train, Y_cost, Y_test = Y[ : floor(row * 0.70)], Y[floor(row * 0.70) : floor(row * 0.85)], Y[floor(row * 0.85) :]
    for i in range(5000):#815
        #tmp = cost_function(X_cost, Y_cost, thetas, nb_theta, bias)
        #cost_res.append((tmp[0] + tmp[1] + tmp[2] + tmp[3]) / 4)
        #tmp = cost_function(X_train, Y_train, thetas, nb_theta, bias)
        #cost_res2.append((tmp[0] + tmp[1] + tmp[2] + tmp[3]) / 4)
        #cost_res1.append(tmp[0])
        #cost_res2.append(tmp[1])
       # cost_res3.append(tmp[2])
        #cost_res4.append(tmp[3])
        #index.append(i)
        gradient_descent(X_train, Y_train, thetas, nb_theta, bias)
    #plt.plot(index, cost_res, color='red')
    #plt.plot(index, cost_res2, color='green')
   # plt.plot(index, cost_res1)
   # plt.plot(index, cost_res2)
   # plt.plot(index, cost_res3)
   # plt.plot(index, cost_res4)
    #plt.show()
    get_quality_theta(X_cost, Y_cost, thetas, nb_theta, bias)

def main():
    if (len(sys.argv) <= 1):
        sys.exit("No name file")
    if (len(sys.argv) >= 3):
        sys.exit("too much file")
    data, Y = read_file()
    X_scale = scale_feature(data).transpose()
    #X_scale = np.c_[np.ones(X_scale.shape[0]), X_scale]
    #tmp_mat = np.copy(X_scale)
    #row = tmp_mat.shape[0]
    #col = tmp_mat.shape[1]
    #for i in range(int(row)):
     #   for j in range(int(col)):
     #       tmp_mat[i][j] = tmp_mat[i][j]**2
    #X_scale = np.c_[X_scale, tmp_mat]
    nb_theta = max(Y)
    thetas = np.zeros((int(nb_theta), X_scale.shape[1]), dtype=float)
#    tmp_thetas = np.zeros((int(nb_theta), data.shape[1]), dtype=float)
    make_predi(X_scale, Y, thetas, nb_theta)

if __name__ == "__main__":
    main()
