# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logreg_train.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/07/10 09:51:06 by msukhare          #+#    #+#              #
#    Updated: 2018/07/13 13:29:29 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# #---> Algo WITHOUT VECTORIZATION

def read_file():
    try:
        data = pd.read_csv(sys.argv[1])
    except:
        sys.exit("File doesn't exist")
    data.drop(['First Name'], axis = 1, inplace = True)
    data.drop(['Last Name'], axis = 1, inplace = True)
    data.drop(['Birthday'], axis = 1, inplace = True)
    data.drop(['Index'], axis = 1, inplace = True)
    data['Best Hand'] = data['Best Hand'].map({'Left' : 0, 'Right': 1})
    data['Hogwarts House'] = data['Hogwarts House'].map({'Ravenclaw' : 3, 'Slytherin': 2, 'Gryffindor' : 1, 'Hufflepuff' : 4})
    for key in data:
        values = {key : data[key].quantile(0.50)}
        data.fillna(value=values, inplace=True)
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
            X_scale[i][j] = data[key][j] / (desc[key]['max'] - desc[key]['min'])
        i += 1
    return (X_scale)

#def hypo(X, i, thetas, th):
#    return (1 / (1 + np.exp(-thetas[th].dot(X[i]))))

def g(X, thetas, th):
    tmp = np.reshape(thetas[th], (14, 1))
    return (1 / (1 + np.e**-(X.dot(tmp))))

def get_new_y(Y, th, row):
    new_Y = np.zeros((row, Y.shape[1]), dtype=float)
    for i in range(int(row)):
        if (Y[i] == (th + 1)):
            new_Y[i][0] = 1
    return (new_Y)

def get_cost(X, thetas, hs, Y):
    return (-Y.transpose().dot(np.log(g(X, thetas, hs))) - (1 - Y).transpose().dot(np.log(1 - g(X, thetas, hs))))

def cost_function(X, Y, thetas, nb_theta):
    row = X.shape[0]
    ret = []
    for hs in range(int(nb_theta)):
        new_y = get_new_y(Y, hs, row)
        tmp = ((1 / row) * get_cost(X, thetas, hs, new_y))
        ret.append(tmp[0][0])
    return (ret)
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

#def get_somme(X, Y, j, thetas, th):
#    res = 0
#    row = X.shape[0]
#    for i in range(int(row)):
#        if (Y[i][0] == (th + 1)):
#            res += ((hypo(X, i, thetas, th) - 1) * X[i][j])
#        else:
#            res += ((hypo(X, i, thetas, th) - 0) * X[i][j])
#    return (((0.03 / row) * res))

def gradient_descent(X, Y, thetas, tmp_thetas, nb_theta):
    th = 0
    row = X.shape[0]
    for th in range(int(nb_theta)):
        tmp_t = np.reshape(thetas[th], (14, 1))
        tmp_Y = get_new_y(Y, th, row)
        tmp = tmp_t - (0.1 / row) * (X.transpose().dot((g(X, thetas, th) - tmp_Y)))
        size = tmp.shape[0]
        for i in range(int(size)):
            thetas[th][i] = tmp[i][0]
   # th = 0
   # for th in range(int(nb_theta)):
   #     col = thetas.shape[1]
   #     for j in range(int(col)):
   #         tmp_thetas[th][j] = thetas[th][j] - get_somme(X, Y, j, thetas, th)
   # for th in range(int(nb_theta)):
   #     col = thetas.shape[1]
   #     for j in range(int(col)):
   #         thetas[th][j] = tmp_thetas[th][j]

def make_predi(X, Y, thetas, tmp_thetas, nb_theta):
    cost_res = []
    i = 0
    index = []
    for i in range(200):
     #   tmp = cost_function(X, Y, thetas, nb_theta)
    #    cost_res.append((tmp[0] + tmp[1] + tmp[2] + tmp[3]) / 4)
        #index.append(i)
        gradient_descent(X, Y, thetas, tmp_thetas, nb_theta)
    #plt.plot(index, cost_res)
    #plt.show()
    print(thetas)

def main():
    if (len(sys.argv) <= 1):
        sys.exit("No name file")
    if (len(sys.argv) >= 3):
        sys.exit("too much file")
    data, Y = read_file()
    X_scale = scale_feature(data).transpose()
    nb_theta = max(Y)
    thetas = np.zeros((int(nb_theta), data.shape[1]), dtype=float)
    tmp_thetas = np.zeros((int(nb_theta), data.shape[1]), dtype=float)
    make_predi(X_scale, Y, thetas, tmp_thetas, nb_theta)

if __name__ == "__main__":
    main()
