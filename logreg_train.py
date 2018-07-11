# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logreg_train.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/07/10 09:51:06 by msukhare          #+#    #+#              #
#    Updated: 2018/07/11 17:10:46 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
import sys

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

def hypo(X, i, thetas, th):
    return (1 / (1 + np.exp(-thetas[th].dot(X[i]))))

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

def somme_for_grad(X, Y, thetas, th, feat):
    row = X.shape[0]
    res = 0
    for i in range(row):
        if (Y[i][0] == (th + 1)):
            res += ((hypo(X, i, thetas, th) - 1) * X[i][feat])
        else:
            res += ((hypo(X, i, thetas, th) - 0) * X[i][feat])
    return ((0.03 / row) * res)

def gradient_descent(X, Y, thetas, tmp_thetas, nb_theta):
    for th in range(int(nb_theta)):
        col = tmp_thetas.shape[1]
        for i in range(col):
            tmp_thetas[th][i] = thetas[th][i] - somme_for_grad(X, Y, thetas, th, i)
    for th in range(int(nb_theta)):
        col = tmp_thetas.shape[1]
        for i in range(col):
            thetas[th][i] = tmp_thetas[th][i]

def make_predi(X, Y, thetas, tmp_thetas, nb_theta):
    for i in range(20):
        cost_res = cost_function(X, Y, thetas, nb_theta)
        gradient_descent(X, Y, thetas, tmp_thetas, nb_theta)
        print(i)
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
