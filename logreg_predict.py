import pandas as pd
import numpy as np
from read_and_complete_data import read_file
from logreg_train import predict_Y
import csv
import sys

def read_thetas():
    try:
        thetas = pd.read_csv("thetas.csv")
    except:
        sys.exit("thetas.csv doesn't exist, use logreg_trasin.py create it")
    thetas = thetas.iloc[:]
    thetas = np.array(thetas.values, dtype=float)
    return (thetas)

def write_pred_in_file(pred_Y):
    try:
        file = csv.writer(open("house.csv", "w"), delimiter=',',\
                quoting=csv.QUOTE_MINIMAL)
    except:
        sys.exit("fail to create house.csv")
    for i in range(pred_Y.shape[0]):
        if (pred_Y[i] == 1):
            name_house = "Gryffindor"
        elif (pred_Y[i] == 2):
            name_house = "Slytherin"
        elif (pred_Y[i] == 3):
            name_house = "Ravenclaw"
        elif (pred_Y[i] == 4):
            name_house = "Hufflepuff"
        file.writerow([i, name_house])

def main():
    thetas = read_thetas()
    X = read_file("predict")
    X = np.c_[np.ones((X.shape[0], 1), dtype=float), X]
    pred_Y = predict_Y(X, thetas, thetas.shape[0])
    write_pred_in_file(pred_Y)

if (__name__ == "__main__"):
    main()
