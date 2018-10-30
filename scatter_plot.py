# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    scatter_plot.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/30 02:09:52 by msukhare          #+#    #+#              #
#    Updated: 2018/10/30 04:22:07 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

def read_file():
    try:
        data = pd.read_csv(sys.argv[1])
    except:
        sys.exit("File doesn't exist")
    data = data.drop(['First Name', 'Last Name', 'Birthday', 'Index'], axis=1)
    data['Best Hand'] = data['Best Hand'].map({'Right': 0, 'Left': 1})
    for key in data:
        if (key != "Hogwarts House"):
            data.fillna(value={key: data[key].mean()}, inplace=True)
    return (data)

def split_data_by_house(data, key):
    frst_h = []
    sec_h = []
    th_h = []
    four_h = []
    for i in range(data.shape[0]):
        if (pd.notna(data[key][i])):
            if (data['Hogwarts House'][i] == "Gryffindor"):
                frst_h.append(data[key][i])
            elif (data['Hogwarts House'][i] == "Slytherin"):
                sec_h.append(data[key][i])
            elif (data['Hogwarts House'][i] == "Hufflepuff"):
                th_h.append(data[key][i])
            else:
                four_h.append(data[key][i])
    return (frst_h, sec_h, th_h, four_h)

def show_the_same_feature(first_h, sec_h, th_h, four_h, sc_first_h, sc_sec_h, sc_th_h, sc_four_h):
    plt.scatter(first_h, sc_first_h, s = 10, c='red', marker='x', label='Gryffindor')
    plt.scatter(sec_h, sc_sec_h, s = 10, c='green', marker='x', label='Slytherin')
    plt.scatter(th_h, sc_th_h, s = 10, c='yellow', marker='x', label='Hufflepuff')
    plt.scatter(four_h, sc_four_h, s = 10, c='blue', marker='x', label='Ravenclaw')
    plt.xlabel("Astronomy")
    plt.ylabel("Defense Against the Dark Arts")
    plt.legend()
    plt.title("two similar features")
    plt.show()

def show_scatters_plots(data):
    for key in data:
        if (key != "Hogwarts House"):
            f, axs = plt.subplots(2, 7, figsize=(18, 14))
            f.suptitle(key)
            first_h, sec_h, th_h, four_h = split_data_by_house(data, key)
            i = 0
            j = 0
            for key1 in data:
                if (j == 7):
                    i += 1
                    j = 0
                if (key1 != "Hogwarts House" and key != key1):
                    first_h1, sec_h1, th_h1, four_h1 = split_data_by_house(data, key1)
                    axs[i, j].set_title(key1)
                    axs[i, j].scatter(first_h, first_h1, s = 10, c='red',\
                            marker='x', label='Gryffindor')
                    axs[i, j].scatter(sec_h, sec_h1, s= 10, c='green', marker='x',\
                            label='Slytherin')
                    axs[i, j].scatter(th_h, th_h1, s= 10, c='yellow', marker='x',\
                            label='Hufflepuff')
                    axs[i, j].scatter(four_h, four_h1, s= 10, facecolor='blue', marker='x',\
                            label='Ravenclaw')
                    axs[i, j].legend()
                    j += 1
        plt.show()

def main():
    if (len(sys.argv) <= 1):
        sys.exit("No name file")
    if (len(sys.argv) >= 3):
        sys.exit("too much file")
    data = read_file()
    #this function shows all scatter plot, used to find the similare features
    #show_scatters_plots(data)
    first_h, sec_h, th_h, four_h = split_data_by_house(data, "Astronomy")
    sc_first_h, sc_sec_h, sc_th_h, sc_four_h = split_data_by_house(data, "Defense Against the Dark Arts")
    show_the_same_feature(first_h, sec_h, th_h, four_h, sc_first_h, sc_sec_h, sc_th_h, sc_four_h)

if __name__ == "__main__":
    main()
