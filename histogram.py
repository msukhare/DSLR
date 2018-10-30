# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    histogram.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/29 19:09:25 by msukhare          #+#    #+#              #
#    Updated: 2018/10/30 04:22:09 by msukhare         ###   ########.fr        #
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
    for i in range(data[key].shape[0]):
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

def show_histogramme(data):
    f, axs = plt.subplots(2, 7, figsize=(18, 14))
    i = 0
    j = 0
    for key in data:
        if (j == 7):
            j = 0
            i += 1
        if (key != "Hogwarts House"):
            first_h, sec_h, th_h, four_h = split_data_by_house(data, key)
            axs[i, j].set_title(key)
            axs[i, j].hist(first_h, bins='auto', facecolor='red', alpha = 0.4, label='Gryffindor')
            axs[i, j].hist(sec_h, bins='auto', facecolor='green', alpha = 0.5, label='Slytherin')
            axs[i, j].hist(th_h, bins='auto', facecolor='yellow', alpha = 0.5, label='Hufflepuff')
            axs[i, j].hist(four_h, bins='auto', facecolor='blue', alpha = 0.3, label='Ravenclaw')
            axs[i, j].legend()
            j += 1
    plt.show()

def show_most_homogenous_feat(data):
    f, axs = plt.subplots(2, 1, figsize=(18, 14))
    first_h, sec_h, th_h, four_h = split_data_by_house(data, "Arithmancy")
    axs[0].set_title("Most homogenous feature Arithmancy")
    axs[0].hist(first_h, bins='auto', facecolor='red', alpha = 0.4, label='Gryffindor')
    axs[0].hist(sec_h, bins='auto', facecolor='green', alpha = 0.5, label='Slytherin')
    axs[0].hist(th_h, bins='auto', facecolor='yellow', alpha = 0.5, label='Hufflepuff')
    axs[0].hist(four_h, bins='auto', facecolor='blue', alpha = 0.3, label='Ravenclaw')
    axs[0].legend()
    first_h, sec_h, th_h, four_h = split_data_by_house(data, "Care of Magical Creatures")
    axs[1].set_title("Most homogenous feature Care of Magical Creatures")
    axs[1].hist(first_h, bins='auto', facecolor='red', alpha = 0.4, label='Gryffindor')
    axs[1].hist(sec_h, bins='auto', facecolor='green', alpha = 0.5, label='Slytherin')
    axs[1].hist(th_h, bins='auto', facecolor='yellow', alpha = 0.5, label='Hufflepuff')
    axs[1].hist(four_h, bins='auto', facecolor='blue', alpha = 0.3, label='Ravenclaw')
    axs[1].legend()
    plt.show()

def main():
    if (len(sys.argv) <= 1):
        sys.exit("No name file")
    if (len(sys.argv) >= 3):
        sys.exit("too much file")
    data = read_file()
    show_histogramme(data)
    show_most_homogenous_feat(data)

if __name__ == "__main__":
    main()
