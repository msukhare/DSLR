# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    pair_plot.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/07/09 16:34:31 by msukhare          #+#    #+#              #
#    Updated: 2018/10/30 04:55:34 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def check_col_str(data):
    for ele in data:
        if (pd.notna(ele) and type(ele) is str):
                return (1);
    return (0)

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

def main():
    if (len(sys.argv) <= 1):
        sys.exit("No name file")
    if (len(sys.argv) >= 3):
        sys.exit("too much file")
    data = read_file()
    ##after see the pair plot, histogram, scatterplot i decied to move up best hand
    ##, arytmencie, care of magic creature and astronomy
    data = data.drop(['Best Hand', 'Astronomy', 'Arithmancy', 'Care of Magical Creatures'],\
            axis=1)
    sns.pairplot(data, hue="Hogwarts House")
    plt.show()

if __name__ == "__main__":
    main()
