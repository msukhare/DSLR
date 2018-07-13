# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    pair_plot.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/07/09 16:34:31 by msukhare          #+#    #+#              #
#    Updated: 2018/07/10 09:46:58 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import seaborn as sns
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
    keys = []
    for key in data:
        if (check_col_str(data[key]) or key == "index"):
            data.drop([key], axis = 1, inplace = True)
        else:
            keys.append(key)
    return (data, keys)

def main():
    if (len(sys.argv) <= 1):
        sys.exit("No name file")
    if (len(sys.argv) >= 3):
        sys.exit("too much file")
    data, keys = read_file()
    iris = sns.load_dataset("iris")
    iris.head()
    sns.pairplot(iris, hue='species', size=2.5)
    sns.show()

if __name__ == "__main__":
    main()
