# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    describe.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/07/10 11:14:18 by msukhare          #+#    #+#              #
#    Updated: 2018/10/29 19:07:51 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
import sys
from math import sqrt
from math import floor
from math import ceil

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
        if (check_col_str(data[key])):
            data.drop([key], axis = 1, inplace = True)
        else:
            keys.append(key)
    data_describe = pd.DataFrame(0.000,\
            index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], columns = keys)
    return (data, data_describe, keys)

def ft_count(data):
    nb_ele = 0
    for ele in data:
        if (pd.notna(ele)):
            nb_ele += 1
    return (nb_ele)

def get_min_or_max(data, min_or_max):
    i = 0
    while (pd.isna(data[i])):
        i += 1
    mn_mx = data[i]
    for ele in data:
        if (min_or_max == 0 and mn_mx > ele):
            mn_mx = ele
        elif (min_or_max == 1 and mn_mx < ele):
            mn_mx = ele
    return (mn_mx)

def ft_mean(data, total):
    somme = 0
    for ele in data:
        if (pd.notna(ele)):
            somme += ele
    return (float(somme / total))

def ft_std(data, mean, total):
    to_div = 0
    for ele in data:
        if (pd.notna(ele)):
            to_div += (ele - mean)**2
    return (float(sqrt(to_div / (total - 1))))

def ft_quantile(data, total, quant):
    tmp = np.sort(np.array(data.values, dtype=float))
    i = ceil(total * quant)
    if (total % 2 != 0):
        return (tmp[i])
    return (float(tmp[i - 1] + ((tmp[i] - tmp[i - 1]) * (1 - quant))))

def main():
    if (len(sys.argv) <= 1):
        sys.exit("No name file")
    if (len(sys.argv) >= 3):
        sys.exit("too much file")
    data, data_describe, keys = read_file()
    for key in keys:
        data_describe[key]['count'] = ft_count(data[key])
        data_describe[key]['min'] = get_min_or_max(data[key], 0)
        data_describe[key]['max'] = get_min_or_max(data[key], 1)
        data_describe[key]['mean'] = float(ft_mean(data[key], data_describe[key]['count']))
        data_describe[key]['std'] = float(ft_std(data[key], data_describe[key]['mean'],\
                data_describe[key]['count']))
        data_describe[key]['25%'] = float(ft_quantile(data[key], data_describe[key]['count'], 0.25))
        data_describe[key]['50%'] = float(ft_quantile(data[key], data_describe[key]['count'], 0.50))
        data_describe[key]['75%'] = float(ft_quantile(data[key], data_describe[key]['count'], 0.75))
    print(data_describe)

if __name__ == "__main__":
    main()
