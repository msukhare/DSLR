import pandas as pd
import os
import sys
import numpy as np
import argparse

from math import sqrt

def ft_count(data):
    nb_ele = 0
    for ele in data:
        if pd.notna(ele) is True:
            nb_ele += 1
    return nb_ele

def ft_quantile(data, quant):
    total = ft_count(data)
    if total == 0:
        return np.nan
    sorted_data = np.sort(np.array(data.values, dtype=float))
    h = (total - 1) * quant
    h_down = np.floor(h)
    return sorted_data[int(h_down)] + (h - h_down) * (sorted_data[int(np.ceil(h))] - sorted_data[int(h_down)])

def ft_mean(data):
    somme = 0.0
    total = float(ft_count(data))
    if total == 0:
        return np.nan
    for ele in data:
        if pd.notna(ele) is True:
            somme += ele
    return float(somme / total)

def ft_std(data):
    to_div = 0
    mean = ft_mean(data)
    total = ft_count(data)
    if total == 0:
        return np.nan
    for ele in data:
        if pd.notna(ele):
            to_div += (ele - mean)**2
    return float(sqrt(to_div / (total - 1)))

def ft_min(data):
    to_ret = None
    for ele in data:
        if pd.notna(ele) is True and\
            (to_ret is None or ele < to_ret):
            to_ret = ele
    return to_ret

def ft_max(data):
    to_ret = None
    for ele in data:
        if pd.notna(ele) is True and\
            (to_ret is None or ele > to_ret):
            to_ret = ele
    return to_ret

def col_contains_str(data):
    for ele in data:
        if pd.notna(ele) is True and\
            type(ele) is str:
            return True
    return False

def read_file(data_path):
    data = pd.read_csv(data_path)
    if data.empty is True:
        raise Exception('%s is empty' %data_path)
    for key in data.keys():
        if col_contains_str(data[key]) is True:
            data.drop([key], axis=1, inplace=True)
    return data

def main(args):
    try:
        data = read_file(args.data_path)
    except Exception as error:
        sys.exit(str(error))
    describer = pd.DataFrame(0.000,\
                            index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
                            columns=list(data.keys()))
    for key in data.keys():
        describer[key]['count'] = ft_count(data[key])
        describer[key]['min'] = ft_min(data[key])
        describer[key]['max'] = ft_max(data[key])
        describer[key]['mean'] = float(ft_mean(data[key]))
        describer[key]['std'] = float(ft_std(data[key]))
        describer[key]['25%'] = float(ft_quantile(data[key], 0.25))
        describer[key]['50%'] = float(ft_quantile(data[key], 0.50))
        describer[key]['75%'] = float(ft_quantile(data[key], 0.75))
    print(describer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',\
                        nargs='?',\
                        type=str,\
                        help="""correspond to path of csv file""")
    parsed_args = parser.parse_args()
    if parsed_args.data_path is None:
        sys.exit("Error: missing name of CSV data to use")
    if os.path.exists(parsed_args.data_path) is False:
        sys.exit("Error: %s doesn't exists" %parsed_args.data_path)
    if os.path.isfile(parsed_args.data_path) is False:
        sys.exit("Error: %s must be a file" %parsed_args.data_path)
    main(parsed_args)