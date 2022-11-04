import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import argparse
import seaborn as sns
import matplotlib

def replace_nan_by_mean(column):
    index_nan = column.index[column.apply(np.isnan)]
    column[index_nan] = np.mean(column)
    return column

def replace_nan_by_correlated_value(X, correlated):
    max_value = np.max(X)
    X /= max_value
    correlated /= np.max(correlated)
    index_nan = X.index[X.apply(np.isnan)]
    X[index_nan] = correlated[index_nan]
    return X * max_value

def read_file(path_to_data):
    data = pd.read_csv(path_to_data)
    data['Defense Against the Dark Arts'] = replace_nan_by_correlated_value(data['Astronomy'].copy(deep=True), data['Defense Against the Dark Arts'])
    tmp = data['Hogwarts House']
    data = data.drop(['Best Hand', 'Arithmancy', 'Care of Magical Creatures', 'Defense Against the Dark Arts', 'First Name', 'Last Name', 'Birthday', 'Index', 'Hogwarts House'], inplace=False, axis=1).reset_index(drop=True)
    for column in data:
        data[column] = replace_nan_by_mean(data[column].copy(deep=True))
    data['Hogwarts House'] = tmp
    return data

def main(args):
    data = read_file(args.data_path)
    sns.pairplot(data, hue="Hogwarts House")
    plt.show()

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