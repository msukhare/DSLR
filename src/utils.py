import pandas as pd
import numpy as np

def replace_nan_by_mean(column):
    pass

def replace_nan_by_correlated(X, correlated):
    pass

def read_data_csv_cls(path_to_data, train=False, seed=423):
    data = pd.read_csv(path_to_data)
    Y = None
    labels = None
    if data.empty is True:
        raise Exception('%s is empty' %path_to_data)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    if train is True:
        labels = {lab: i for i, lab in enumerate(sorted(list(set(data['Hogwarts House']))))}
        Y = np.asarray(data['Hogwarts House'].map(labels))
    data = data.drop(['Hogwarts House'], inplace=False).reset_index(drop=True)
    data[] = replace_nan_by_correlated_value(data[], data[])
    data = data.drop(['', '', ''], inplace=False).reset_index(drop=True)
    for column in data:
        data[column] = replace_nan_by_mean(data[column])
    return np.asarray(data), Y