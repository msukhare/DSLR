import pandas as pd
import numpy as np

def replace_nan_by_mean(column):
    index_nan = column.index[column.apply(np.isnan)]
    column[index_nan] = np.mean(column)
    return column

def replace_nan_by_correlated_value(X, correlated):
    index_nan = X.index[X.apply(np.isnan)]
    X[index_nan] = correlated[index_nan]
    return X

def read_data_csv_cls(path_to_data, train=False, seed=423):
    Y = None
    labels = None
    data = pd.read_csv(path_to_data)
    if data.empty is True:
        raise Exception('%s is empty' %path_to_data)
    if train is True:
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        labels = {lab: i for i, lab in enumerate(sorted(list(set(data['Hogwarts House']))))}
        Y = np.asarray(data['Hogwarts House'].map(labels))
    data['Defense Against the Dark Arts'] = replace_nan_by_correlated_value(data['Defense Against the Dark Arts'].copy(deep=True), data['Astronomy'])
    data = data.drop(['Best Hand', 'Arithmancy', 'Care of Magical Creatures', 'Astronomy', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Index'], inplace=False, axis=1).reset_index(drop=True)
    for column in data:
        data[column] = replace_nan_by_mean(data[column].copy(deep=True))
    return np.asarray(data), Y, labels, list(data.keys())