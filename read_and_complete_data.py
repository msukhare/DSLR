import pandas as pd
import numpy as np
import sys

def scale_feature(data):
    desc = data.describe()
    X_scale = np.zeros((data.shape[1], data.shape[0]), dtype=float)
    i = 0
    for key in data:
        for j in range(int(desc[key]['count'])):
            X_scale[i][j] = (data[key][j] - desc[key]['mean']) / desc[key]['std'] #(desc[key]['max'] - desc[key]['min'])
        i += 1
    return (X_scale)

def replace_nan_by_similare(replace, by):
    new_feat = np.zeros((replace.shape[0],1), dtype=float)
    for i in range(int(replace.shape[0])):
        if (pd.isna(replace[i])):
            new_feat[i][0] = by[i]
        else:
            new_feat[i][0] = replace[i]
    return (new_feat)

def get_content_file(file):
    try:
        content = pd.read_csv(file)
    except:
        print("fail to open", file)
        sys.exit()
    return (content)

def read_file(to_ret):
    data_train = get_content_file("dataset_train.csv")
    data_test = get_content_file("dataset_test.csv")
    data = [data_train, data_test]
    data = pd.concat(data).reset_index(drop=True)
    data.drop(['First Name', 'Last Name', 'Birthday', 'Index', 'Best Hand', 'Arithmancy',\
            'Care of Magical Creatures'], axis=1, inplace=True)
    data['Hogwarts House'] = data['Hogwarts House'].map({'Ravenclaw' : 3, 'Slytherin': 2,\
            'Gryffindor' : 1, 'Hufflepuff' : 4})
    for key in data:
        if (key != "Hogwarts House" and key != "Astronomy"):
            data.fillna(value={key: data[key].mean()}, inplace=True)
    data['Astronomy'] = replace_nan_by_similare(data['Astronomy'], \
            data['Defense Against the Dark Arts'])
    if (to_ret == "predict"):
        data.drop(['Hogwarts House', 'Defense Against the Dark Arts'], axis=1, inplace=True)
        return (scale_feature(data.iloc[1600: ].reset_index(drop=True)).transpose())
    data = data.iloc[0: 1600]
    data = data.sample(frac=1, random_state=52).reset_index(drop=True)
    Y = data.iloc[:, 0:1]
    Y = np.array(Y.values, dtype=float)
    data.drop(['Hogwarts House', 'Defense Against the Dark Arts'], axis=1, inplace=True)
    return (scale_feature(data).transpose(), Y)

