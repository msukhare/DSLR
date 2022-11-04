import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse

LABELS = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
COLORS = ['red', 'green', 'yellow', 'blue']
ALPHA = [0.4, 0.5, 0.5, 0.3]

def read_file(data_path):
    data = pd.read_csv(data_path)
    if data.empty is True:
        raise Exception('%s is empty' %data_path)
    data = data.fillna(value={'Arithmancy': data['Arithmancy'].mean()}, inplace=False)
    data = data.fillna(value={"Care of Magical Creatures": data["Care of Magical Creatures"].mean()}, inplace=False)
    return data

def show_homogenous_feat(data, feat_name):
    for idx, label in enumerate(LABELS):
        to_show = data[data['Hogwarts House'] == label]
        plt.hist(to_show[feat_name], bins='auto', facecolor=COLORS[idx], alpha=ALPHA[idx], label=label)
    plt.legend()
    plt.title("Homogenous feature %s" %feat_name)
    plt.show()

def main(args):
    try:
        data = read_file(args.data_path)
    except Exception as error:
        sys.exit("%s" %str(error))
    show_homogenous_feat(data, 'Arithmancy')
    show_homogenous_feat(data, 'Care of Magical Creatures')

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
