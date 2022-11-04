import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse

LABELS = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
COLORS = ['red', 'green', 'yellow', 'blue']

def read_file(data_path):
    data = pd.read_csv(data_path)
    if data.empty is True:
        raise Exception('%s is empty' %data_path)
    data = data.fillna(value={'Astronomy': data['Astronomy'].mean()}, inplace=False)
    data = data.fillna(value={"Defense Against the Dark Arts": data["Defense Against the Dark Arts"].mean()}, inplace=False)
    return data

def show_the_same_feature(data):
    for idx, label in enumerate(LABELS):
        to_show = data[data['Hogwarts House'] == label].reset_index(drop=True)
        plt.scatter(to_show['Astronomy'], to_show['Defense Against the Dark Arts'], s=10, c=COLORS[idx], marker='x', label=label)
    plt.xlabel("Astronomy")
    plt.ylabel("Defense Against the Dark Arts")
    plt.legend()
    plt.title("two similar features")
    plt.show()

def main(args):
    data = read_file(args.data_path)
    show_the_same_feature(data)

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