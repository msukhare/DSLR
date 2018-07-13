import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

def read_file():
    try:
        data = pd.read_csv(sys.argv[1])
    except:
        sys.exit("File doesn't exist")
    keys = []
    for key in data:
        keys.append(key)
    return (data, keys)

def split_data_by_house(data, key, house):
    frst_h = []
    sec_h = []
    th_h = []
    four_h = []
    len = data[key].shape[0]
    i = 0
    while (i < len):
        if (pd.notna(data[key][i]) and pd.notna(data[house][i])):
            if (data[house][i] == "Gryffindor"):
                frst_h.append(data[key][i])
            elif (data[house][i] == "Slytherin"):
                sec_h.append(data[key][i])
            elif (data[house][i] == "Hufflepuff"):
                th_h.append(data[key][i])
            else:
                four_h.append(data[key][i])
        i += 1
    return (frst_h, sec_h, th_h, four_h)

def put_histogramme(first_h, sec_h, th_h, four_h, key):
    plt.title(key)
    plt.ylabel("number students who get this grade")
    plt.xlabel("grades")
    plt.hist(first_h, bins='auto', facecolor='red', alpha = 0.4, label='Gryffindor')
    plt.hist(sec_h, bins='auto', facecolor='green', alpha = 0.5, label='Slytherin')
    plt.hist(th_h, bins='auto', facecolor='yellow', alpha = 0.5, label='Hufflepuff')
    plt.hist(four_h, bins='auto', facecolor='blue', alpha = 0.3, label='Ravenclaw')
    plt.legend()
    plt.show()

def main():
    if (len(sys.argv) <= 1):
        sys.exit("No name file")
    if (len(sys.argv) >= 3):
        sys.exit("too much file")
    data, keys = read_file()
    i = 6
    first_h, sec_h, th_h, four_h = split_data_by_house(data, keys[i], keys[1])
    put_histogramme(first_h, sec_h, th_h, four_h, keys[i])

if __name__ == "__main__":
    main()
