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

def put_scatter_plot(first_h, sec_h, th_h, four_h, sc_first_h, sc_sec_h, sc_th_h, sc_four_h, keys, i, j):
    plt.scatter(first_h, sc_first_h, s = 10, c='red', marker='x', label='Gryffindor')
    plt.scatter(sec_h, sc_sec_h, s = 10, c='green', marker='x', label='Slytherin')
    plt.scatter(th_h, sc_th_h, s = 10, c='yellow', marker='x', label='Hufflepuff')
    plt.scatter(four_h, sc_four_h, s = 10, c='blue', marker='x', label='Ravenclaw')
    plt.xlabel(keys[j])
    plt.ylabel(keys[i])
    plt.legend()
    plt.title("two similar features")
    plt.show()

def main():
    if (len(sys.argv) <= 1):
        sys.exit("No name file")
    if (len(sys.argv) >= 3):
        sys.exit("too much file")
    data, keys = read_file()
    i = 17
    j = 18
    first_h, sec_h, th_h, four_h = split_data_by_house(data, keys[i], keys[1])
    sc_first_h, sc_sec_h, sc_th_h, sc_four_h = split_data_by_house(data, keys[j], keys[1])
    put_scatter_plot(first_h, sec_h, th_h, four_h, sc_first_h, sc_sec_h, sc_th_h, sc_four_h, keys, i, j)

if __name__ == "__main__":
    main()
