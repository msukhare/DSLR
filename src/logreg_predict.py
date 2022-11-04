import os
import sys
import argparse

from easyML import LogisticReg,\
                    scaling_features,\
                    split_data
from utils import read_data_csv_cls

def main(args):
    try:
        classificator = LogisticReg()
        params_scaling, labels = classificator.load_weights(args.weights_file)
        data = read_data_csv_cls(args.data_path)
        data, __ = scaling_features(data[0], params_scaling)
        Y_pred = classificator.predict(data)
    except Exception as error:
        sys.exit('Error: ' + str(error))
    with open(args.path_predictions, 'w') as fd:
        fd.write('Index,Hogwarts House\n')
        for index, ele in enumerate(Y_pred):
            fd.write("%d,%s\n" %(index, list(labels.keys())[list(labels.values()).index(ele)]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',\
                        nargs='?',\
                        type=str,\
                        help="""correspond to path of csv file""")
    parser.add_argument('path_predictions',\
                        nargs='?',\
                        type=str,\
                        help="""correspond to path where store predictions on dataset""")
    parser.add_argument('weights_file',\
                        nargs='?',\
                        type=str,\
                        help="""correspond to path where stored weights after training and
                                informations about pipeline""")
    parsed_args = parser.parse_args()
    if parsed_args.data_path is None:
        sys.exit("Error: missing name of CSV data to use")
    if os.path.exists(parsed_args.data_path) is False:
        sys.exit("Error: %s doesn't exists" %parsed_args.data_path)
    if os.path.isfile(parsed_args.data_path) is False:
        sys.exit("Error: %s must be a file" %parsed_args.data_path)
    if parsed_args.path_predictions is None:
        sys.exit("Error: missing name of CSV data to use")
    if parsed_args.weights_file is None:
        sys.exit("Error: missing name of CSV data to use")
    if os.path.exists(parsed_args.weights_file) is False:
        sys.exit("Error: %s doesn't exists" %parsed_args.weights_file)
    if os.path.isfile(parsed_args.weights_file) is False:
        sys.exit("Error: %s must be a file" %parsed_args.weights_file)
    main(parsed_args)