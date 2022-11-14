import os
import sys
import argparse

from easyML import LogisticReg,\
                    scaling_features,\
                    split_data
from utils import read_data_csv_cls

def main(args):
    try:
        X, Y, labels, columns_name = read_data_csv_cls(args.data_path, train=True)
    except Exception as error:
        sys.exit('Error: ' + str(error))
    X, params_to_save = scaling_features(X, None, args.type_of_features_scaling)
    classificator = LogisticReg(args.kernel,\
                                args.l2,\
                                args.learning_rate,\
                                args.lambda_value,\
                                args.epochs,\
                                args.batch_size,\
                                args.early_stopping,\
                                args.validation_fraction,\
                                args.n_epochs_no_change,\
                                args.tol,\
                                args.validate,\
                                args.accuracy,\
                                args.precision,\
                                args.recall,\
                                args.f1_score,\
                                args.average)
    try:
        classificator.fit(X, Y)
    except Exception as error:
        sys.exit('Error:' + str(error))
    if args.feat_importance is True:
        if args.type_of_features_scaling != 'standardization':
            print("Warning: data must be standardized before using features importance")
        classificator.features_importance(labels, columns_name, args.nb_feat_importance_show)     
    classificator.save_weights(args.file_where_store_weights, params_to_save, labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',\
                        nargs='?',\
                        type=str,\
                        help="""correspond to path of csv file""")
    parser.add_argument('--file_where_store_weights',\
                        nargs='?',\
                        type=str,\
                        help="""correspond to path where store weights after training and
                                informations about pipeline""")
    parser.add_argument('--kernel',\
                        nargs='?',\
                        type=str,\
                        default="OVR",\
                        const="OVR",\
                        choices=['OVR', 'MULTINOMIAL'],\
                        help="""correspond to kernel to use during training.
                                By default OVR""")
    parser.add_argument('--l2',\
                        dest='l2',\
                        action='store_true',
                        help="""if pass as params will do a logistic Ridge regression
                                by default logitic regression""")
    parser.add_argument("--lambda_value",\
                        nargs='?',\
                        type=float,\
                        default=0.01,\
                        const=0.01,\
                        help="""correspond to value to use if l2 regularization is use.
                                By default 0.01""")
    parser.add_argument('--type_of_features_scaling',\
                        nargs='?',\
                        type=str,\
                        default="standardization",\
                        const="standardization",\
                        choices=['standardization', 'rescaling', 'normalization'],\
                        help="""correspond to technic use for features scaling.
                                By default standardization""")
    parser.add_argument('--learning_rate',\
                        nargs='?',\
                        type=float,\
                        default=0.1,\
                        const=0.1,\
                        help="""correspond to learning rate used during training.
                                By default 0.1""")
    parser.add_argument('--epochs',\
                        nargs='?',\
                        type=int,\
                        default=100,\
                        const=100,\
                        help="""correspond to numbers of epochs to do during training.
                                By default 100""")
    parser.add_argument('--batch_size',\
                        nargs='?',\
                        type=int,\
                        default=None,\
                        const=None,\
                        help="""correspond to numbers of sample to use for one iteration.
                                By default None all samples are used during one iteration""")
    parser.add_argument('--early_stopping',\
                        dest='early_stopping',\
                        action='store_true',
                        help="""if pass as params will do early stopping on val set, base on tol and
                                n_epochs_no_change in gradient descent""")
    parser.add_argument('--validation_fraction',\
                        nargs='?',\
                        type=float,\
                        default=0.10,\
                        const=0.10,\
                        help="""correspond to percentage data use during training as val set in gradient descent.
                                Used if early_stopping is True or validate is set True.
                                By default 0.10 percentage of data""")
    parser.add_argument('--n_epochs_no_change',\
                        nargs='?',\
                        type=int,\
                        default=5,\
                        const=5,\
                        help="""correspond to numbers of epochs wait until cost function don't change.
                                Only used in gradient descent and if --early_stoping is set at True.
                                By default 5 epochs""")
    parser.add_argument('--tol',\
                        nargs='?',\
                        type=float,\
                        default=1e-3,\
                        const=1e-3,\
                        help="""correspond to stopping criteron in early stopping.
                            Only used in gradient descent and if --early_stopping is set at True.
                            By default 1e-3""")
    parser.add_argument('--feat_importance',\
                        dest='feat_importance',\
                        action='store_true',
                        help="""if pass as params will show features importance at the end of training""")
    parser.add_argument('--nb_feat_importance_show',\
                        nargs='?',\
                        type=int,\
                        default=3,\
                        const=3,\
                        help="""correspond to numbers features importance to plot.
                                By default 1""")
    parser.add_argument('--validate',\
                        dest='validate',\
                        action='store_true',
                        help="""if pass as params will do evaluation on validation set during training,
                                By default show only loss function, you can add other metrics""")
    parser.add_argument('--accuracy',\
                        dest='accuracy',\
                        action='store_true',
                        help="""if pass as params will compute accuracy on validation set
                                validate must be pass as params to show accuracy""")
    parser.add_argument('--precision',\
                        dest='precision',\
                        action='store_true',
                        help="""if pass as params will compute precision on validation set
                                validate must be pass as params to show precision""")
    parser.add_argument('--recall',\
                        dest='recall',\
                        action='store_true',
                        help="""if pass as params will compute recall on validation set
                                validate must be pass as params to show recall""")
    parser.add_argument('--f1_score',\
                        dest='f1_score',\
                        action='store_true',
                        help="""if pass as params will compute f1_score on vaslidation set
                                validate must be pass as params to show f1_score""")
    parser.add_argument('--average',\
                        nargs='?',\
                        type=str,\
                        default="macro",\
                        const="macro",\
                        choices=['micro', 'macro'],\
                        help="""correspond to type of average to use during compute of metrics.
                                By default macro""")
    parsed_args = parser.parse_args()
    if parsed_args.data_path is None:
        sys.exit("Error: missing name of CSV data to use")
    if os.path.exists(parsed_args.data_path) is False:
        sys.exit("Error: %s doesn't exists" %parsed_args.data_path)
    if os.path.isfile(parsed_args.data_path) is False:
        sys.exit("Error: %s must be a file" %parsed_args.data_path)
    main(parsed_args)