from proj1_helpers import load_csv_data, create_csv_submission
from costs import predict_labels, predict_labels_submission
from helpers import read_json
from kfold_cv import build_folds
from training import read_best_hyperparameters

from os import path
import argparse
from timeit import default_timer as timer


def read_weights(model):
    """
    Reads the weights for a corresponding model.
    """
    WEIGHTS_DIR = "hyperparams"
    WEIGHTS_FILENAME = f"weights_{model}.json"
    w = read_json(path.join(WEIGHTS_DIR, WEIGHTS_FILENAME))['weights']
    return w


def read_test_set():
    """
    Reads training data.

    Returns
    ----------
    y : ndarray of shape (n_samples,)
        Array of labels

    x : ndarray of shape (n_samples, n_features)
        Training data

    ids_test : ndarray of shape (n_samples,)
        IDs of data
    """

    DATA_FOLDER = 'Data/'
    TEST_DATASET = "test.csv"

    start = timer()
    y, x, ids_test = load_csv_data(path.join(DATA_FOLDER, TEST_DATASET), sub_sample=False)
    print(f'Data Loaded - Time: {timer() - start:.3f}\n')

    return y, x, ids_test


def main(args):
    """Compute a gradient for Logistic Regression loss.

        Parameters
        ----------
        args.method: which model to make predictions with

        Return
        ----------
        submission.csv : the predicted labels on the test set

        """
    model = args.method
    # get weights
    weights = read_weights(model)
    # get hyperparameters (degree needed for polynomial expansion)
    hyperparameters = read_best_hyperparameters(model)
    # build folds needs hyperparameter values as individual values and not as lists
    hyperparameters = {key: hyperparameters[key][0] for key in hyperparameters.keys()}
    # get test data
    test_y, test_x, ids_test = read_test_set()
    # build polynomial expansion and impute/standardize
    test_x_processed, _, _, _ = build_folds(test_y, test_x, [], -1, hyperparameters, cross_validate=False)
    # do prediction
    if 'logistic' in model:
        y_pred = predict_labels_submission(weights, test_x_processed, log=True)
    else:
        y_pred = predict_labels_submission(weights, test_x_processed, log=False)
    # save the prediction
    create_csv_submission(ids_test, y_pred, "submission.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="CLI for the run.py which generates the predictions.")
    parser.add_argument('-m', '--method', type=str, help='Which method to use to predict.', required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
