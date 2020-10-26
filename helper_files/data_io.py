import csv
import json
from os import path
from timeit import default_timer as timer
import numpy as np

from helper_files.helpers import sigmoid


def load_csv_data(data_path, skip_header=1, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=skip_header, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=skip_header)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # delete raw values in data set
    input_data = np.delete(input_data,[_ for _ in range(12,30)] ,axis = 1)

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = 0
    yb[np.where(np.char.startswith(y, '-'))] = 0

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def save_csv_data(data_path, data):
    print('---Saving Data---')
    np.savetxt(data_path, data, delimiter=",", header='Take some space')


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def read_json(filename):
    """
    Reads the given file (.json) and returns the object.
    filename: name of the file (with .json extension)
    """
    if ".json" not in filename:
        raise NameError("Filename needs to include .json extension!")
    with open(filename, "r") as f:
        return json.load(f)


def write_json(filename, object):
    """
    Writes a given object to a json file.
    filename: name of the file (with .json extension)
    object: object to write
    """
    if ".json" not in filename:
        raise NameError("Filename needs to include .json extension!")
    with open(filename, "w") as f:
        json.dump(object, f)


def read_training_set():
    """
    Reads training data.

    Returns
    ----------
    y : ndarray of shape (n_samples,)
        Array of labels

    x : ndarray of shape (n_samples, n_features)
        Training data

    ids_train : ndarray of shape (n_samples,)
        IDs of data
    """

    DATA_FOLDER = 'Data/'
    TRAIN_DATASET = path.join(DATA_FOLDER, "train.csv")

    start = timer()
    y, x, ids_train = load_csv_data(TRAIN_DATASET, sub_sample=False)
    print(f'Data Loaded - Time: {timer() - start:.3f}\n')

    return y, x, ids_train


def read_best_hyperparameters(model):
    """
    Reads the input best performing set of hyperparameters for a given model.

    Parameters
    ----------
    model : string selecting ['gd', 'sgd', 'ridge', 'least_squares', 'logistic', 'regularized_logistic']
        Machine learning methods

    Returns
    ----------
    hyperparameters : dict containing the best performing set of hyperparameters
    """

    HYPERPARAMS_FOLDER = 'hyperparams_weights/'
    HYPERPARAMS_BEST_VALUES = f'best_hyperparams_{model}.json'

    filename = path.join(HYPERPARAMS_FOLDER, HYPERPARAMS_BEST_VALUES)
    hyperparameters = read_json(filename)

    return hyperparameters


def predict_labels_submission(weights, data, log=False):
    y_pred = predict_labels(weights, data, log=log)
    y_pred[np.where(y_pred == 0)] = -1
    return y_pred


def predict_labels(weights, data, log=False):
    """
    Generates class predictions given weights, and a test data matrix

    Parameters
    ----------
    weights : ndarray of shape (n_weights,)
        Weight vector

    data : ndarray of shape (n_samples, n_features)
        Test data

    log : bool
        True if using (regularized) logistic regression


    Returns
    ----------
    y_pred : ndarray of shape (n_samples,)
        Array of predicted labels
    """

    if log:
        y_pred = sigmoid(np.dot(data, weights))
    else:
        y_pred = np.dot(data, weights)

    y_pred[np.where(y_pred <= .5)] = 0
    y_pred[np.where(y_pred > .5)] = 1

    return y_pred