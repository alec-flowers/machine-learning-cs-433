# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np
import json
import csv

models = ["gd", "sgd", "ridge", "least_squares", "logistic", "regularized_logistic"]
model_to_string = {
    "gd": "GD", "sgd": "SGD", "ridge": "RIDGE", "least_squares": "LS", "logistic": "LR", "regularized_logistic": "LRL"
}


def split_data(x, y, ratio, seed=1):
    """
    Splits the dataset based on the split ratio.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_features)
        Dataset

    y : ndarray of shape (n_samples,)
        Array of labels

    ratio : np.float64
        Data will be splitted according to this ratio, where this value corresponds to training

    Returns
    ----------
    x2_train : ndarray
        Data subset dedicated to training

    y2_train : ndarray
        Labels corresponding to the data subset dedicated to training

    x2_test : ndarray
        Data subset dedicated to test

    y2_test : ndarray
        Labels correspondig to the data subset dedicated to test
    """

    np.random.seed(seed)
    # concatinate x and y into one array
    x_y_concat = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)]
    # get x and y so that when x_y_concat is shuffled, these are shuffled as well
    x2 = x_y_concat[:, :x.size // len(x)].reshape(x.shape)
    y2 = x_y_concat[:, x.size // len(x):].reshape(y.shape)
    # shuffle to get random split
    np.random.shuffle(x_y_concat)
    index_to_split_x = int(len(x) * ratio)
    index_to_split_y = int(len(y) * ratio)
    x2_train = x2[:index_to_split_x]
    x2_test = x2[index_to_split_x:]
    y2_train = y2[:index_to_split_y]
    y2_test = y2[index_to_split_y:]
    return x2_train, y2_train, x2_test, y2_test


def build_poly(x, degree):
    """
    Polynomial basis functions for multivariate inputs

    Parameters
    ----------
    x : ndarray of shape (n_rows, n_col)
        Array of training data

    degree : int > 0
        polynomial degree

    Returns
    ----------
    poly : np.array (n_rows, expansion)
        x data expanded to the polynomial degree

    ind : list (expansion)
        Expanded terms using index of original X array. (Note am using 1 based indexing)

        ex '111' column 1 of x - c1^3
        ex '122' means column 1 and column 2 of x - c1*c2^2
    """
    assert degree > 0, 'Degree must be a natural number'
    row = x.shape[0]
    col = x.shape[1]
    if degree == 1:
        return np.c_[np.ones(row), x], [[str(i + 1)] for i in range(col)]
    if degree >= 2:
        poly, ind = build_poly(x, degree - 1)
        set_ind = set(tuple(i) for i in ind)

        p_col = poly.shape[1]
        for i in range(col):
            for j in range(1, p_col):
                temp = sorted(ind[i] + ind[j - 1])
                set_temp = tuple(temp)

                # To not duplicate data:
                if (set_temp not in set_ind):
                    mult = x[:, i] * poly[:, j]
                    poly = np.c_[poly, mult]  # !ERROR
                    ind.append(temp)
                    set_ind.add(set_temp)
        return poly, ind


def load_data(sub_sample=True, add_outlier=False):
    """
    Loads data and converts it to the metrics system.
    """
    path_dataset = "./Data/height_weight_genders.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454

    # sub-sample
    if sub_sample:
        height = height[::50]
        weight = weight[::50]

    if add_outlier:
        # outlier experiment
        height = np.concatenate([height, [1.1, 1.2]])
        weight = np.concatenate([weight, [51.5 / 0.454, 55.3 / 0.454]])

    return height, weight, gender


def standardize(x):
    """Standardizes the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generates a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


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


def load_csv_data(data_path, skip_header=1, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=skip_header, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=skip_header)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # column 4 stays delete (5, 6, 12, 26, 27, 28)
    # column 9 stays delete (21, 29)
    # x = np.delete(x,[5,6,12,21,26,27,28,29],axis = 1)

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
