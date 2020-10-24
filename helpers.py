# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np
import json


def split_data(x, y, ratio, seed=1):
    """
    splits the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    return: x training split, y training split, x test split, y test split
    """
    # set seed
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
    """Polynomial basis functions for multivariate inputs
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
    """Load data and convert it to the metrics system."""
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
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
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
