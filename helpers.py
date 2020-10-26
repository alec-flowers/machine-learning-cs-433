# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np

models = ["gd", "sgd", "ridge", "least_squares", "logistic", "regularized_logistic"]
model_to_string = {
    "gd": "GD", "sgd": "SGD", "ridge": "RIDGE", "least_squares": "LS", "logistic": "LR", "regularized_logistic": "LRL"
}


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


def standardize(x):
    """Standardizes the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def sigmoid(t):
    """Applies the sigmoid function on t."""
    sigmoid = 1 / (1 + np.exp(-t))
    return sigmoid


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
