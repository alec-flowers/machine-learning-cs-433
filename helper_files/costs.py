# -*- coding: utf-8 -*-
"""Functions used to compute the loss."""
import numpy as np

from helpers import sigmoid
from helper_files.data_io import predict_labels


def compute_error(y, tx, w):
    """
    Computes error e.
    """
    return y - tx.dot(w)


def mse(e):
    'Calculates and returns MSE between two vectors of same size'
    return np.sum(e ** 2) / (2 * len(e))


def mae(e):
    'Calculates and returns MAE between two vectors of same size'
    return np.sum(np.abs(e)) / len(e)


def rmse(e):
    'Calculates and returns RMSE between two vectors of same size'
    return np.sqrt(2 * mse(e))


def compute_loss(y, tx, w, error_fn='MSE'):
    """
    Calculates the loss between dependent variable and prediction.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Array of labels

    tx : ndarray of shape (n_samples, n_features)
        Training data

    w : ndarray of shape (n_weights,)
        Weight vector

    error_fn : string selecting ['MSE', 'MAE', 'RMSE']

    Returns
    ----------
    error : np.float64
        error between dependent variable and prediction

    """

    e = compute_error(y, tx, w)
    if error_fn == 'MSE':
        error = mse(e)
    elif error_fn == 'MAE':
        error = mae(e)
    elif error_fn == 'RMSE':
        error = rmse(e)
    else:
        raise NotImplementedError('Did not match a loss function')
    return error


def calc_accuracy(y_actual, tx, w, model):
    """
    Calculates accuracy of the predicted labels of a test set for a given model

    Parameters
    ----------
    y_actual : ndarray of shape (n_samples,)
        Array of actual (real) labels

    tx : ndarray of shape (n_samples, n_features)
        Test data

    w : ndarray of shape (n_weights,)
        Weight vector

    model : string selecting ['gd', 'sgd', 'ridge', 'least_squares', 'logistic', 'regularized_logistic']
        Machine learning methods

    Returns
    ----------
    accuracy : np.float64
        Accuracy of the predicted labels
    """
    if 'logistic' in model:
        y_pred = predict_labels(w, tx, True)
    else:
        y_pred = predict_labels(w, tx)

    correct = np.sum(y_pred == y_actual)
    accuracy = correct / len(y_actual)
    return accuracy


def calculate_logistic_loss(y, tx, w):
    """Computes the loss: negative log likelihood."""
    pred = sigmoid(tx @ w)
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    N = len(y)
    return np.squeeze(-loss) / N
