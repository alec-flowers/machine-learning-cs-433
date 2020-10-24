# -*- coding: utf-8 -*-
"""Functions used to compute the loss."""
import numpy as np
from proj1_helpers import predict_labels


def compute_error(y, tx, w):
    """
    Computes error e.
    """
    return y - tx.dot(w)


def compute_mse(y, tx, w):
    """Calculates the loss using MSE (Mean Squared Error). """
    N = len(y)
    e = compute_error(y, tx, w)
    loss = 1 / (2 * N) * np.sum(e ** 2)
    return loss


def compute_mae(y, tx, w):
    """Calculates the loss using MAE (Mean Absolute Error)."""
    N = len(y)
    e = compute_error(y, tx, w)
    loss = 1 / (2 * N) * np.sum(np.abs(e))
    return loss


def mse(e):
    """Calculates and returns MSE between two vectors of same size"""
    return np.sum(e ** 2) / (2 * len(e))


def mae(e):
    """Calculates and returns MAE between two vectors of same size"""
    return np.sum(np.abs(e)) / len(e)


def rmse(e):
    """Calculates and returns RMSE between two vectors of same size"""
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


def calc_accuracy(y_actual, tx, w):
    y_pred = predict_labels(w, tx)
    correct = np.sum(y_pred == y_actual)
    return correct / len(y_actual)

def sigmoid(t):
    """Applies the sigmoid function on t."""
    sigmoid = 1 / (1 + np.exp(-t))
    return sigmoid

def calculate_logistic_loss(y, tx, w):
	"""Computes the loss: negative log likelihood."""
	pred = sigmoid(tx @ w)
	loss1 = y.T.dot(np.log(pred))
	loss2 = (1-y).T.dot(np.log(1-pred))
	loss = loss1 + loss2
	return np.squeeze(-loss)
    #### JANET:
    #log = np.sum(np.log(1 + np.exp(tx @ w)))
    #minus = - np.sum(y * (tx @ w))
    #loss = minus + log
    #return loss

def test():
    y = np.array([2, 3, 4, 3])
    tx = np.array([[1, 7], [1, 3], [1, 1], [1, 2]])
    w = np.array([1, 2])
    mse = compute_mse(y, tx, w)
    mae = compute_mae(y, tx, w)
    print("MSE: " + str(mse))
    print("MAE: " + str(mae))


if __name__ == "__main__":
    test()
