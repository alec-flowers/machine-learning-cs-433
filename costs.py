# -*- coding: utf-8 -*-
"""Functions used to compute the loss."""
import numpy as np

def compute_error(y, tx, w):
	"""
	Computes error e.
	"""
	return y - tx.dot(w)

def compute_mse(y, tx, w):
	"""Calculate the loss using MSE (Mean Squared Error). """
	N = len(y)
	e = compute_error(y, tx, w)
	loss = 1 / (2 * N) * np.sum(e ** 2)
	return loss

def compute_mae(y, tx, w):
	"""Calculate the loss using MAE (Mean Absolute Error)."""
	N = len(y)
	e = compute_error(y, tx, w)
	loss = 1 / (2 * N) * np.sum(np.abs(e))
	return loss


def mse(y, y_hat):
	'Calculates and returns MSE between two vectors of same size'
    return np.sum((y - y_hat)**2) / (2 * len(y_hat))
    
def mae(y, y_hat):
	'Calculates and returns MAE between two vectors of same size'
    return np.sum(np.abs(y - y_hat)) / len(y_hat)

def rmse(y, y_hat):
	'Calculates and returns RMSE between two vectors of same size'
    return np.sqrt(2*np.sum((y - y_hat)**2) / (2 * len(y_hat)))

def compute_loss_alec(y, tx, w, error_fn = 'MSE'):
    """
	Calculate the loss between dependent variable and prediction.

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

    y_hat = np.sum(tx * w,axis=1)
    if error_fn == 'MSE':
        error = mse(y, y_hat)
    elif error_fn == 'MAE':
        error = mae(y, y_hat)
	elif error_fn == 'RMSE':
		error = rmse(y, y_hat)
    else:
        print('Did not match a loss function')
    return error

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
