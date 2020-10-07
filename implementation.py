import numpy as np

from costs import compute_loss_alec


def least_squares(y, tx):
	"""Least Squares
	
	Parameters
	----------
	y : ndarray of shape (n_samples,)
		Array of labels

	tx : ndarray of shape (n_samples, n_features)
		Training data

	Returns
	----------
	w : np.array of shape(1, D)
		Optimal weights calculated using normal equations.

	loss : np.float64
		RMSE loss for corresponding weight value
		
	"""
	w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
	loss = compute_loss_alec(y, tx, w, 'RMSE')
	return w, loss


def ridge_regression(y, tx, lambda_):
	"""Ridge Regression
	
	Parameters
	----------
	y : ndarray of shape (n_samples,)
		Array of labels

	tx : ndarray of shape (n_samples, n_features)
		Training data

	lambda : float [0, 1]
		parsimony penalty

	Returns
	----------
	w : np.array of shape(1, D)
		Optimal weights calculated using normula equations.

	loss : np.float64
		RMSE loss for corresponding weight value
		
	"""

	N = len(tx)
	D = tx.shape[1]

	w = np.linalg.solve(tx.T.dot(tx) + 2 * N * lambda_ * np.identity(D), tx.T.dot(y))
	loss = compute_loss_alec(y, tx, w, 'RMSE')

	return w, loss
