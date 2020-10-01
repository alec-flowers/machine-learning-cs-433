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
