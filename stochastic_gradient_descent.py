# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

from costs import *
from helpers import batch_iter


def compute_stoch_gradient(y, tx, w):
	"""Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
	y_hat = np.sum(tx * w, axis=1)
	N = len(tx)
	e = y - y_hat
	
	gradient = (-1/N) * tx.T.dot(e)
	
	return gradient


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
	"""Stochastic gradient descent algorithm."""

	# Define parameters to store w and loss
	ws = [initial_w]
	losses = []

	for n_iter in range(max_iters):
		"""Compute gradient and loss"""
		loss = compute_mse(y, tx, ws[n_iter])

		for batch_y, batch_tx in batch_iter(y, tx, batch_size):
			gradient = compute_stoch_gradient(y, tx, ws[n_iter])
			w = ws[n_iter] - gamma * gradient

		# store w and loss
		ws.append(w)
		losses.append(loss)
		print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
			bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

	return losses, ws
