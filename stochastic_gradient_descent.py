# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

from costs import *
from helpers import batch_iter


def compute_stoch_gradient(y, tx, w, batch_size):
	"""Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
	N = len(y)
	y_batch, tx_batch = batch_iter(y, tx, batch_size=batch_size, num_batches=1)
	e = compute_error(y_batch, tx_batch, w)
	g = -1 / N * (np.transpose(tx)).dot(e)
	return e, g


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
	"""Stochastic gradient descent algorithm."""

	# Define parameters to store w and loss
	ws = [initial_w]
	losses = []
	w = initial_w

	for n_iter in range(max_iters):
		"""Compute gradient and loss"""
		e, gradient = compute_stoch_gradient(y, tx, w, batch_size)
		w = w - gamma * gradient
		loss = compute_mse(y, tx, w)
		# store w and loss
		ws.append(w)
		losses.append(loss)
		print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
			bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

	return losses, ws
