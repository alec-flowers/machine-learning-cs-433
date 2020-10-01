# -*- coding: utf-8 -*-
"""Gradient Descent"""

import numpy as np
from costs import compute_mse, compute_error


def compute_gradient(y, tx, w):
	"""Compute the gradient."""
	e = compute_error(y, tx, w)
	N = len(y)
	gradient = -1 / N * (np.transpose(tx)).dot(e)
	return e, gradient


def gradient_descent(y, tx, initial_w, max_iters, gamma):
	"""Gradient descent algorithm."""
	# Define parameters to store w and loss
	ws = [initial_w]
	losses = []
	w = initial_w
	for n_iter in range(max_iters):
		"""Compute gradient and loss"""
		e, gradient = compute_gradient(y, tx, w)
		w = w - gamma * gradient
		loss = compute_mse(y, tx, w)
		# store w and loss
		ws.append(w)
		losses.append(loss)
		print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
			bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
	return losses, ws
