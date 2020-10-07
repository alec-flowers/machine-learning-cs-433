# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

from costs import *
from helpers import batch_iter


def compute_stoch_gradient(y, tx, w):
	"""Compute a gradient for MSE. 
	
	Parameters
	----------
	y : ndarray of shape (n_samples,)
		Array of labels

	tx : ndarray of shape (n_samples, n_features)
		Training data

	w : ndarray of shape (n_weights,)
		Weight vector

	Returns
	----------
	gradient : ndarray of shape (n_weights, )
		gradient of MSE
	
	"""
	N = len(tx)
	e = compute_error(y, tx, w)
	gradient = (-1/N) * tx.T.dot(e)

	return gradient


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, num_batches):
	"""Stochastic Gradient Descent algorithm.  

	batch_size selected at 1 this is classic SGD. batch_size > 1 this is now Minibatch
	SGD. Using MSE as the loss function.
	
	Parameters
	----------
	y : ndarray of shape (n_samples,)
		Array of labels

	tx : ndarray of shape (n_samples, n_features)
		Training data

	initial_w : ndarray of shape (n_weights,)
		Weight vector

	batch_size : int
		size of 

	max_iters : int
		number of training epochs

	gamma : float
		learning rate

	num_batches: int
		number of minibatches to use during each iteration

	Returns
	----------
	losses : list of shape (max_iters+1, )
		MSE loss for corresponding weight values
		Index relates to the epoch

	ws : list of shape (max_iters+1, )
		Weight values updated by gradient
		Index relates to the epoch
	
	"""

	W0 = 0
	ws = [initial_w]
	losses = [compute_mse(y, tx, ws[W0])]

	for n_iter in range(max_iters):

		for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=num_batches):
			'''ws[-1] selects the last element in the list.'''
			gradient = compute_stoch_gradient(batch_y, batch_tx, ws[-1])
			w = ws[-1] - gamma * gradient
			loss = compute_mse(y, tx, w)

			ws.append(w)
			losses.append(loss)
			
		print("Gradient Descent({bi}/{ti}): loss={l:.6f}, w0={w0:.3f}, w1={w1:.3f}".format(
				bi=n_iter, ti=max_iters - 1, l=losses[-1], w0=w[0], w1=w[1]))

	return losses, ws
