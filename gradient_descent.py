# -*- coding: utf-8 -*-
"""Gradient Descent"""

import numpy as np
from helpers import batch_iter
from costs import compute_mse, compute_mae, compute_error

def compute_gradient(y, tx, w):
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
	y_hat = np.sum(tx * w, axis=1)
	N = len(tx)
	e = y - y_hat
	
	gradient = (-1/N) * tx.T.dot(e)

	return gradient

def gradient_descent(y, tx, initial_w, max_iters, gamma):
	"""Gradient Descent algorithm.  

	Every epoch takes sums errors across all y - e and is therefore computationally more expensive than SGD.
	
	Parameters
	----------
	y : ndarray of shape (n_samples,)
		Array of labels

	tx : ndarray of shape (n_samples, n_features)
		Training data

	initial_w : ndarray of shape (n_weights,)
		Weight vector

	max_iters : int
		number of training epochs

	gamma : float
		learing rate

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
		gradient = compute_gradient(y, tx, ws[n_iter])
		w = ws[n_iter] - gamma * gradient
	
		ws.append(w)
		loss = compute_mse(y, tx, ws[n_iter+1])
		losses.append(loss)
		
		print("GD({bi}/{ti}): loss={l:.6f}, w0={w0:.3f}, w1={w1:.3f}".format(
			bi=n_iter, ti=max_iters - 1, l=losses[n_iter], w0=w[0], w1=w[1]))

	return losses, ws


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
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
		learing rate

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

		for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches = 1):
			'''note if we choose a batch_iter(num_batches > 1) then this will not be
			updating properly because I use n_iter to index into ws and compute the loss 
			which does not increase if we loop throug this for loop multiple times. Try putting 2 
			in num_batches you will see what I am saying.'''
			gradient = compute_gradient(batch_y, batch_tx, ws[n_iter])
			w = ws[n_iter] - gamma * gradient
		
			ws.append(w)
			loss = compute_mse(y, tx, ws[n_iter+1])
			losses.append(loss)
			
			print("SGD({bi}/{ti}): loss={l:.6f}, w0={w0:.3f}, w1={w1:.3f}".format(
				bi=n_iter, ti=max_iters - 1, l=losses[n_iter], w0=w[0], w1=w[1]))

	return losses, ws