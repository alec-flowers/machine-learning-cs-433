# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""


def compute_stoch_gradient(y, tx, w):  #y and tx correspond to the mini-batch
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # implement stochastic gradient computation.It's same as the gradient descent.
    N = len(y)
    e = y - tx.dot(w)    #error vector
    g = -1/N * (np.transpose(tx)).dot(e)   #g = stochastic gradient
    return e, g

    
    
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    
    from helpers import batch_iter
    
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size = batch_size, num_batches = 1):
        
            """Compute gradient and loss"""
            e, gradient = compute_stoch_gradient(y_batch, tx_batch, w)
            loss = compute_loss_MSE(y, tx, w)
        
            w = w - gamma*gradient  #w is w(t+1) = w(t) - gamma*gradient(w(t))
        
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws