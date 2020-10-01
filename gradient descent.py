# -*- coding: utf-8 -*-
"""Gradient Descent"""

from grid_search import compute_loss

def compute_gradient(y, tx, w):
    #Janet
    
    """Compute the gradient."""
    N = len(y)
    e = y - tx.dot(w)    #error vector
    gradient = -1/N * (np.transpose(tx)).dot(e) #és una multiplicació escalar! perquè sinó no suma els elements
                #tx: 2xN,     e: Nx1 (o al revés però tan és)
    return e, gradient


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    #Janet
    
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        """Compute gradient and loss"""
        e, gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        w = w - gamma*gradient  #w is w(t+1) = w(t) - gamma*gradient(w(t))
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws