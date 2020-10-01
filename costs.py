# -*- coding: utf-8 -*-
"""Functions used to compute the loss."""
    
def compute_mse(y, tx, w):
    """Calculate the loss using MSE (Mean Squared Error). """
    N = len(y)
    loss = 1/(2*N)*np.sum((y - tx.dot(w))**2)   
    return loss

def compute_mae(y, tx, w):
    """Calculate the loss using MAE (Mean Absolute Error)."""
    N = len(y)
    loss = 1/(2*N)*np.sum(np.abs(y - tx.dot(w)))  
    return loss

    #I am editing from gitlab.com
