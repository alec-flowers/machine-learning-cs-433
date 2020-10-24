import numpy as np

from costs import *
from helpers import batch_iter


# Machine Learning Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""Gradient Descent"""

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
    N = len(tx)
    e = compute_error(y, tx, w)
    gradient = (-1 / N) * tx.T.dot(e)
    return gradient


def gradient_descent(y, tx, initial_w, epsilon, gamma, error_type='MSE'):
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

    epsilon: float
        do gradient descent until residual is smaller than epsilon

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
    losses = [compute_loss(y, tx, ws[W0], error_type)]
    residual = 1e100
    i = 0

    while residual > epsilon or i < 1000:
        gradient = compute_gradient(y, tx, ws[-1])
        w = ws[-1] - gamma * gradient
        loss = compute_loss(y, tx, w, error_type)

        ws.append(w)
        losses.append(loss)
        residual = abs(losses[-1] - losses[-2])
        i += 1

        #print("GD({bi}/{ti}): loss={l:.6f}".format(bi=n_iter, ti=max_iters - 1, l=losses[-1]))

    return ws[-1], losses[-1]


def stochastic_gradient_descent(y, tx, initial_w, batch_size, epsilon, gamma, num_batches):
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
        size of  the batches

    epsilon : float
        do gradient descent until residual is smaller than epsilon

    gamma : float
        learing rate

    num_batches: int
        Number of mini batches for one iteration

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
    residual = 1e100
    i = 0

    while residual > epsilon or i < 1000:

        for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=num_batches):
            '''note if we choose a batch_iter(num_batches > 1) then this will not be
            updating properly because I use n_iter to index into ws and compute the loss 
            which does not increase if we loop throug this for loop multiple times. Try putting 2 
            in num_batches you will see what I am saying.'''
            gradient = compute_gradient(batch_y, batch_tx, ws[-1])
            w = ws[-1] - gamma * gradient
            loss = compute_mse(y, tx, ws[-1])

            ws.append(w)
            losses.append(loss)
        residual = abs(losses[-1] - losses[-2])
        i += 1

            #print("SGD({bi}/{ti}): loss={l:.6f}, w0={w0:.3f}, w1={w1:.3f}".format(bi=n_iter, ti=max_iters - 1, l=losses[-1], w0=w[0], w1=w[1]))

    return ws[-1], losses[-1]



""" Least squares """

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
        MSE loss for corresponding weight value

    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w, 'MSE')
    return w, loss



"""Ridge regression"""

def ridge_regression(y, tx, lambda_):
    """Ridge Regression

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Array of labels

    tx : ndarray of shape (n_samples, n_features)
        Training data

    lambda_ : float [0, 1]
        parsimony penalty

    Returns
    ----------
    w : np.array of shape(1, D)
        Optimal weights calculated using normula equations.

    loss : np.float64
        MSE loss for corresponding weight value

    """

    N = len(tx)
    D = tx.shape[1]

    w = np.linalg.solve(tx.T.dot(tx) + 2 * N * lambda_ * np.identity(D), tx.T.dot(y))
    loss = compute_loss(y, tx, w, 'MSE')

    return w, loss


"""Logistic regression"""
def calculate_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    gradient = tx.T @ (sigmoid(tx @ w) - y)
    return gradient


def logistic_regression(y, tx, initial_w, max_iters, threshold, gamma, batch_size=1, num_batches=1):
    # Logistic regression (using Stochastic Gradient Descent):
    losses = []
    ws = [initial_w]
    for iter in range(max_iters):

        # Learning by stochastic gradient descent
        w = ws[-1]
        for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=num_batches):
            gradient = calculate_gradient_logistic(y, tx, w)  # this is a subgradient
            w = w - gamma * gradient
            loss = calculate_logistic_loss(y, tx, w)

        losses.append(loss)  # we take the last value of loss and w !!!!!
        ws.append(w)  # we take the last value of loss and w !!!!!

        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    print("loss={l}".format(l=calculate_logistic_loss(y, tx, w)))
    return ws[-1], losses[-1]


# Regularized logistic regression (using SGD): TODO: check
def regularized_logistic_regression(y, tx, initial_w, max_iter, threshold, gamma, lambda_, batch_size = 1,
                                          num_batches = 1):
    losses = []
    ws = [initial_w]
    for iter in range(max_iter):
        # Learning by stochastic gradient descent
        w = ws[-1]
        for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=num_batches):
            gradient = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
            w = w - gamma * gradient
            loss = calculate_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T @ w)

        losses.append(loss)
        ws.append(w)

        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    print("loss={l}".format(l=calculate_logistic_loss(y, tx, w)))
    return ws[-1], losses[-1]

