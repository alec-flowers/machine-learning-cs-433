import numpy as np

from helper_files.costs import compute_loss, compute_error, calculate_logistic_loss
from helper_files.helpers import batch_iter, sigmoid


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


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
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

    max_iters:
        Maximum iterations to do

    gamma : float
        learing rate

    Returns
    ----------
    w : np.array of shape(1, D)
        Optimal weights calculated.

    loss : np.float64
        MSE loss for corresponding weight value
    """
    # initialize
    error_type = 'MSE'
    W0 = 0
    ws = [initial_w]
    losses = [compute_loss(y, tx, ws[W0], error_type)]
    # do gradient descent
    for iter in range(max_iters):
        gradient = compute_gradient(y, tx, ws[-1])
        w = ws[-1] - gamma * gradient
        loss = compute_loss(y, tx, w, error_type)
        ws.append(w)
        losses.append(loss)
        if iter % int(max_iters / 5) == 0:
            print("GD({bi}/{ti}): loss={l:.6f}".format(bi=iter, ti=max_iters - 1, l=losses[-1]))
    return ws[-1], losses[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
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

    max_iters:
        Maximum iterations to do

    gamma : float
        learing rate


    Returns
    ----------
    w : np.array of shape(1, D)
        Optimal weights calculated.

    loss : np.float64
        MSE loss for corresponding weight value
    """
    # init
    error_type = 'MSE'
    batch_size = 1
    num_batches = 1
    W0 = 0
    ws = [initial_w]
    losses = [compute_loss(y, tx, ws[W0], error_type)]
    # do stochastic gradient descent
    for iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=num_batches):
            gradient = compute_gradient(batch_y, batch_tx, ws[-1])
            w = ws[-1] - gamma * gradient
            loss = compute_loss(y, tx, w, error_type)
            ws.append(w)
            losses.append(loss)
        if iter % int(max_iters / 5) == 0:
            print(
                "SGD({bi}/{ti}): loss={l:.6f}, w0={w0:.3f}, w1={w1:.3f}".format(bi=iter, ti=max_iters - 1, l=losses[-1],
                                                                                w0=w[0], w1=w[1]))
    return ws[-1], losses[-1]


def least_squares(y, tx):
    """Least Squares algorithm.

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


def ridge_regression(y, tx, lambda_):
    """Ridge Regression algorithm.

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
        Optimal weights calculated using normulaequations.

    loss : np.float64
        MSE loss for corresponding weight value

    """

    N = len(tx)
    D = tx.shape[1]

    w = np.linalg.solve(tx.T.dot(tx) + 2 * N * lambda_ * np.identity(D), tx.T.dot(y))
    loss = compute_loss(y, tx, w, 'MSE')

    return w, loss


def calculate_gradient_logistic(y, tx, w):
    """Compute a gradient for Logistic Regression loss.

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
            gradient of Logistic Regression loss

        """

    gradient = tx.T @ (sigmoid(tx @ w) - y)
    return gradient


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic Gradient Descent algorithm.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Array of labels

    tx : ndarray of shape (n_samples, n_features)
        Training data

    initial_w : ndarray of shape (n_weights,)
        Weight vector

    max_iters:
        Maximum iterations to do

    gamma : float
        learing rate

    Returns
    ----------
    w : np.array of shape(1, D)
        Optimal weights calculated.

    loss : np.float64
        MSE loss for corresponding weight value

    """
    losses = []
    ws = [initial_w]
    # using gradient descent
    for iter in range(max_iters):
        w = ws[-1]
        gradient = calculate_gradient_logistic(y, tx, w)
        w = w - (gamma) * gradient
        loss = calculate_logistic_loss(y, tx, w)
        losses.append(loss)
        ws.append(w)
        if iter % int(max_iters / 5) == 0:
            print(f"Current iteration={iter}, loss={loss}")
    print("loss={l}".format(l=calculate_logistic_loss(y, tx, w)))
    return ws[-1], losses[-1]


def regularized_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Logistic Gradient Descent algorithm.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Array of labels

    tx : ndarray of shape (n_samples, n_features)
        Training data

    initial_w : ndarray of shape (n_weights,)
        Weight vector

    max_iters:
        Maximum iterations to do

    gamma : float
        learing rate

    Returns
    ----------
    w : np.array of shape(1, D)
        Optimal weights calculated.

    loss : np.float64
        MSE loss for corresponding weight value

    """
    losses = []
    ws = [initial_w]
    for iter in range(max_iters):
        # Learning by gradient descent
        w = ws[-1]
        gradient = calculate_gradient_logistic(y, tx, w)
        lamb = np.multiply(lambda_, w)
        gradient = np.add(gradient, lamb)
        w = w - gamma * gradient
        loss = calculate_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T @ w)
        losses.append(loss)
        ws.append(w)
        if iter % int(max_iters / 5) == 0:
            print("Current iteration = {i}, loss = {l}".format(i=iter, l=loss))
    print("loss={l}".format(l=calculate_logistic_loss(y, tx, w)))
    return ws[-1], losses[-1]
