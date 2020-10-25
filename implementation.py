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


def gradient_descent(y, tx, initial_w, max_iters, epsilon, gamma, error_type='MSE'):
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

    for iter in range(max_iters):
        gradient = compute_gradient(y, tx, ws[-1])
        w = ws[-1] - gamma * gradient
        loss = compute_loss(y, tx, w, error_type)

        ws.append(w)
        losses.append(loss)

        if iter % int(max_iters/5) == 0:
            print("GD({bi}/{ti}): loss={l:.6f}".format(bi=iter, ti=max_iters - 1, l=losses[-1]))

        # converge criterion
        # if len(losses) > 2 and np.abs(losses[-1] - losses[-2]) < epsilon:
        #     print('Epsilon Break')
        #     break
    return ws[-1], losses[-1]


def stochastic_gradient_descent(y, tx, initial_w, max_iters, batch_size, epsilon, gamma, num_batches):
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

    epsilon: float
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

    for iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=num_batches):
            gradient = compute_gradient(batch_y, batch_tx, ws[-1])
            w = ws[-1] - gamma * gradient
            loss = compute_mse(y, tx, ws[-1])

            ws.append(w)
            losses.append(loss)

        if iter % int(max_iters/5) == 0:
            print("SGD({bi}/{ti}): loss={l:.6f}, w0={w0:.3f}, w1={w1:.3f}".format(bi=iter, ti=max_iters - 1, l=losses[-1], w0=w[0], w1=w[1]))

        # converge criterion
        # if len(losses) > 2 and np.abs(losses[-1] - losses[-2]) < epsilon:
        #     print('Epsilon Break')
        #     break
    return ws[-1], losses[-1]



""" Least squares """

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



"""Ridge regression"""

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


def logistic_regression(y, tx, initial_w, max_iters, threshold, gamma, batch_size=1, num_batches=1):
    # Logistic regression (using Stochastic Gradient Descent):
    losses = []
    ws = [initial_w]

    for iter in range(max_iters):

        # Learning by stochastic gradient descent
        w = ws[-1]
        for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=num_batches):
            gradient = calculate_gradient_logistic(y, tx, w)
            w = w - (gamma) * gradient
            loss = calculate_logistic_loss(y, tx, w)

        losses.append(loss)
        ws.append(w)

        if iter % int(max_iters/5) == 0:
            print(f"Current iteration={iter}, loss={loss}")

        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    print("loss={l}".format(l=calculate_logistic_loss(y, tx, w)))
    return ws[-1], losses[-1]


# Regularized logistic regression (using SGD):
def regularized_logistic_regression(y, tx, initial_w, max_iters, threshold, gamma, lambda_, batch_size = 1,
                                          num_batches = 1):
    losses = []
    ws = [initial_w]
    for iter in range(max_iters):
        # Learning by stochastic gradient descent
        w = ws[-1]
        for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=num_batches):
            gradient = calculate_gradient_logistic(y, tx, w)
            lamb = 2 * lambda_ * np.array(w)
            gradient = gradient + lamb
            w = w - gamma * gradient
            loss = calculate_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T @ w)

        losses.append(loss)
        ws.append(w)

        if iter % int(max_iters/5) == 0:
            print("Current iteration = {i}, loss = {l}".format(i=iter, l=loss))

        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return ws[-1], losses[-1]

