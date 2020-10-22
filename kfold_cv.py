from timeit import default_timer as timer

import numpy as np
from implementation import ridge_regression, gradient_descent, stochastic_gradient_descent, least_squares, \
    logistic_regression, regularized_logistic_regression
from costs import compute_loss


def product(*args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


class ParameterGrid:
    def __init__(self, grid):
        grid = [grid]
        self.grid = grid

    def __iter__(self):
        """Iterate over the points in the grid.
        Returns
        -------
        params : iterator over dict of str to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, hp, model, cross_validate=True):
    """return the loss of the specified model"""
    if cross_validate:
        assert k < len(k_indices), 'K is larger than the number of k-folds we create'
        # get k'th subgroup in test, others in train: TODO
        train_i = np.concatenate(np.delete(k_indices, k, axis=0))
        test_i = k_indices[k]

        train_x = x[train_i]
        train_y = y[train_i]
        test_x = x[test_i]
        test_y = y[test_i]
    else:
        train_x = x
        train_y = y
        test_x = np.zeros(x.shape)
        test_y = np.zeros(y.shape)

    # Calculation of losses using the specified model
    # gradient descent:
    if model == 'gd':
        initial_w = [0 for _ in range(x.shape[1])]
        epsilon = hp['epsilon']
        gamma = hp['gamma']

        weights, loss_tr = gradient_descent(train_y, train_x, initial_w, epsilon, gamma)
        loss_te = compute_loss(test_y, test_x, weights, 'MSE')

    # stochastic gradient descent:
    elif model == 'sgd':
        initial_w = [0 for _ in range(x.shape[1])]
        batch_size = hp['batch_size']
        num_batches = hp['num_batches']
        epsilon = hp['epsilon']
        gamma = hp['gamma']

        weights, loss_tr = stochastic_gradient_descent(train_y, train_x, initial_w, batch_size, epsilon, gamma,
                                                       num_batches)
        loss_te = compute_loss(test_y, test_x, weights, 'MSE')

    # least squares:
    elif model == 'least_squares':
        weights, loss_tr = least_squares(train_y, train_x)
        loss_te = compute_loss(test_y, test_x, weights, 'MSE')

    # ridge regression:
    elif model == 'ridge':
        lambda_ = hp['lambda']

        weights, loss_tr = ridge_regression(train_y, train_x, lambda_)
        loss_te = compute_loss(test_y, test_x, weights, 'MSE')

    # logistic regression: TODO
    elif model == 'logistic':
        initial_w = [0 for _ in range(x.shape[1])]
        max_iters = hp['max_iters']
        threshold = hp['threshold']
        gamma = hp['gamma']
        num_batches = hp['num_batches']
        batch_size = hp['batch_size']

        weights, loss_tr = logistic_regression(train_y, train_x, initial_w, max_iters, threshold, gamma, batch_size,
                                               num_batches)
        loss_te = compute_loss(test_y, test_x, weights, 'MSE')

    # regularized logistic regression: TODO
    elif model == 'regularized_logistic':
        initial_w = [0 for _ in range(x.shape[1])]
        max_iters = hp['max_iters']
        threshold = hp['threshold']
        gamma = hp['gamma']
        lambda_ = hp['lambda_']
        num_batches = hp['num_batches']
        batch_size = hp['batch_size']

        weights, loss_tr = regularized_logistic_regression(train_y, train_x, initial_w, max_iters, threshold, gamma,
                                                           lambda_, batch_size,
                                                           num_batches)
        loss_te = compute_loss(test_y, test_x, weights, 'MSE')


    return loss_tr, loss_te, weights
