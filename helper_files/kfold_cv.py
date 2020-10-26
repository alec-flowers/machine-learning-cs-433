from timeit import default_timer as timer
import numpy as np

from helper_files.implementations_private import ridge_regression, gradient_descent, stochastic_gradient_descent, least_squares, \
    logistic_regression, regularized_logistic_regression
from helper_files.costs import compute_loss, calc_accuracy
from helper_files.helpers import build_poly
from helper_files.data_pre_process import impute, standardize


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


def product(*args, repeat=1):
    """
    Creates all possible combinations between lists given in *args:
    product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    """
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def build_folds(y, x, k_indices, k, hp, cross_validate=True):
    """
    Builds the folds specified in the arguments.

    Parameters
    -------
    hp: dict containing a set of hyperparameters

    x : ndarray of shape (n_samples, n_features)
        Training data

    y : ndarray of shape (n_samples,)
        Array of labels

    k_indices : list
        list holding the indices for every fold
    k: int
        which fold to build
    cross_validate: bool
        if False, build no test fold. Used in training.py
    """
    print("Starting Pre-Processing...")
    x[x <= -999] = np.nan
    if cross_validate:
        assert k < len(k_indices), 'K is larger than the number of k-folds we create'
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

    train_median = np.nanmedian(train_x, axis=0)
    train_x = impute(train_x, train_median)
    test_x = impute(test_x, train_median)

    split = train_x.shape[0]
    temp_x = np.append(train_x, test_x, axis=0)

    # Making polynomial if asked
    if 'degrees' in hp.keys():
        start = timer()
        if 'poly_indices' in hp.keys():
            temp_x_append = np.delete(temp_x, hp['poly_indices'], axis = 1)
            temp_x = temp_x[:, hp['poly_indices']]
        poly_x, _ = build_poly(temp_x, hp['degrees'])
        if 'poly_indices' in hp.keys():
            poly_x = np.c_[poly_x, temp_x_append]

        end = timer()
        print(f'Poly Time: {end - start:.3f}')
    else:
        raise KeyError('Hyperparameter should have at least degree = 1')

    train_x = poly_x[:split]
    test_x = poly_x[split:]
    train_mean = np.nanmean(train_x, axis=0)
    train_sd = np.nanstd(train_x, axis=0)
    train_x = standardize(train_x, train_mean, train_sd)
    test_x = standardize(test_x, train_mean, train_sd)

    print("Pre-Processing finished...")
    return train_x, train_y, test_x, test_y


def cross_validation(train_x, train_y, test_x, test_y, hp, model):
    """
    Builds the folds specified in the arguments.

    Parameters
    -------
    train_x: ndarray of shape (n_train_samples, degree of polynomial expansion)

    train_y : ndarray of shape (n_train_samples, )
        Training data

    test_x: ndarray of shape (n_test_samples, degree of polynomial expansion)
        Array of labels

    test_y: ndarray of shape (n_test_samples,)
    hp: dict
        dict containing the hyperparameters (e.g. learning rate)
    model:
        which model to do cross validation for
    """

    # Calculation of losses using the specified model
    # gradient descent:
    learning_curve = None
    if model == 'gd':
        initial_w = [0 for _ in range(train_x.shape[1])]
        gamma = hp['gamma']
        max_iters = hp['max_iters']

        weights, loss_tr, learning_curve = gradient_descent(train_y, train_x, initial_w, max_iters, gamma, test_y,
                                                            test_x)
        loss_te = compute_loss(test_y, test_x, weights, 'MSE')

    # stochastic gradient descent:
    elif model == 'sgd':
        initial_w = [0 for _ in range(train_x.shape[1])]
        max_iters = hp['max_iters']
        batch_size = hp['batch_size']
        num_batches = hp['num_batches']
        gamma = hp['gamma']

        weights, loss_tr, learning_curve = stochastic_gradient_descent(train_y, train_x, initial_w, max_iters,
                                                                       batch_size, gamma,
                                                                       num_batches, test_y, test_x)
        loss_te = compute_loss(test_y, test_x, weights, 'MSE')

    # least squares:
    elif model == 'least_squares':
        weights, loss_tr = least_squares(train_y, train_x)
        loss_te = compute_loss(test_y, test_x, weights, 'MSE')

    # ridge regression:
    elif model == 'ridge':
        lambda_ = hp['lambda']

        weights, loss_tr = ridge_regression(train_y, train_x, lambda_)
        # calculate the loss for train and test data:
        loss_te = compute_loss(test_y, test_x, weights, 'MSE')

    elif model == 'logistic':
        initial_w = [0 for _ in range(train_x.shape[1])]
        max_iters = hp['max_iters']
        gamma = hp['gamma']
        num_batches = hp['num_batches']
        batch_size = hp['batch_size']

        weights, loss_tr, learning_curve = logistic_regression(train_y, train_x, initial_w, max_iters, gamma,
                                                               batch_size,
                                                               num_batches, test_y, test_x)
        loss_te = compute_loss(test_y, test_x, weights, 'MSE')

    elif model == 'regularized_logistic':
        initial_w = [0 for _ in range(train_x.shape[1])]
        max_iters = hp['max_iters']
        gamma = hp['gamma']
        lambda_ = hp['lambda_']
        num_batches = hp['num_batches']
        batch_size = hp['batch_size']

        weights, loss_tr, learning_curve = regularized_logistic_regression(train_y, train_x, initial_w, max_iters,
                                                                           gamma,
                                                                           lambda_, batch_size, num_batches, test_y,
                                                                           test_x)
        loss_te = compute_loss(test_y, test_x, weights, 'MSE')

    acc = calc_accuracy(test_y, test_x, weights, model)

    return loss_tr, loss_te, acc, weights, learning_curve
