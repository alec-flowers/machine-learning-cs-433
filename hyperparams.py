from timeit import default_timer as timer

import numpy as np
import argparse
from os import path

from kfold_cv import ParameterGrid, cross_validation, build_k_indices, build_folds
from helpers import write_json, read_json, read_training_set


def best_model_selection(model, hyperparameters, x, y, k_fold=4, seed=1):
    """
    Best Model Selection:

    Finds the best performing set of hyperparameters (considering accuracy) for a specific model with a grid search
    by generating all possible combinations of hyperparameters and determining their performance using k-fold
    cross validation.

    Parameters
    ----------
    model: string selecting ['gd', 'sgd', 'ridge', 'least_squares', 'logistic', 'regularized_logistic']
        Machine learning methods

    hyperparameters: dict containing a set of hyperparameters

    x : ndarray of shape (n_samples, n_features)
        Training data

    y : ndarray of shape (n_samples,)
        Array of labels

    k_fold : int
        Number of folds for the K-fold cross validation

    seed : random number generator

    Returns
    ----------
    hp_star : dict
        Best performing set of hyperparameters

    acc_star : float
        Accuracy given the best performing set of hyperparameters

    w : list
        Weights computed using the given model and the best performing set of hyperparameters
    acc_list: list
        list with the test accuracies for the last hyperparameter combination, needed for the boxplots
    learning_curve_list: list
        list with the test losses from cross validation, needed for the loss plots
    """

    # Combination of hyperparameters
    hyperparam = ParameterGrid(hyperparameters)
    loss = []
    accuracy = []
    weights = []
    poly_dict = {}

    # Loop over different combinations of hyperparameters to find the best one
    for hp in hyperparam:
        k_indices = build_k_indices(y, k_fold, seed)
        loss_list = []
        acc_list = []
        learning_curve_list = []

        # Performs K-Cross Validation using the selected model to get the minimum loss
        start = timer()
        for k in range(k_fold):
            # if polynomials were already calculated get from "cache"
            if k in poly_dict and hp['degrees'] in poly_dict[k]:
                train_x = poly_dict[k][hp['degrees']][0]
                train_y = poly_dict[k][hp['degrees']][1]
                test_x = poly_dict[k][hp['degrees']][2]
                test_y = poly_dict[k][hp['degrees']][3]
            # if polynomials not calculated, calculate and save in "cache"
            else:
                train_x, train_y, test_x, test_y = build_folds(y, x, k_indices, k, hp)
                poly_dict[k] = {}
                poly_dict[k][hp['degrees']] = [train_x, train_y, test_x, test_y]
            # do cross validation
            loss_tr, loss_te, acc, weight, learning_curve = cross_validation(train_x, train_y, test_x, test_y, hp,
                                                                             model)
            loss_list.append(loss_te)
            acc_list.append(acc)
            learning_curve_list.append(learning_curve)
        loss.append(np.mean(loss_list))  # This is a list of loss* for each group of hyperparameters
        accuracy.append(np.mean(acc_list))
        weights.append(weight)
        end = timer()
        print(
            f'Hyperparameters: {hp}  Avg Loss: {np.mean(loss_list):.5f} Avg Accuracy: {accuracy[-1]:.4f} Time: {(end - start):.3f}')

    # Selection of the best hyperparameters considering accuracy
    min_acc_idx = np.argmax(accuracy)
    acc_star = accuracy[min_acc_idx]
    hp_star = list(hyperparam)[min_acc_idx]
    # needs params as a list for the enumeration in ParameterGrid to work
    hp_star = {key: [value] for key, value in
               hp_star.items()}
    w = weights[min_acc_idx]

    return hp_star, acc_star, w, acc_list, learning_curve_list


def read_init_hyperparam(model):
    """
    Reads the input collection of hyperparameters for a given model and loads the training dataset.
    """
    HYPERPARAMS_FOLDER = 'hyperparams_weights/'
    HYPERPARAMS_INIT_VALUES = 'init_hyperparams.json'

    return read_json(path.join(HYPERPARAMS_FOLDER, HYPERPARAMS_INIT_VALUES))[model]


def save_hyperparams(model, hp_star):
    """
    Saves the best performing set of hyperparameters (hp_star) of a given model into a .json file
    """
    filename = f"hyperparams_weights/best_hyperparams_{model}.json"
    write_json(filename, hp_star)


def find_hyperparams(model):
    """
    Main function which finds the best performing set of hyperparameters for a given model (['gd', 'sgd', 'ridge',
    'least_squares', 'logistic', 'regularized_logistic']) and saves them into a .json file.
    """
    print(f"Loading data...")
    y, x, ids_train = read_training_set()
    hyperparameters = read_init_hyperparam(model)
    print(f"Starting selection of best performing hyperparameters of {model}...")
    hp_star, acc_star, weights, accuracy, _ = best_model_selection(model, hyperparameters, x, y, k_fold=4, seed=1)
    print("Finished...")
    print(f'Best performing hyperparameters: {hp_star}  , Accuracy: {acc_star:.5f} , Weights: {weights}')
    print(f"Saving best performing hyperparameters as "f"best_hyperparams_{model}.json...")
    save_hyperparams(model, hp_star)


def parse_args():
    parser = argparse.ArgumentParser(description="CLI for the hyperparams_weights.py which finds the best hyperparameters.")
    parser.add_argument('-m', '--method', type=str, help='Which method to use to predict.', required=True,
                        choices=['sgd', 'gd', 'ridge', 'least_squares', 'logistic', 'regularized_logistic'])

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    find_hyperparams(args.method)
