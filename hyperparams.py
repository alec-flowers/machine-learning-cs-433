from timeit import default_timer as timer

import numpy as np
import argparse
from os import path

from proj1_helpers import load_csv_data
from kfold_cv import ParameterGrid, cross_validation, build_k_indices
from helpers import write_json, read_json


def parse_args():
    parser = argparse.ArgumentParser(description="CLI for the hyperparams.py which finds the best hyperparameters.")
    parser.add_argument('-m', '--method', type=str, help='Which method to use to predict.', required=True,
                        choices=['sgd', 'gd', 'ridge', 'least_squares', 'logistic', 'regularized_logistic'])

    args = parser.parse_args()

    return args


def best_model_selection(model, hyperparameters, x, y, k_fold=4, seed=1):
    """
    This function iterates over the hyperparameters of a given model to find the best set of hyperparameters,
    which give the minimum loss.
    Performs k-fold Cross Validation over each set of hyperparameters using the selected model.

    Parameters
    ----------
    model: string selecting ['gd', 'sgd', 'ridge', 'least_squares', 'logistic', 'regularized_logistic']
        Machine learning methods

    hyperparameters:

    x:

    y:

    k_fold:

    seed:

    Returns
    ----------
    TODO
    """

    # Combination of hyperparameters
    hyperparam = ParameterGrid(hyperparameters)
    loss = []
    accuracy = []
    weights = []

    # Loop over different combinations of hyperparameters to find the best one
    for hp in hyperparam:
        k_indices = build_k_indices(y, k_fold, seed)
        loss_list = []
        acc_list = []

        # Performs K-Cross Validation using the selected model to get the minimum loss
        start = timer()
        for k in range(k_fold):
            loss_tr, loss_te, acc, weight = cross_validation(y, x, k_indices, k, hp, model)
            loss_list.append(loss_te)
            acc_list.append(acc)
        loss.append(np.mean(loss_list))  # This is a list of loss* for each group of hyperparameters
        accuracy.append(np.mean(acc_list))
        weights.append(weight)
        end = timer()

        print(
            f'Hyperparameters: {hp}  Avg Loss: {np.mean(loss_list):.5f} Avg Accuracy: {accuracy[-1]:.4f} Time: {(end - start):.3f}')

    # loss_star = min(loss)  # This is the best loss*, which corresponds to a specific set of hyperparameters
    # hp_star = list(hyperparam)[loss.index(loss_star)]  # this is the hyperparameter that corresponds to the best loss*

    min_acc_idx = np.argmax(accuracy)  # This is the index of the best accuracy
    acc_star = accuracy[min_acc_idx]
    hp_star = list(hyperparam)[min_acc_idx]  # corresponding hyperparameters
    w = weights[min_acc_idx]  # corresponding weights

    return hp_star, acc_star, w


def save_hyperparams(model, hp_star):
    filename = f"hyperparams/best_hyperparams_{model}.json"
    hp_star['model'] = model
    write_json(filename, hp_star)


def find_hyperparams(model):
    y, x, ids_train, hyperparameters = read_hyperparam_input(model)
    hp_star, loss_star, weights = best_model_selection(model, hyperparameters, x, y, k_fold=4, seed=1)
    save_hyperparams(model, hp_star)


def read_hyperparam_input(model):
    DATA_FOLDER = 'Data/'
    TRAIN_DATASET = path.join(DATA_FOLDER, "train.csv")
    HYPERPARAMS_FOLDER = 'hyperparams/'
    HYPERPARAMS_INIT_VALUES = 'init_hyperparams.json'

    start = timer()
    y, x, ids_train = load_csv_data(TRAIN_DATASET, sub_sample=True)
    hyperparameters = read_json(path.join(HYPERPARAMS_FOLDER, HYPERPARAMS_INIT_VALUES))[model]
    end = timer()
    print(f'Data Loaded - Time: {end - start:.3f}\n')

    return y, x, ids_train, hyperparameters


if __name__ == "__main__":
    args = parse_args()
    find_hyperparams(args.method)