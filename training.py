from timeit import default_timer as timer

import argparse
from os import path

from proj1_helpers import load_csv_data
from helpers import read_json, build_poly, write_json
from kfold_cv import cross_validation


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI for the training.py which finds the trains a given model with its best hyperparameters.")
    parser.add_argument('-m', '--method', type=str, help='Which method to train.', required=True,
                        choices=['sgd', 'gd', 'ridge', 'least_squares', 'logistic', 'regularized_logistic'])

    args = parser.parse_args()

    return args


def read_best_hyperparameters(model):
    HYPERPARAMS_FOLDER = 'hyperparams/'
    HYPERPARAMS_BEST_VALUES = f'best_hyperparams_{model}.json'
    filename = path.join(HYPERPARAMS_FOLDER, HYPERPARAMS_BEST_VALUES)
    return read_json(filename)


def read_training_set():
    # model = 'gd'

    DATA_FOLDER = 'Data/'
    TRAIN_DATASET = path.join(DATA_FOLDER, "train.csv")

    start = timer()
    y, x, ids_train = load_csv_data(TRAIN_DATASET, sub_sample=True)
    print(f'Data Loaded - Time: {timer() - start:.3f}\n')

    return y, x, ids_train


def train(model):
    y, x, ids_train = read_training_set()
    hyperparameters = read_best_hyperparameters(model)

    # build polynomial from hyperparams
    if 'degrees' in hyperparameters.keys():
        start = timer()
        px, ind = build_poly(x, hyperparameters['degrees'])
        print(f'Poly Time: {timer() - start:.3f}')
    else:
        raise KeyError("Hyperparameters should at least have degree as a key!")

    print(f"Starting training of {model}...")
    loss_tr, _, _, weight = cross_validation(y, px, [], 0, hyperparameters, model, cross_validate=False)
    print("Finished...")
    print("Loss training: " + str(loss_tr))
    hyperparameters['weights'] = weight.tolist()

    WEIGHTS_DIR = "hyperparams"
    WEIGHTS_FILENAME = f"weights_{model}.json"
    print(f"Saving weights and corresponding hyperparameters to {path.join(WEIGHTS_DIR, WEIGHTS_FILENAME)}")
    write_json(path.join(WEIGHTS_DIR, WEIGHTS_FILENAME), hyperparameters)


if __name__ == '__main__':
    args = parse_args()
    train(args.method)
