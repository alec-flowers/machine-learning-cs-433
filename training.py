import argparse
from os import path

from data_io import write_json, read_training_set, read_best_hyperparameters
from kfold_cv import cross_validation, build_folds


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI for the training.py which finds the trains a given model with its best hyperparameters.")
    parser.add_argument('-m', '--method', type=str, help='Which method to train.', required=True,
                        choices=['sgd', 'gd', 'ridge', 'least_squares', 'logistic', 'regularized_logistic'])

    args = parser.parse_args()

    return args


def train(model):
    """
    Main function which trains a given model (['gd', 'sgd', 'ridge', 'least_squares', 'logistic', 'regularized_logistic'])
    using 0-fold Cross Validation (no test fold), finds and saves the weights with the corresponding hyperparameters.
    """

    y, x, ids_train = read_training_set()
    hyperparameters = read_best_hyperparameters(model)

    # do data parameters
    if 'degrees' not in hyperparameters.keys():
        raise KeyError("Hyperparameters should at least have degree as a key!")
    # build folds needs hyperparameter values as individual values and not as lists
    hyperparameters = {key: hyperparameters[key][0] for key in hyperparameters.keys()}
    train_x, train_y, test_x, test_y = build_folds(y, x, [], -1, hyperparameters, cross_validate=False)
    print(f"Starting training of {model}...")
    loss_tr, _, _, weight, _ = cross_validation(train_x, train_y, test_x, test_y, hyperparameters, model)
    print("Finished...")
    print("Loss training: " + str(loss_tr))
    hyperparameters['weights'] = weight.tolist()

    WEIGHTS_DIR = "hyperparams_weights"
    WEIGHTS_FILENAME = f"weights_{model}.json"
    print(f"Saving weights and corresponding hyperparameters to {path.join(WEIGHTS_DIR, WEIGHTS_FILENAME)}")
    write_json(path.join(WEIGHTS_DIR, WEIGHTS_FILENAME), hyperparameters)


if __name__ == '__main__':
    args = parse_args()
    train(args.method)
