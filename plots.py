# -*- coding: utf-8 -*-
"""function for plot."""
import matplotlib.pyplot as plt
import numpy as np
import argparse

from hyperparams import best_model_selection
from helpers import models, model_to_string, read_training_set, read_best_hyperparameters


def viz_accuracy(k_folds=10, seed=1):
    fig, axs = plt.subplots(nrows=1, ncols=1)
    y, x, ids_train = read_training_set()

    all_accuracies = []
    for idx, model in enumerate(models):
        hyperparameters = read_best_hyperparameters(model)
        _, _, _, accuracies, _ = best_model_selection(model, hyperparameters, x, y, k_fold=k_folds, seed=seed)
        all_accuracies.append(accuracies)
    axs.boxplot(all_accuracies, labels=[model_to_string[model] for model in models])
    axs.set_ylabel("Accuracy on the Test Fold", fontsize=26)
    axs.tick_params(axis='both', labelsize=26)
    fig.suptitle("Boxplot of the Accuracies", fontsize=32)
    plt.show()


def learning_curve_plot(learning_curve, model, hp):
    fig, ax = plt.subplots(figsize=(11, 6))
    avg_test = []
    avg_train = []

    for fold in learning_curve:
        iters = fold[0]
        test = fold[1]
        train = fold[2]
        ax.plot(iters, test, color='orange', alpha=.5, linewidth=.5, linestyle='--')
        ax.plot(iters, train, color='b', alpha=.5, linewidth=.5, linestyle='--')

        avg_test.append(test)
        avg_train.append(train)

    ax.plot(iters, np.mean(np.array(avg_test), axis=0), color='orange', linewidth=1.2, label='Avg Test Error')
    ax.plot(iters, np.mean(np.array(avg_train), axis=0), color='b', linewidth=1.2, label='Avg Train Error')

    ax.legend(loc='upper right', fontsize=24)
    ax.set_xlabel('Iters', fontsize=26)
    if 'logistic' in model:
        ax.set_ylabel('L2 Loss', fontsize=26)
    else:
        ax.set_ylabel('MSE', fontsize=26)
    ax.set_title('Learning Curve Convergence', fontsize=32)
    ax.tick_params('both', labelsize=18)

    fig.savefig(f'./img/Learning_Curve_{model}_{hp}.pdf')
    print(f'Plot Saved as - Learning_Curve_{model}_{hp}.pdf')
    plt.show()


def viz_loss(model, k_folds=4, seed=1):
    y, x, ids_train = read_training_set()
    hyperparameters = read_best_hyperparameters(model)
    _, _, _, _, learning_curves_list = best_model_selection(model, hyperparameters, x, y, k_fold=k_folds, seed=seed)
    learning_curve_plot(learning_curves_list, model, hyperparameters)


def parse_args():
    parser = argparse.ArgumentParser(description="CLI for the plot.py which is used for plotting.")
    parser.add_argument('-p', '--plot', type=str, help='Which plot to generate.', required=True,
                        choices=['box', 'gd', 'sgd', 'logistic', 'regularized_logistic'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.plot == 'box':
        viz_accuracy()
    else:
        viz_loss(args.plot)
