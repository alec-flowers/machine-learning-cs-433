from timeit import default_timer as timer

import numpy as np

from proj1_helpers import load_csv_data
from kfold_cv import ParameterGrid, cross_validation, build_k_indices
from helpers import write_json, build_poly


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
    weights = []
    poly_dict = {}

    # Loop over different combinations of hyperparameters to find the best one
    for hp in hyperparam:
        k_indices = build_k_indices(y, k_fold, seed)
        loss_list = []

        # Making polynomial if asked
        if 'degrees' in hyperparameters.keys():
            # Checks if the polynomial has already been calculated
            if hp['degrees'] in poly_dict:
                px = poly_dict[hp['degrees']]
            # Calculates the polynomial and saves it in a dictionary
            else:
                start = timer()
                px, ind = build_poly(x, hp['degrees'])
                poly_dict[hp['degrees']] = px
                end = timer()
                print(f'Poly Time: {end-start:.3f}')
        else:
            px = x

        # Performs K-Cross Validation using the selected model to get the minimum loss
        start = timer()
        for k in range(k_fold):
            loss_tr, loss_te, weight = cross_validation(y, px, k_indices, k, hp, model)
            loss_list.append(loss_te)
        loss.append(np.mean(loss_list))   #This is a list of loss* for each group of hyperparameters
        weights.append(weight)
        end = timer()

        print(f'Hyperparameters: {hp}  Avg Loss: {np.mean(loss_list):.5f}  Time: {end-start:.3f}')

    loss_star = min(loss)   #This is the best loss*, which corresponds to a specific set of hyperparameters
    hp_star = list(hyperparam)[loss.index(loss_star)]  #this is the hyperparameter that corresponds to the best loss*
    w = weights[loss.index(loss_star)]

    return(hp_star, loss_star, w)




if __name__ == "__main__":
    DATA_FOLDER = 'data/'
    TRAIN_DATASET = DATA_FOLDER + "train.csv"

    start = timer()
    y, x, ids_train = load_csv_data(TRAIN_DATASET, sub_sample = True)
    end = timer()
    print(f'Data Loaded - Time: {end-start:.3f}\n')

    ### Ridge Regression test
    #model = 'ridge'
    #hyperparameters = {'degrees':[1, 2], 'lambda':np.logspace(-4, 0, 15)}
    #hp_star, loss_star, weights = best_model_selection(model, hyperparameters, x, y, k_fold=4, seed=1)
    #print(f'Best Parameters found with {model}: - loss*: {loss_star:.5f}, hp*: {hp_star}')  #, weights: {weights}')


    ### Gradient Descent test
    #model = 'gd'
    #hyperparameters = {'initial_w':[[0 for _ in range(x.shape[1]+1)]],
    #                     'max_iters':[500],
    #                     'gamma':[.00000001]}
    #hp_star, loss_star, weights = best_model_selection(model, hyperparameters, x, y, k_fold=2, seed=1)
    #print(f'Best Parameters found with {model}: - loss*: {loss_star:.5f}, hp*: {hp_star} , weights: {weights}')

    ### Stochastic Gradient Descent test
    #model = 'sgd'
    #hyperparameters = {'initial_w': [[0 for _ in range(x.shape[1] + 1)]],
    #                   'max_iters': [500],
    #                   'gamma': [.00000001],
    #                   'num_batches': [2,6],
    #                   'batch_size': [2]}
    #hp_star, loss_star, weights = best_model_selection(model, hyperparameters, x, y, k_fold=2, seed=1)
    #print(f'Best Parameters found with {model}: - loss*: {loss_star:.5f}, hp*: {hp_star} , weights: {weights}')

    ### Least squares test
    model = 'least_squares'
    hyperparameters = {}  #!!! This one has no hyperparameters, so maybe we should print it differently
    hp_star, loss_star, weights = best_model_selection(model, hyperparameters, x, y, k_fold=2, seed=1)
    print(f'Best Parameters found with {model}: - loss*: {loss_star:.5f}, hp*: {hp_star} , weights: {weights}')


    # write_json('ridge_bp.json', hp_star)