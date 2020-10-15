from timeit import default_timer as timer

import numpy as np

from proj1_helpers import load_csv_data
from kfold_cv import ParameterGrid, cross_validation, build_k_indices
from helpers import write_json, build_poly


def best_model_selection(model, hyperparameters, x, y, k_fold=4, seed=1):
    hyperparam = ParameterGrid(hyperparameters)  ###Alec's function!!! D'aquí fa una combinació dels hiperparàmetres
    loss = []
    weights = []
    poly_dict = {}

    for hp in hyperparam:
        k_indices = build_k_indices(y, k_fold, seed) #!!
        loss_list = []

        if 'degrees' in hyperparameters.keys():
            if hp['degrees'] in poly_dict:
                px = poly_dict[hp['degrees']]
            else:
                start = timer()
                px, ind = build_poly(x, hp['degrees'])
                poly_dict[hp['degrees']] = px
                end = timer()
                print(f'Poly Time: {end-start:.3f}')
        else:
            px = x

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


        #JO FAIG EL QUE SERIA EL CROSS_VALIDATION_DEMO:
        #selecciono un set d'hiperparàmetres i li passo a la funció de l'Alec amb el model que selecciono.
        #amb això ell calcularà (amb cross validation) w* i loss*
        #això ho he de fer amb un LOOP per tots els sets d'hiperparàmetres (que els selecciono amb el parameters_grid())
        #llavors tindré una llista de loss* i w* pels diferents hiperparàmetres
        #d'aquesta llista he de triar el millor loss: loss**, w** i hiperparàmetres** per un tipus de model

        #podria fer això per tots els models i llavors triar el super millor loss*** on triaria quin és el millor model


if __name__ == "__main__":
    DATA_FOLDER = 'data/'
    TRAIN_DATASET = DATA_FOLDER + "train.csv"

    start = timer()
    y, x, ids_train = load_csv_data(TRAIN_DATASET, sub_sample = True)
    end = timer()
    print(f'Data Loaded - Time: {end-start:.3f}\n')

    #Ridge Regression
    model = 'ridge'
    hyperparameters = {'degrees':[1, 2],
                        'lambda':np.logspace(-4, 0, 15)}

    hp_star, loss_star, weights = best_model_selection(model, hyperparameters, x, y, k_fold=4, seed=1)
    print(f'Best Parameters - loss*: {loss_star:.5f}, hp*: {hp_star}')  #, weights: {weights}')

    # Gradient Descent
    # model = 'gd'
    # hyperparameters = {'initial_w':[[0 for _ in range(x.shape[1]+1)]],
    #                     'max_iters':[500], 
    #                     'gamma':[.00000001]}

    # hp_star, loss_star, weights = best_model_selection(model, hyperparameters, x, y, k_fold=2, seed=1)
    # print('loss*: {}, hp*: {}, weights: {}'.format(loss_star, hp_star, weights))

    # write_json('ridge_bp.json', hp_star)