# Predicting Higgs Boson with Machine Learning
The Higgs boson is a particle which gives mass to other elementary particles. In this project, we are using data provided by the ATLAS collaboration at CERN and perform machine learning methods to predict signal or background from an observed event.

## Table of contents
* [Overview](#Overview)
    * [a) Instructions for KFold CV and Hyperparameter Search](#Instructions for KFold CV and Hyperparameter Search)
    * [b) Instructions for optimal weights selection and submission](#Instructions for optimal weights selection and submission)
* [File structure](#File structure)
* [Setup](#Setup)

## Overview
There are two workflows that we have built which serve seperate purposes.

a) The first workflow is for testing and tinkering to try and find the best combination of hyperparameters and feature selection. For this, K-fold Cross Validation is performed over all possible combinations of hyperparameters for a given model.

b) The second workflow is for creating the best model and weights for a submission attempt. 


### a) Instructions for KFold CV & Hyperparameter Search:
1) In `init_hyperparams.json` adjust the hyperparameters for all the models or the model of interest. Do not edit the keys as the keys represent the hyperparameters. 
2) Run `hyperparams.py`. For this, use the command line: 

    ```
    $ python3 hyperparams.py -m <name_of_model>
    ```
    Where `<name_of_model>` is a choice between: 'gd', 'sgd', 'ridge', 'least_squares', 'logistic' and 'regularized_logistic', corresponding to each machine learning method.
3) The best performing set of hyperparameters will be saved at `~\hyperparams`, with the file name `best_hyperparameters_{model}.json` according to the method. 

### b) Instructions for optimal weights selection & submission
1) Use the best parameters from `\hyperparams` computed previously in (a).
2) Run `training.py`. For this, use the command line: 

    ```
    $ python3 training.py -m <name_of_model>
    ```
    Where `<name_of_model>` is a choice between: 'gd', 'sgd', 'ridge', 'least_squares', 'logistic' and 'regularized_logistic', corresponding to each machine learning method.
3) The computed weights and the corresponding hyperparameters will be saved at `~\hyperparams`, with the file name `weights_{model}.json` according to the method. 
4) Run `run.py   !!!!!

```
$ python3 run.py
```



## File Structure
Here is the file structure of the project: 
```bash
Project
|
|-- Data
|   |-- test.csv
|    |-- train.csv
|
|-- hyperparams
|    |-- best_hyperparams_gd.json
|    |-- best_hyperparams_least_squares.json
|    |-- best_hyperparams_ridge.json
|    |-- best_hyperparams_sgd.json
|    |-- init_hyperparams.json
|
|-- costs.py
|-- data_process.py
|-- grid_search.py
|-- helpers.py
|-- hyperparams.py
|-- implementation.py
|-- kfold_cv.py
|-- plots.py
|-- proj1_helpers.py
|-- run.py
|-- training.py
```

## Setup
```
numpy 1.19.2
matplotlib
```

Numpy is the one dependency to run all the code that is apart of the machine learning pipeline as stated in the project requirements. We only use plotting libraries like matplotlib to create plots for our report. 

Python Standard Library
```
timeit
default_timer
os
path
argparse
```


We use a couple of packages built-into the python standard library: `timeit` is used to give feedback on running time of various processes. `os` and `argparse` are used to take in user input from the command line to make selections and run the various programs. 