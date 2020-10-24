# Machine Learning CS-433
This is the repo for the Machine Learning Course CS-433.

[Moodle Link](https://moodle.epfl.ch/course/view.php?id=14221)
[Overleaf Link](https://www.overleaf.com/2112266929fkchyyjkvvsw)
[ML Class Competition Page](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs)
[Old Competition Link](https://www.kaggle.com/c/higgs-boson/data)


# Predicting Higgs Boson with Machine Learning
## Overview


There are two workflows that we have built which serve seperate purposes.
1) The first workflow is for testing and tinkering to try and find the best combination of hyperparameters and feature selection.

2) The second workflow is for creating the best model and weights for a submission attempt. 

## Instructions for KFold CV & Hyperparameter Search:
1) In init_hyperparams.json adjust the hyperparameters for all the models or the model of interest. Do not edit the keys as the keys represent they hyperparameters 

## Instructions for optimal weights selection & submission


## File Structure
Here is the file structure of the project: 

- Data\
| - test.csv
| - train.csv

- hyperparams\
| - best_hyperparams_gd.json
| - best_hyperparams_least_squares.json
| - best_hyperparams_ridge.json
| - best_hyperparams_sgd.json
| - init_hyperparams.json

| - costs.py
| - data_process.py
| - grid_search.py
| - helpers.py
| - hyperparams.py
| - implementation.py
| - kfold_cv.py
| - plots.py
| - proj1_helpers.py
| - run.py
| - training.py

# Packages
| - numpy 1.19.2
| - matplotlib

Numpy is the one dependency to run all the code that is apart of the machine learning pipeline as stated in the project requirements. We only use plotting libraries like matplotlib to create plots for our report. 

Python Standard Library
| - timeit
| | - default_timer
| - os
| | - path
| - argparse

We use a couple packages built-into the python standard library. timeit is used to give feedback on running time of various processes. os and argparse are used to take in user input from the command line to make selections and run the various programs. 