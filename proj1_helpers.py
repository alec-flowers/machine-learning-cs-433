# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def load_csv_data(data_path, skip_header = 1, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=skip_header, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=skip_header)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    #column 4 stays delete (5, 6, 12, 26, 27, 28)
    #column 9 stays delete (21, 29)
    #x = np.delete(x,[5,6,12,21,26,27,28,29],axis = 1)

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = 0
    yb[np.where(np.char.startswith(y,'-'))] = 0
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def save_csv_data(data_path, data):
    print('---Saving Data---')
    np.savetxt(data_path, data, delimiter = ",", header = 'Take some space')

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})