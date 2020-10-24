from timeit import default_timer as timer
from proj1_helpers import load_csv_data, save_csv_data
import numpy as np


# Preprocessing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def impute(data, how = 'median'):
    '''
    Imputes the -999 values with either the mean or median of the column.
    '''

    data[data <= -999] = np.nan
    if how == 'mean':
        col_agg = np.nanmean(data, axis = 0)
    elif how == 'median':
        col_agg = np.nanmedian(data,axis = 0)
    
    inds = np.where(np.isnan(data))
    data[inds] = np.take(col_agg, inds[1])

    return data

def normalize(data):
    '''
    Normalizes data and subtract mean divide by standard deviation.
    '''
    mean = np.nanmean(data, axis = 0)
    std = np.nanstd(data, axis = 0)
    data = (data - mean) / std

    return data

def main():
    y, input_data_train, ids = load_csv_data("Data/train.csv", sub_sample = False)
    #yb_data_test, input_data_test, ids_test = load_csv_data("Data/test.csv")
    x = input_data_train

    #column 4 stays delete (5, 6, 12, 26, 27, 28)
    #column 9 stays delete (21, 29)
    x = np.delete(x,[5,6,12,21,26,27,28,29],axis = 1)

    data = np.c_[ids, y , x]
    start = timer()
    save_csv_data('./Data/p_train.csv', data)
    end = timer()
    print(f'Data Saved : {end-start:.3f} sec')

if __name__ == "__main__":
    main()