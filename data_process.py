from timeit import default_timer as timer
from proj1_helpers import load_csv_data, save_csv_data
import numpy as np


# Preprocessing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def standardize(data, mean, std):
    '''
    Standardizes data and subtracts mean divided by standard deviation.
    '''
    data = (data - mean)
    #because column 1 are all 1's the std deviation is 0. Fixing divide by 0 error
    data = np.divide(data, std, out=np.ones((data.shape[0],data.shape[1])), where=std!=0)

    return data

def impute(data, median):
    '''
    Imputes the -999 values with either the mean or median of the column.
    '''
    inds = np.where(np.isnan(data))
    data[inds] = np.take(median, inds[1])

    return data

def normalize(data, max_ , min_):
    """
    Normalizes data given a maximum and minimum value.
    """
    data = (data - min_)
    diff = max_ - min_
    data = np.divide(data, diff, out=np.ones((data.shape[0],data.shape[1])), where=diff!=0)
    
    return data

def main():
    y, input_data_train, ids = load_csv_data("Data/train.csv", sub_sample = False)
    #yb_data_test, input_data_test, ids_test = load_csv_data("Data/test.csv")
    x = input_data_train

    x = np.delete(x,[5,6,12,21,26,27,28,29],axis = 1)

    data = np.c_[ids, y , x]
    start = timer()
    save_csv_data('./Data/p_train.csv', data)
    end = timer()
    print(f'Data Saved : {end-start:.3f} sec')

if __name__ == "__main__":
    main()