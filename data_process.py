import numpy as np


def standardize(data, mean, std):
    '''
    Standardizes data and subtracts mean divided by standard deviation.
    '''
    data = (data - mean)
    # because column 1 are all 1's the std deviation is 0. Fixing divide by 0 error
    data = np.divide(data, std, out=np.ones((data.shape[0], data.shape[1])), where=std != 0)

    return data


def impute(data, median):
    '''
    Imputes the -999 values with either the mean or median of the column.
    '''
    inds = np.where(np.isnan(data))
    data[inds] = np.take(median, inds[1])

    return data


def normalize(data, max_, min_):
    """
    Normalizes data given a maximum and minimum value.
    """
    data = (data - min_)
    diff = max_ - min_
    data = np.divide(data, diff, out=np.ones((data.shape[0], data.shape[1])), where=diff != 0)

    return data
