#import os
#import sys
import numpy as np
#import logging
#import time


def train_test_split_indices(total_number, test_fraction):
    """
    This function produces a random list of indices drawn uniformly from the list 'range(total_number)'.
    It returns about 'test_fraction*total_number' indices, which are suitable for splitting a set into
    a training and a test set.
    """
    return set(list(np.random.choice(total_number, size = [int(test_fraction * total_number)], replace = False)))



def choose_samples(data_x, data_y, number):
    """
    This funtion chooses 'number' samples from data_x and data_y, and returns a pair of matrices in the form
    of a dictionary {'input':<...>, 'output':<...>}. This is meant to be used to choose a validation set from
    a subset of the data which might have been set aside for validation purposes.
    """
    test_indices = set(list(np.random.choice(len(data_x), size = [number], replace = False)))
    test_in   = []
    test_out  = []
    for index, point in enumerate(data_x):
        if index in test_indices:
            test_in.append(data_x[index])
            test_out.append(data_y[index])
    return {'input': np.array(test_in), 'output': np.array(test_out)}
