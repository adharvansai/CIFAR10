import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

"""This script implements the functions for reading data.
"""
def loadpickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file,encoding='bytes')
    return data

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE

    meta_data = loadpickle(data_dir + '/batches.meta')
    train_data = np.empty((0, 3072))
    train_labels = []
    for i in range(5):
        data_load = loadpickle(data_dir + "/data_batch_" + str(i+1))
        train_labels = train_labels + data_load[b'labels']
        train_data = np.vstack((train_data,data_load[b'data']))

    x_train = train_data
    y_train = np.array(train_labels)

    data_load = loadpickle(data_dir + "/test_batch")
    x_test = data_load[b'data']
    y_test = np.array(data_load[b'labels'])


    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    x_test = []
    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    x_train_new, x_valid, y_train_new, y_valid = train_test_split(x_train,y_train,train_size=train_ratio)
    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

