#!/usr/bin/env python3
"""Contains the sensitivity function"""

import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix:
    :param confusion: confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    :return: numpy.ndarray of shape (classes,)
        containing the sensitivity of each class
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=1)
