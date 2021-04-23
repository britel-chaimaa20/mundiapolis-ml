#!/usr/bin/env python3
"""Contains the precision function"""

import numpy as np


def precision(confusion):
    """
    :param confusion: is a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct
        labels and column indices represent the predicted labels
    :return: numpy.ndarray of shape (classes,)
        containing the precision of each class
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=0)
