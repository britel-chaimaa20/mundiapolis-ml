#!/usr/bin/env python3
"""Contains the specificity function"""

import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a confusion matrix:
    :param confusion: is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct
    labels and column indices represent the predicted labels
    :return: numpy.ndarray of shape (classes,)
    containing the specificity of each class
    """

    total = np.sum(confusion)
    true_positive = np.diagonal(confusion)

    actual = np.sum(confusion, axis=1)
    predicted = np.sum(confusion, axis=0)

    false_positive = predicted - true_positive
    actual_negative = total - actual
    true_negative = actual_negative - false_positive

    return true_negative / actual_negative
