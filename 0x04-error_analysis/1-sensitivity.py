#!/usr/bin/env python3
"""
Defines the function that calculates the sensitivity
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity 
    """
    classes = confusion.shape[0]
    sensitivity = []
    for row in range(classes):
        correct = 0
        total = 0
        for column in range(classes):
            if row == column:
                correct += confusion[row][column]
            total += confusion[row][column]
        sensitivity.append(correct / total)
    return np.asarray(sensitivity)
