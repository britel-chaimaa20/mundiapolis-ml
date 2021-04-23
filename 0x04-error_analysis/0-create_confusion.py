#!/usr/bin/env python3
"""
Defines the function that creates a confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix
    """
    return np.matmul(labels.transpose(), logits)
