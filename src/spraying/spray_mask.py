"""Create a binary spray mask from pixels predicted as Weed."""

import numpy as np

from src.evaluation.confusion_matrix import as_numpy, confusion_matrix


def make_spray_mask(pred, target=None, ignore_index=255):
    pred = as_numpy(pred)
    if pred.ndim != 2 or not np.issubdtype(pred.dtype, np.integer):
        raise ValueError("Prediction must be a [H,W] integer class mask")
    if target is None:
        if np.any((pred < 0) | (pred > 2)):
            raise ValueError("Prediction IDs must be 0/1/2")
        return pred == 2
    target = as_numpy(target)
    confusion_matrix(pred, target, ignore_index=ignore_index)  # Validate aligned labels.
    return (pred == 2) & (target != ignore_index)
