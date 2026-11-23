"""Error IDs: 0 correct, 1 other, 2 missed weed, 3 sprayed crop, 255 ignored."""

import numpy as np

from src.evaluation.confusion_matrix import as_numpy, confusion_matrix

ERROR_PALETTE = np.array([[40, 40, 40], [255, 210, 0], [0, 120, 255], [255, 0, 0]], dtype=np.uint8)
ERROR_NAMES = ["Correct", "Other error", "Missed weed", "Sprayed crop"]


def make_error_map(pred, target, ignore_index=255):
    pred, target = as_numpy(pred), as_numpy(target)
    confusion_matrix(pred, target, ignore_index=ignore_index)
    if pred.ndim != 2:
        raise ValueError("Error map needs one [H,W] prediction")
    valid = target != ignore_index
    errors = np.zeros(pred.shape, dtype=np.uint8)
    errors[valid & (pred != target)] = 1
    errors[valid & (target == 2) & (pred != 2)] = 2
    errors[valid & (target == 1) & (pred == 2)] = 3
    errors[~valid] = 255
    return errors


def colorize_error_map(errors):
    errors = as_numpy(errors)
    if errors.ndim != 2 or not np.isin(errors, [0, 1, 2, 3, 255]).all():
        raise ValueError("Unexpected error map IDs")
    rgb = np.full((*errors.shape, 3), 160, dtype=np.uint8)
    valid = errors != 255
    rgb[valid] = ERROR_PALETTE[errors[valid]]
    return rgb
