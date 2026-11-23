"""Integer pixel counts. Rows = ground truth, columns = prediction."""

import numpy as np


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def confusion_matrix(pred, target, num_classes=3, ignore_index=255):
    pred, target = as_numpy(pred), as_numpy(target)
    if pred.shape != target.shape or target.ndim not in (2, 3):
        raise ValueError("pred/target must have equal [H,W] or [B,H,W] shapes")
    if not np.issubdtype(pred.dtype, np.integer) or not np.issubdtype(target.dtype, np.integer):
        raise ValueError("pred/target must contain integer class IDs")
    valid = target != ignore_index
    truth, guess = target[valid], pred[valid]
    if np.any((truth < 0) | (truth >= num_classes)):
        raise ValueError("Target contains an invalid class ID")
    if np.any((guess < 0) | (guess >= num_classes)):
        raise ValueError("Prediction contains an invalid class ID")
    index = truth.astype(np.int64) * num_classes + guess.astype(np.int64)
    return np.bincount(index, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def validate_matrix(matrix, num_classes=3):
    matrix = as_numpy(matrix)
    if matrix.shape != (num_classes, num_classes):
        raise ValueError(f"Expected a {num_classes}x{num_classes} confusion matrix")
    if not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0):
        raise ValueError("Confusion matrix must contain nonnegative integer counts")
    return matrix.astype(np.int64, copy=False)


class ConfusionMatrix:
    def __init__(self, num_classes=3, ignore_index=255):
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        self.matrix += confusion_matrix(pred, target, self.num_classes, self.ignore_index)

    def compute(self):
        return self.matrix.copy()

    def reset(self):
        self.matrix.fill(0)
