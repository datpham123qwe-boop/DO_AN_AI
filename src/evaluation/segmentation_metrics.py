"""Metrics calculated from dataset-wide counts, never averaged per image."""

from src.datasets.label_mapping import CLASS_NAMES
from src.evaluation.confusion_matrix import confusion_matrix, validate_matrix


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def metrics_from_confusion_matrix(matrix):
    matrix = validate_matrix(matrix)
    ground_truth, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    per_class = {}
    for class_id, name in enumerate(CLASS_NAMES):
        tp = int(matrix[class_id, class_id])
        fp, fn = int(predicted[class_id]) - tp, int(ground_truth[class_id]) - tp
        per_class[name] = {
            "iou": safe_ratio(tp, tp + fp + fn),
            "precision": safe_ratio(tp, tp + fp),
            "recall": safe_ratio(tp, tp + fn),
            "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "gt_pixels": int(ground_truth[class_id]),
            "predicted_pixels": int(predicted[class_id]),
        }
    ious = [per_class[name]["iou"] for name in CLASS_NAMES]
    # mIoU requires all three class IoUs; an undefined class yields null.
    miou = sum(ious) / 3 if all(value is not None for value in ious) else None
    return {
        "per_class": per_class, "miou": miou,
        "crop_iou": per_class["Crop"]["iou"],
        "weed_iou": per_class["Weed"]["iou"],
        "weed_recall": per_class["Weed"]["recall"],
        "pixel_accuracy": safe_ratio(int(matrix.trace()), int(matrix.sum())),
        "valid_pixels": int(matrix.sum()),
    }


def segmentation_metrics(pred, target, ignore_index=255):
    return metrics_from_confusion_matrix(confusion_matrix(pred, target, ignore_index=ignore_index))
