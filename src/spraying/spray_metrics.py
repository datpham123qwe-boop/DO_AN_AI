"""Measure pixel-based weed coverage, crop harm and spray area."""

from src.evaluation.confusion_matrix import confusion_matrix, validate_matrix
from src.evaluation.segmentation_metrics import safe_ratio


def spray_metrics_from_confusion_matrix(matrix):
    matrix = validate_matrix(matrix)
    gt_crop, gt_weed = int(matrix[1].sum()), int(matrix[2].sum())
    covered, harmed = int(matrix[2, 2]), int(matrix[1, 2])
    sprayed, total = int(matrix[:, 2].sum()), int(matrix.sum())
    coverage, area = safe_ratio(covered, gt_weed), safe_ratio(sprayed, total)
    return {
        "weed_coverage": coverage,
        "crop_wrongly_sprayed_rate": safe_ratio(harmed, gt_crop),
        "spray_area_ratio": area,
        "estimated_pixel_area_reduction": 1 - area if area is not None else None,
        "weed_miss_rate": 1 - coverage if coverage is not None else None,
        "gt_weed_pixels": gt_weed, "covered_weed_pixels": covered,
        "missed_weed_pixels": gt_weed - covered,
        "gt_crop_pixels": gt_crop, "wrongly_sprayed_crop_pixels": harmed,
        "sprayed_valid_pixels": sprayed, "valid_pixels": total,
    }


def spray_metrics(pred, target, ignore_index=255):
    return spray_metrics_from_confusion_matrix(confusion_matrix(pred, target, ignore_index=ignore_index))
