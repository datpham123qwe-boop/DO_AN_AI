"""Evaluate segmentation and spray masks using dataset-wide pixel counts."""

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from src.datasets.label_mapping import CLASS_NAMES, map_labels
from src.evaluation.confusion_matrix import ConfusionMatrix, confusion_matrix
from src.evaluation.error_maps import ERROR_NAMES, colorize_error_map, make_error_map
from src.evaluation.figures import save_confusion_figure, save_qualitative_figure
from src.evaluation.inference import predict_logits
from src.evaluation.io import runtime_info, write_json
from src.evaluation.segmentation_metrics import metrics_from_confusion_matrix
from src.spraying.spray_mask import make_spray_mask
from src.spraying.spray_metrics import spray_metrics_from_confusion_matrix


def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else [])
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_failure_cases(rows, maximum=20):
    if maximum < 0:
        raise ValueError("maximum must be nonnegative")
    errors = [row for row in rows if row["error_pixels"] > 0]
    rules = [
        ("crop_safety", "crop_wrongly_sprayed_rate", "wrongly_sprayed_crop_pixels"),
        ("weed_coverage", "weed_miss_rate", "missed_weed_pixels"),
        ("overall_error", "error_rate", "error_pixels"),
    ]
    queues = []
    for reason, ratio, count in rules:
        ranked = sorted([r for r in errors if r[ratio] is not None and r[count] > 0],
                        key=lambda r: (-r[ratio], -r[count], r["filename"]))
        queues.append((reason, iter(ranked)))
    chosen, names = [], set()
    while len(chosen) < maximum:
        added = False
        for reason, queue in queues:
            candidate = next((row for row in queue if row["filename"] not in names), None)
            if candidate is not None:
                chosen.append({**candidate, "selection_reason": reason, "manual_category": "",
                               "manual_notes": ""})
                names.add(candidate["filename"])
                added = True
                if len(chosen) == maximum:
                    break
        if not added:
            break
    return chosen


def evaluate_dataset(model, dataset, output, device, protocol, checkpoint_info,
                     config, experiment_id, batch_size=1, num_workers=0,
                     limit=None, max_figures=20):
    if batch_size < 1 or num_workers < 0 or max_figures < 0 or (limit is not None and limit < 1):
        raise ValueError("Invalid batch/worker/limit/figure count")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    amount = min(limit, len(dataset)) if limit is not None else len(dataset)
    subset = Subset(dataset, range(amount))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    for name in ("predictions", "spray_masks", "error_maps", "error_ids", "figures"):
        (output / name).mkdir(exist_ok=True)
    model.eval()
    accumulator = ConfusionMatrix(ignore_index=protocol["ignore_index"])
    rows = []
    for batch in loader:
        images = batch["image"].to(device)
        logits = predict_logits(model, images, protocol["tile_size"], protocol["amp_requested"])
        predictions = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        targets = batch["mask"].numpy()
        for name, pred, target in zip(batch["filename"], predictions, targets):
            matrix = confusion_matrix(pred, target, ignore_index=protocol["ignore_index"])
            accumulator.matrix += matrix
            segmentation = metrics_from_confusion_matrix(matrix)
            spraying = spray_metrics_from_confusion_matrix(matrix)
            error_pixels = int(matrix.sum() - matrix.trace())
            row = {
                "filename": name, "miou": segmentation["miou"],
                "background_iou": segmentation["per_class"]["Background"]["iou"],
                "crop_iou": segmentation["crop_iou"], "weed_iou": segmentation["weed_iou"],
                "weed_recall": segmentation["weed_recall"], **spraying,
                "error_pixels": error_pixels,
                "error_rate": error_pixels / int(matrix.sum()) if matrix.sum() else None,
            }
            rows.append(row)
            Image.fromarray(pred).save(output / "predictions" / name)
            spray = make_spray_mask(pred, target, protocol["ignore_index"])
            Image.fromarray(spray.astype(np.uint8) * 255).save(output / "spray_masks" / name)
            errors = make_error_map(pred, target, protocol["ignore_index"])
            Image.fromarray(errors).save(output / "error_ids" / name)
            Image.fromarray(colorize_error_map(errors)).save(output / "error_maps" / name)
        if len(rows) == len(predictions) or len(rows) % 100 < len(predictions) or len(rows) == amount:
            print(f"Evaluated {len(rows)}/{amount} images", flush=True)

    matrix = accumulator.compute()
    segmentation = metrics_from_confusion_matrix(matrix)
    spraying = spray_metrics_from_confusion_matrix(matrix)
    np.save(output / "confusion_matrix.npy", matrix)
    write_csv(output / "confusion_matrix.csv",
              [{"ground_truth": name, **{label: int(matrix[i, j]) for j, label in enumerate(CLASS_NAMES)}}
               for i, name in enumerate(CLASS_NAMES)])
    write_csv(output / "per_image_metrics.csv", rows)
    selected = select_failure_cases(rows, max_figures)
    for row in selected:
        name = row["filename"]
        with Image.open(output / "predictions" / name) as file:
            pred = np.array(file)
        with Image.open(dataset.sem_dir / name) as file:
            target = map_labels(np.array(file), dataset.mask_mode)
        with Image.open(dataset.img_dir / name) as file:
            rgb = np.array(file.convert("RGB"))
        figure = "figures/" + Path(name).stem + ".png"
        save_qualitative_figure(output / figure, rgb, pred, target,
                                title=f"{name} | selected for {row['selection_reason']}")
        row["figure"] = figure
    fields = list(selected[0]) if selected else ["filename", "selection_reason", "manual_category", "manual_notes", "figure"]
    write_csv(output / "selected_cases.csv", selected, fields)
    save_confusion_figure(output / "figures/confusion_matrix.png", matrix)
    processed_names = [row["filename"] for row in rows]
    expected = {"train": 1407, "val": 772}.get(dataset.split)
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id, "checkpoint": checkpoint_info,
        "dataset": {
            "root": str(dataset.root.resolve()), "split": dataset.split,
            "available_images": len(dataset), "evaluated_images": len(rows),
            "expected_official_image_count": expected,
            "filenames_sha256": hashlib.sha256("\n".join(processed_names).encode()).hexdigest(),
        },
        "quality_flags": {
            "partial_evaluation": len(rows) != len(dataset),
            "official_split_size_matched": len(dataset) == expected,
            "smoke_checkpoint": bool(checkpoint_info.get("is_smoke_checkpoint", False)),
        },
        "protocol": {**protocol, "batch_size": batch_size,
                     "amp_enabled": bool(protocol["amp_requested"] and device.type == "cuda"),
                     "confusion_matrix_orientation": "rows=ground_truth, columns=prediction",
                     "aggregation": "sum pixel counts over entire evaluated split",
                     "undefined_metric": "null; mIoU requires all 3 class IoUs defined",
                     "ratio_units": "0..1; multiply by 100 for percentage"},
        "config": config, "class_names": CLASS_NAMES,
        "confusion_matrix": matrix.tolist(), "segmentation": segmentation, "spraying": spraying,
        "error_map_legend": {str(i): name for i, name in enumerate(ERROR_NAMES)} | {"255": "Ignored"},
        "selected_cases": selected,
        "runtime": runtime_info(device),
        "notes": ["Weed Coverage equals Weed Recall for spray = predicted Weed.",
                  "Estimated Pixel-Area Reduction is not measured herbicide saving.",
                  "Selected cases are ranked pixel errors; biological/lighting categories require manual review."],
    }
    write_json(output / "metrics.json", report)
    return report
