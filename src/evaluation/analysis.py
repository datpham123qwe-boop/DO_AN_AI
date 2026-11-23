"""Comparison guards used by the result-analysis notebook."""

import json
from pathlib import Path


def load_report(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        raise ValueError(f"Unsupported metrics schema: {path}")
    return report


def comparison_signature(report):
    protocol, dataset = report["protocol"], report["dataset"]
    return {
        "split": dataset["split"], "image_count": dataset["evaluated_images"],
        "filenames_sha256": dataset["filenames_sha256"],
        "class_names": report["class_names"], "mask_mode": protocol["mask_mode"],
        "ignore_index": protocol["ignore_index"], "normalization": protocol["normalization"],
        "inference_mode": protocol["mode"], "tile_size": protocol["tile_size"],
        "padding": protocol["padding"],
        "amp_enabled": protocol.get("amp_enabled", bool(protocol.get("amp_requested") and report.get("runtime", {}).get("gpu"))),
    }


def training_signature(report):
    config = report["config"]
    training = config.get("training", {})
    return {"seed": config.get("seed"), "crop_size": config.get("data", {}).get("crop_size"),
            "batch_size": training.get("batch_size"),
            "accumulation_steps": training.get("accumulation_steps", 1),
            "freeze_batchnorm": training.get("freeze_batchnorm", False),
            "weight_decay": training.get("weight_decay", 0.01)}


def assert_comparable(reports, equal_epoch_budget=False, equal_training_protocol=False):
    if not reports:
        raise ValueError("No evaluation reports")
    reference = None
    budgets = set()
    reference_training = None
    for name, report in reports.items():
        flags = report["quality_flags"]
        if flags["partial_evaluation"] or flags["smoke_checkpoint"] or not flags["official_split_size_matched"]:
            raise ValueError(f"{name}: preview/smoke/nonstandard split is not a final comparison")
        signature = comparison_signature(report)
        if signature["split"] != "val" or signature["image_count"] != 772:
            raise ValueError(f"{name}: requires the full 772-image validation split")
        if reference is not None:
            different = [key for key in signature if signature[key] != reference[key]]
            if different:
                raise ValueError(f"{name}: evaluation protocol differs: {different}")
        reference = signature
        budgets.add(report["config"].get("training", {}).get("epochs"))
        if equal_epoch_budget and report["checkpoint"].get("training_budget_completed") is False:
            raise ValueError(f"{name}: training log has not reached the configured epoch budget")
        if equal_training_protocol:
            current_training = training_signature(report)
            if reference_training is not None and current_training != reference_training:
                raise ValueError(f"{name}: training protocol differs: {current_training}")
            reference_training = current_training
    if equal_epoch_budget and (len(budgets) != 1 or None in budgets):
        raise ValueError(f"Training epoch budgets differ or are unknown: {budgets}")


def choose_ce_experiment(reports):
    ce = {name: report for name, report in reports.items()
          if report["config"].get("loss", {}).get("name") in ("ce", "cross_entropy")}
    architectures = {report["checkpoint"]["model_name"] for report in ce.values()}
    if architectures != {"deeplab", "segformer"} or len(ce) != 2:
        raise ValueError("Need exactly one full DeepLab CE and one full SegFormer CE report")
    assert_comparable(ce, equal_epoch_budget=True, equal_training_protocol=True)
    if any(report["checkpoint"].get("training_budget_completed") is not True for report in ce.values()):
        raise ValueError("Need training logs confirming completion of both CE epoch budgets before selecting E3")
    if any(report["segmentation"]["miou"] is None for report in ce.values()):
        raise ValueError("Cannot rank an undefined mIoU")
    # Declare the selection rule in advance: primary mIoU, tie-break Weed IoU.
    return sorted(ce, key=lambda name: (-ce[name]["segmentation"]["miou"],
                  -(ce[name]["segmentation"]["weed_iou"] or 0), name))[0]


def summary_row(name, report):
    seg, spray = report["segmentation"], report["spraying"]
    return {"experiment": name, "model": report["checkpoint"]["model_name"],
            "loss": report["config"].get("loss", {}).get("name"),
            "epoch_budget": report["config"].get("training", {}).get("epochs"),
            "best_epoch": report["checkpoint"].get("epoch"),
            "training_budget_completed": report["checkpoint"].get("training_budget_completed"),
            "miou": seg["miou"], "background_iou": seg["per_class"]["Background"]["iou"],
            "crop_iou": seg["crop_iou"], "weed_iou": seg["weed_iou"],
            "weed_recall": seg["weed_recall"], "weed_coverage": spray["weed_coverage"],
            "crop_wrongly_sprayed_rate": spray["crop_wrongly_sprayed_rate"],
            "spray_area_ratio": spray["spray_area_ratio"],
            "estimated_pixel_area_reduction": spray["estimated_pixel_area_reduction"]}
