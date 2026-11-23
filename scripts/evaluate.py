"""Evaluate one trusted checkpoint on train/val and export metrics + error figures."""

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from src.datasets.augmentation import val_transform
from src.datasets.phenobench import PhenoBenchDataset
from src.evaluation.io import (choose_device, inference_protocol, load_model_bundle,
                               prepare_output, resolve_path)
from src.evaluation.pipeline import evaluate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Required for legacy checkpoints without embedded config")
    parser.add_argument("--model-name", "--model_name", choices=["deeplab", "segformer"])
    parser.add_argument("--data-root")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", choices=["E1", "E2", "E3"])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--inference-mode", choices=["auto", "full", "tiled"], default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit", type=int, help="Pipeline preview only; report is marked partial")
    parser.add_argument("--max-figures", type=int, default=20)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_figures < 0 or (args.limit is not None and args.limit < 1):
        parser.error("Invalid batch/worker/limit/figure count")
    device = choose_device(args.device)
    model, config, info = load_model_bundle(args.checkpoint, device, args.config,
                                           args.model_name, args.allow_smoke_checkpoint)
    root = resolve_path(args.data_root or config["data"]["root"])
    if not (root / args.split / "semantics").is_dir():
        raise FileNotFoundError(f"No GT at {root}; pass --data-root containing train/val")
    dataset = PhenoBenchDataset(root, args.split, val_transform(), config["data"].get("mask_mode", "merge"))
    protocol = inference_protocol(config, args.inference_mode, args.tile_size, not args.no_amp)
    loss = config.get("loss", {}).get("name", "ce")
    experiment = args.experiment_id or config.get("experiment_id") or (
        "E3" if loss in ("wce", "weighted_cross_entropy") else "E1" if info["model_name"] == "deeplab" else "E2")
    output = prepare_output(args.output_dir)
    report = evaluate_dataset(model, dataset, output, device, protocol, info, config, experiment,
                              args.batch_size, args.num_workers, args.limit, args.max_figures)
    source_manifest = REPO / "source_manifest.json"
    if source_manifest.is_file():
        shutil.copy2(source_manifest, output / source_manifest.name)
    print("mIoU:", report["segmentation"]["miou"])
    print("Weed Recall / Coverage:", report["segmentation"]["weed_recall"])
    print("Crop Wrongly Sprayed Rate:", report["spraying"]["crop_wrongly_sprayed_rate"])
    print("Quality flags:", report["quality_flags"])
    print("Saved:", output / "metrics.json")


if __name__ == "__main__":
    main()
