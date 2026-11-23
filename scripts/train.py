"""Train DeepLabV3+ or SegFormer with CE/WCE and resumable checkpoints."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.augmentation import MEAN, STD, train_transform, val_transform
from src.datasets.label_mapping import CLASS_NAMES
from src.datasets.phenobench import PhenoBenchDataset
from src.losses.losses import build_loss
from src.models.deeplab import modelDeepLabV3
from src.models.segformer import modelSegformer
from src.training.trainer import Trainer, set_seed


def repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path):
    return yaml.safe_load(repo_path(path).read_text(encoding="utf-8"))


def environment_info(device):
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]}
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "platform": platform.platform(), "packages": packages,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "device": str(device), "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name", required=True, choices=["deeplab", "segformer"])
    parser.add_argument("--resume")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--smoke", action="store_true", help="Separate pipeline check; not a full experiment")
    args = parser.parse_args()
    config = load_config(args.config)
    model_name = args.model_name
    config["model_name"] = model_name
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    training = config["training"]
    # Validate the supported batch-update and full-image validation protocol.
    if (training.get("accumulation_steps", 1) != 1
            or training.get("freeze_batchnorm", False)
            or training.get("validation_tile_size") is not None
            or training.get("loss_aggregation", "batch_mean") != "batch_mean"
            or training.get("val_batch_size", training["batch_size"]) != training["batch_size"]):
        raise ValueError("Training requires one optimizer step per batch, full-image validation and equal train/val batch sizes")
    resume = args.resume or training.get("resume")
    if args.smoke and resume:
        parser.error("--smoke cannot be combined with --resume")
    if args.smoke:
        training["epochs"] = 1
        training["output_dir"] = str(training["output_dir"]) + "_smoke"
    config["smoke_test"] = args.smoke
    config["data"]["root"] = str(repo_path(config["data"]["root"]))
    output = repo_path(training["output_dir"])
    training["output_dir"] = str(output)
    if not resume and any((output / name).exists() for name in ("best.pt", "last.pt", "config.yaml", "training_log.csv")):
        raise FileExistsError(f"Experiment exists: {output}. Resume or choose a new output directory.")
    if resume and repo_path(resume).parent != output:
        raise ValueError("Resume into the checkpoint's experiment directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; enable the Kaggle GPU")
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    set_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    train_dataset = PhenoBenchDataset(config["data"]["root"], "train",
        train_transform(config["data"].get("crop_size", 512), seed=config["seed"]),
        config["data"].get("mask_mode", "merge"))
    val_dataset = PhenoBenchDataset(config["data"]["root"], "val", val_transform(),
        config["data"].get("mask_mode", "merge"))
    # Smoke only limits samples; the actual training/validation methods are unchanged.
    train_samples = Subset(train_dataset, range(min(len(train_dataset), 2 * training["batch_size"]))) if args.smoke else train_dataset
    val_samples = Subset(val_dataset, range(min(len(val_dataset), training["batch_size"]))) if args.smoke else val_dataset
    train_loader = DataLoader(train_samples, batch_size=training["batch_size"], shuffle=True,
        num_workers=training.get("num_workers", 0), pin_memory=torch.cuda.is_available(), generator=generator)
    val_loader = DataLoader(val_samples, batch_size=training["batch_size"], shuffle=False,
        num_workers=training.get("num_workers", 0), pin_memory=torch.cuda.is_available())

    model = modelSegformer() if model_name == "segformer" else modelDeepLabV3()
    class_weight = config["loss"].get("class_weight")
    class_stats_path = config["loss"].get("class_stats")
    class_stats = None
    if class_stats_path and (class_weight is None or str(class_weight).lower() in ("auto", "null")):
        class_stats = json.loads(repo_path(class_stats_path).read_text(encoding="utf-8"))
        class_weight = class_stats["class_weights"][config["loss"].get("weight_method", "inverse_sqrt")]
        # Store the weights actually used so checkpoint/resume records the same loss.
        config["loss"]["class_weight"] = class_weight
    criterion = build_loss(config["loss"]["name"], class_weight, config["loss"].get("ignore_index", 255))
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training.get("weight_decay", 0.0001))
    environment = environment_info(device)
    metadata = {
        "model_name": model_name, "config": config, "environment": environment,
        "class_names": CLASS_NAMES, "normalization": {"mean": list(MEAN), "std": list(STD)},
        "class_weights": class_weight, "class_statistics": class_stats,
        "raw_to_project_labels": [0, 1, 2, 1, 2] if config["data"].get("mask_mode", "merge") == "merge" else [0, 1, 2, 255, 255],
        "selection_metric": "min_val_loss", "scheduler": None,
        "training_protocol": "original_batch_mean",
        "resumed_from": str(repo_path(resume)) if resume else None,
    }
    trainer = Trainer(model, device, criterion, optimizer, amp=training.get("amp", True),
                      output_dir=output, metadata=metadata, seed=config["seed"])
    start_epoch = trainer.load_checkpoint(repo_path(resume)) if resume else 1
    if start_epoch > training["epochs"]:
        raise ValueError(f"Already completed {start_epoch - 1} epochs; use best.pt for evaluation")
    suffix = "_resume" if resume else ""
    (output / f"config{suffix}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / f"environment{suffix}.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    lock_name = "requirements_resume.txt" if resume else "requirements-lock.txt"
    (output / lock_name).write_text("\n".join(f"{k}=={v}" for k, v in sorted(environment["packages"].items())) + "\n", encoding="utf-8")
    print(f"Model: {model_name} | device: {device} | train/val: {len(train_samples)}/{len(val_samples)}", flush=True)
    print(f"Output: {output} | epochs: {training['epochs']} | batch: {training['batch_size']}", flush=True)
    trainer.fit(train_loader, val_loader, training["epochs"], start_epoch)


if __name__ == "__main__":
    main()
