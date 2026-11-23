"""Checkpoint loading, inference settings and reproducible report output."""

import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.datasets.augmentation import MEAN, STD
from src.models.factory import build_model

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_output(path):
    path = resolve_path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output is not empty: {path}. Choose a new output directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; check the environment or choose --device cpu")
    return torch.device("cuda" if name != "cpu" and torch.cuda.is_available() else "cpu")


def runtime_info(device):
    return {
        "python": sys.version, "platform": platform.platform(), "device": str(device),
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                     if dist.metadata["Name"]},
    }


def load_model_bundle(checkpoint_path, device, config_path=None, model_name=None,
                      allow_smoke=False):
    path = resolve_path(checkpoint_path)
    # Checkpoints must come from a trusted training run.
    payload = path.read_bytes()
    checkpoint_hash = hashlib.sha256(payload).hexdigest()
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    del payload
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a training checkpoint with model_state_dict")
    metadata = checkpoint.get("metadata", {})
    recorded = metadata.get("config", {})
    if config_path:
        config = yaml.safe_load(resolve_path(config_path).read_text(encoding="utf-8"))
    else:
        config = copy.deepcopy(recorded)
    if not config:
        raise ValueError("Legacy checkpoint has no config; pass --config")
    if recorded and config_path:
        # An external config must not relabel the provenance of a trained model.
        # Dataset relocation is handled explicitly by evaluate.py --data-root.
        for section in ("data", "training", "loss"):
            for key, value in config.get(section, {}).items():
                if key not in ("root", "output_dir") and key in recorded.get(section, {}):
                    if recorded[section][key] != value:
                        raise ValueError(f"External config differs from checkpoint: {section}.{key}")
        config = copy.deepcopy(recorded)
    if recorded and config.get("data", {}).get("mask_mode", "merge") != recorded.get("data", {}).get("mask_mode", "merge"):
        raise ValueError("Config label mapping differs from checkpoint")
    actual_name = model_name or metadata.get("model_name") or config.get("model_name")
    if actual_name not in ("deeplab", "segformer"):
        raise ValueError("Specify --model-name for a legacy checkpoint")
    for name in (metadata.get("model_name"), config.get("model_name")):
        if name and name != actual_name:
            raise ValueError("Requested model architecture differs from checkpoint/config")
    smoke = bool(recorded.get("smoke_test", config.get("smoke_test", False)))
    if smoke and not allow_smoke:
        raise ValueError("Smoke checkpoint is not a final model. Use --allow-smoke-checkpoint only for pipeline checks.")
    normalization = metadata.get("normalization", {"mean": list(MEAN), "std": list(STD)})
    if not np.allclose(normalization["mean"], MEAN) or not np.allclose(normalization["std"], STD):
        raise ValueError("Checkpoint normalization differs from current dataset preprocessing")
    model = build_model(actual_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    info = {
        "path": str(path), "sha256": checkpoint_hash, "model_name": actual_name,
        "epoch": checkpoint.get("epoch"), "val_loss": checkpoint.get("val_loss"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "selection_metric": checkpoint.get("selection_metric", "min_val_loss"),
        "is_smoke_checkpoint": smoke, "training_metadata": metadata,
    }
    completed = None
    for name in ("training_log.csv", "train_log.csv"):
        log_path = path.parent / name
        if log_path.is_file():
            with log_path.open(newline="", encoding="utf-8") as file:
                epochs = [int(row["epoch"]) for row in csv.DictReader(file)
                          if str(row.get("epoch", "")).isdigit()]
            completed = max(epochs) if epochs else None
            break
    budget = config.get("training", {}).get("epochs")
    info["logged_completed_epochs"] = completed
    info["training_budget_completed"] = completed >= budget if completed is not None and budget is not None else None
    return model, config, info


def inference_protocol(config, mode="auto", tile_size=512, amp=True):
    checkpoint_tile = config.get("training", {}).get("validation_tile_size")
    actual_tile = checkpoint_tile if mode == "auto" else (None if mode == "full" else tile_size)
    if actual_tile is not None and (not isinstance(actual_tile, int) or actual_tile < 1):
        raise ValueError("Tile size must be a positive integer")
    return {
        "mode": "tiled" if actual_tile else "full", "tile_size": actual_tile,
        "overlap": 0, "padding": "right/bottom normalized-zero; multiple of 32",
        "amp_requested": amp,
        "normalization": {"mean": list(MEAN), "std": list(STD)},
        "mask_mode": config.get("data", {}).get("mask_mode", "merge"),
        "ignore_index": config.get("loss", {}).get("ignore_index", 255),
        "matches_checkpoint_validation_geometry": actual_tile == checkpoint_tile,
    }
