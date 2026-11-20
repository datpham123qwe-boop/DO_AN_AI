"""
chay python scripts/train --config [1] --model_name [2] --resume [3]
[1]: duong dan file config yaml
[2]: deeplab hoac segformer
[3]: last.pt hoac best.pt hoac None. Mac dinh None
"""


import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.augmentation import (
    train_transform,
    val_transform,
)
from src.datasets.phenobench import PhenoBenchDataset
from src.losses.losses import build_loss
from src.training.trainer import Trainer, set_seed
from src.models.deeplab import modelDeepLabV3
from src.models.segformer import modelSegformer


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required = True
    )
    parser.add_argument(
        "--model_name",
        required = True
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Đường dẫn checkpoint last.pt hoặc best.pt",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    set_seed(config["seed"])
    train_dataset = PhenoBenchDataset(root = config["data"]["root"], 
                                    split="train", 
                                    transform= train_transform(
                                        crop=config["data"].get("crop_size",512)
                                    ),
                                    mask_mode = config["data"].get("mask_mode","merge"),
                                    )

    val_dataset = PhenoBenchDataset(root=config["data"]["root"],
                                    split="val",
                                    transform= val_transform(),
                                    mask_mode=config["data"].get("mask_mode","merge"),)

    train_loader = DataLoader(
        train_dataset, 
        batch_size = config["training"]["batch_size"],
        shuffle= True,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size = config["training"]["batch_size"],
        shuffle= False,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    model_name = args.model_name.lower()
    model = None
    if model_name == "segformer":
        model = modelSegformer()
    elif model_name == "deeplab":
        model = modelDeepLabV3()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "cpu"
    )

    class_weight = config["loss"].get("class_weight")
    class_stats_path = config["loss"].get("class_stats")

    if class_stats_path and (
        class_weight is None or str(class_weight).lower() in ("auto", "null")
    ):
        stats_path = Path(class_stats_path)
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy class stats: {stats_path}. "
                "Chạy scripts/compute_class_stats.py trước."
            )

        with stats_path.open("r", encoding="utf-8") as f:
            class_stats = json.load(f)

        weight_method = config["loss"].get("weight_method", "inverse_sqrt")
        if weight_method not in class_stats["class_weights"]:
            raise ValueError(f"Không có phương thức weight: {weight_method}")
        class_weight = class_stats["class_weights"][weight_method]

    criterion = build_loss(
        name=config["loss"]["name"],
        class_weight=class_weight,
        ignore_index=config["loss"].get("ignore_index", 255),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay",0.0001)
    )
    trainer = Trainer(
                model=model,
                device=device,
                criterion= criterion,
                optimizer=optimizer,
                amp = config["training"].get("amp", True),
                output_dir = config["training"]["output_dir"], 
                )

    resume_path = args.resume or config["training"].get("resume")
    start_epoch = 1
    if resume_path:
        start_epoch = trainer.load_checkpoint(resume_path)
        print(f"Resume training từ epoch {start_epoch}: {resume_path}")

    trainer.fit(train_loader=train_loader,
                val_loader = val_loader, 
                epochs = config["training"]["epochs"],
                start_epoch=start_epoch,
                )

    print()

if __name__ == "__main__":
    main()