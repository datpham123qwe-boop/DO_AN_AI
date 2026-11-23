"""Training and validation; best means lowest validation CE/WCE loss."""

import csv
import math
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, model, device, criterion, optimizer, amp=True,
                 output_dir="checkpoints", metadata=None, seed=42):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}
        self.seed = seed
        self.amp = amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.best_val_loss = float("inf")
        self.log_path = self.output_dir / "training_log.csv"
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(["epoch", "train_loss", "val_loss"])

    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        previous = checkpoint.get("metadata", {})
        if previous.get("model_name") and self.metadata.get("model_name") != previous["model_name"]:
            raise ValueError("Checkpoint model_name does not match this experiment")
        if previous.get("config") and self.metadata.get("config"):
            old_config, new_config = previous["config"], self.metadata["config"]
            if old_config.get("smoke_test") != new_config.get("smoke_test"):
                raise ValueError("Do not resume a smoke checkpoint as a full experiment")
            for section in ("data", "loss"):
                if old_config.get(section) != new_config.get(section):
                    raise ValueError(f"Resume requires the same {section} config")
            if old_config.get("seed") != new_config.get("seed"):
                raise ValueError("Resume requires the same seed")
            for key, default in (("batch_size", None), ("learning_rate", None),
                                 ("weight_decay", 0.0001), ("num_workers", 0), ("amp", True)):
                if old_config.get("training", {}).get(key, default) != new_config.get("training", {}).get(key, default):
                    raise ValueError(f"Resume requires the same training.{key}")
            for saved in (old_config, new_config):
                training = saved.get("training", {})
                if (training.get("accumulation_steps", 1) != 1
                        or training.get("freeze_batchnorm", False)
                        or training.get("validation_tile_size") is not None
                        or training.get("loss_aggregation", "batch_mean") != "batch_mean"
                        or training.get("val_batch_size", training.get("batch_size")) != training.get("batch_size")):
                    raise ValueError("Checkpoint uses a different training protocol; cannot resume with this trainer")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        candidates = [checkpoint.get("best_val_loss"), checkpoint.get("val_loss")]
        # Recover missing best_val_loss from files beside the checkpoint.
        if "best_val_loss" not in checkpoint:
            best_path = checkpoint_path.parent / "best.pt"
            if best_path.is_file() and best_path.resolve() != checkpoint_path.resolve():
                best = torch.load(best_path, map_location="cpu", weights_only=False)
                candidates.extend([best.get("best_val_loss"), best.get("val_loss")])
            log_path = checkpoint_path.parent / "training_log.csv"
            if log_path.is_file():
                with log_path.open(newline="", encoding="utf-8") as file:
                    for row in csv.DictReader(file):
                        candidates.append(float(row["val_loss"]))
            warnings.warn("Legacy checkpoint: recovered best loss from available best.pt/log; "
                          "without these, the original historical best cannot be reconstructed.")
        finite = [float(v) for v in candidates if v is not None and math.isfinite(float(v))]
        if not finite:
            raise ValueError("Checkpoint has no finite validation loss")
        self.best_val_loss = min(finite)
        return int(checkpoint.get("epoch", 0)) + 1

    @staticmethod
    def _resize_logits(logits, target_size):
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return logits

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            images = batch["image"].to(self.device)
            mask = batch["mask"].to(self.device).long()

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.amp
            ):
                logits = self.model(images)
                logits = self._resize_logits(logits, mask.shape[-2:])
                loss = self.criterion(logits, mask)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batches += 1
            total_loss += loss.item()

        if batches == 0:
            raise ValueError("Train DataLoader is empty")
        return total_loss / batches

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                mask = batch["mask"].to(self.device).long()

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.amp
                ):
                    logits = self.model(images)
                    logits = self._resize_logits(logits, mask.shape[-2:])
                    loss = self.criterion(logits, mask)

                total_loss += loss.item()
                batches += 1

        if batches == 0:
            raise ValueError("Validation DataLoader is empty")
        return total_loss / batches

    def save_checkpoint(self, epoch, train_loss, val_loss):
        if not math.isfinite(val_loss):
            raise ValueError("Validation loss is not finite")
        improved = val_loss < self.best_val_loss
        self.best_val_loss = min(self.best_val_loss, val_loss)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "selection_metric": "min_val_loss",
            "metadata": self.metadata,
        }
        self._atomic_save(checkpoint, self.output_dir / "last.pt")
        if improved:
            self._atomic_save(checkpoint, self.output_dir / "best.pt")

    @staticmethod
    def _atomic_save(checkpoint, path):
        temporary = path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    def fit(self, train_loader, val_loader, epochs, start_epoch=1):
        if start_epoch > epochs:
            raise ValueError(f"Checkpoint already reached epoch {start_epoch - 1}; increase training.epochs in the config")
        for epoch in range(start_epoch, epochs + 1):
            # Repeatable at epoch boundaries under the same environment/protocol.
            set_seed(self.seed + epoch)
            dataset = getattr(train_loader.dataset, "dataset", train_loader.dataset)
            transform = getattr(dataset, "transform", None)
            if hasattr(transform, "set_random_seed"):
                transform.set_random_seed(self.seed + epoch)
            if train_loader.generator is not None:
                train_loader.generator.manual_seed(self.seed + epoch)
            print(f"Epoch {epoch}/{epochs}", flush=True)
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)
            with self.log_path.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([epoch, train_loss, val_loss])
            self.save_checkpoint(epoch, train_loss, val_loss)
            print(f"Epoch {epoch}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, "
                  f"best_val_loss={self.best_val_loss:.5f}", flush=True)
