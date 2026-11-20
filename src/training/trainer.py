import numpy as np
import  torch
from pathlib import Path
import csv
import random
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, model, device, criterion, optimizer, amp=True, output_dir="checkpoints"):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.amp = amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.best_val_loss = float("inf")

        self.log_path = self.output_dir / "training_log.csv"
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "train_loss", "val_loss"])

    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        self.best_val_loss = float(
            checkpoint.get("val_loss", self.best_val_loss)
        )
        return int(checkpoint.get("epoch", 0)) + 1

    def _resize_logits(self, logits, target_size):
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
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
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        torch.save(checkpoint, self.output_dir / "last.pt")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.output_dir / "best.pt")

    def fit(self, train_loader, val_loader, epochs, start_epoch=1):
        for epoch in range(start_epoch, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)

            with self.log_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([epoch, train_loss, val_loss])

            self.save_checkpoint(epoch, train_loss, val_loss)

            print(
                f"Epoch {epoch}/{epochs}"
                f" - train_loss = {train_loss:.4f}"
                f"- val_loss = {val_loss:.4f}"
            )