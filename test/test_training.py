from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.augmentation import train_transform
from src.training.trainer import Trainer, set_seed


def trainer_at(path, model=None, **kwargs):
    model = model if model is not None else nn.Conv2d(3, 3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return Trainer(model, "cpu", nn.CrossEntropyLoss(), optimizer,
                   amp=False, output_dir=path, **kwargs)


def test_resume_last_preserves_historical_best(tmp_path):
    trainer = trainer_at(tmp_path)
    trainer.save_checkpoint(1, 0.2, 0.1)
    trainer.save_checkpoint(2, 0.6, 0.5)
    resumed = trainer_at(tmp_path)
    assert resumed.load_checkpoint(tmp_path / "last.pt") == 3
    assert resumed.best_val_loss == 0.1
    resumed.save_checkpoint(3, 0.5, 0.4)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    last = torch.load(tmp_path / "last.pt", weights_only=False)
    assert best["epoch"] == 1
    assert last["best_val_loss"] == 0.1
    resumed.save_checkpoint(4, 0.1, 0.05)
    assert torch.load(tmp_path / "best.pt", weights_only=False)["epoch"] == 4


@pytest.mark.parametrize("source", ["best", "log"])
def test_legacy_checkpoint_recovers_minimum(tmp_path, source):
    trainer = trainer_at(tmp_path)
    trainer.save_checkpoint(1, 0.2, 0.1)
    trainer.save_checkpoint(2, 0.6, 0.5)
    legacy = torch.load(tmp_path / "last.pt", weights_only=False)
    del legacy["best_val_loss"]
    torch.save(legacy, tmp_path / "last.pt")
    if source == "log":
        (tmp_path / "best.pt").unlink()
        (tmp_path / "training_log.csv").write_text(
            "epoch,train_loss,val_loss\n1,0.2,0.1\n2,0.6,0.5\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="Legacy"):
        trainer.load_checkpoint(tmp_path / "last.pt")
    assert trainer.best_val_loss == 0.1


def test_metadata_is_saved_and_wrong_model_rejected(tmp_path):
    trainer = trainer_at(tmp_path, metadata={"model_name": "deeplab"})
    trainer.save_checkpoint(1, 0.2, 0.1)
    checkpoint = torch.load(tmp_path / "last.pt", weights_only=False)
    assert checkpoint["metadata"]["model_name"] == "deeplab"
    assert checkpoint["selection_metric"] == "min_val_loss"
    other = trainer_at(tmp_path, metadata={"model_name": "segformer"})
    with pytest.raises(ValueError, match="model_name"):
        other.load_checkpoint(tmp_path / "last.pt")


def test_seed_repeats_augmentation_sequence():
    image = np.random.default_rng(1).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    first, second = train_transform(64, seed=42), train_transform(64, seed=42)
    for _ in range(4):
        assert torch.equal(first(image=image, mask=mask)["image"],
                           second(image=image, mask=mask)["image"])


def samples(n=5):
    gen = torch.Generator().manual_seed(20)
    return [{"image": torch.rand(3, 8, 10, generator=gen),
             "mask": torch.randint(3, (8, 10), generator=gen)} for _ in range(n)]


def test_epoch_boundary_resume_matches_uninterrupted_training(tmp_path):
    dataset = samples(4)
    def loaders():
        return (DataLoader(dataset, batch_size=1, shuffle=True,
                           generator=torch.Generator().manual_seed(42)),
                DataLoader(dataset, batch_size=2))
    set_seed(7)
    model = nn.Sequential(nn.Conv2d(3, 3, 1), nn.Dropout2d(0.2))
    full = trainer_at(tmp_path / "full", model=deepcopy(model))
    full.fit(*loaders(), epochs=2)
    part = trainer_at(tmp_path / "part", model=deepcopy(model))
    part.fit(*loaders(), epochs=1)
    resumed = trainer_at(tmp_path / "part", model=deepcopy(model))
    start = resumed.load_checkpoint(tmp_path / "part" / "last.pt")
    resumed.fit(*loaders(), epochs=2, start_epoch=start)
    for actual, expected in zip(resumed.model.parameters(), full.model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("weighted", [False, True])
def test_original_batch_mean_validation_matches_original_trainer(tmp_path, weighted):
    trainer = trainer_at(tmp_path)
    if weighted:
        trainer.criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.5, 2.3]))
    loader = DataLoader(samples(5), batch_size=2)
    trainer.model.eval()
    with torch.no_grad():
        expected = sum(trainer.criterion(trainer.model(batch["image"]), batch["mask"]).item()
                       for batch in loader) / len(loader)
    assert trainer.validate(loader) == pytest.approx(expected, abs=1e-7)


def test_original_batch_mean_training_logs_each_batch_equally(tmp_path):
    trainer = trainer_at(tmp_path)
    for group in trainer.optimizer.param_groups:
        group["lr"] = 0
    loader = DataLoader(samples(5), batch_size=2)
    with torch.no_grad():
        expected = sum(trainer.criterion(trainer.model(batch["image"]), batch["mask"]).item()
                       for batch in loader) / len(loader)
    assert trainer.train_one_epoch(loader) == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("option,value", [
    ("accumulation_steps", 4), ("freeze_batchnorm", True),
    ("validation_tile_size", 512), ("loss_aggregation", "pixel_mean"),
])
def test_resume_refuses_a_different_training_protocol(tmp_path, option, value):
    config = {"seed": 42, "data": {}, "loss": {"name": "ce"},
              "training": {"batch_size": 4, "learning_rate": 0.0001}}
    previous = deepcopy(config)
    previous["training"][option] = value
    trainer_at(tmp_path, metadata={"config": previous}).save_checkpoint(1, 0.2, 0.1)
    resumed = trainer_at(tmp_path, metadata={"config": config})
    with pytest.raises(ValueError, match="different training protocol"):
        resumed.load_checkpoint(tmp_path / "last.pt")
