"""Test cho src/datasets/phenobench.py. Đặt tại: tests/test_phenobench.py

Dùng dữ liệu giả từ conftest.py, không cần tải PhenoBench.
"""
import albumentations as A
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.phenobench import PhenoBenchDataset

TO_TENSOR = A.Compose([ToTensorV2()])


def test_len(fake_root, expected):
    assert len(PhenoBenchDataset(fake_root, "train")) == expected["n_train"]
    assert len(PhenoBenchDataset(fake_root, "val")) == expected["n_val"]


def test_files_are_sorted(fake_root, expected):
    ds = PhenoBenchDataset(fake_root, "train")
    assert ds.files == sorted(expected["train_files"])


def test_item_interface(fake_root):
    """Interface đã thống nhất với B và C."""
    item = PhenoBenchDataset(fake_root, "train", transform=TO_TENSOR)[0]
    assert set(item) == {"image", "mask", "filename"}
    assert item["image"].shape == (3, 128, 128)
    assert item["mask"].shape == (128, 128)
    assert item["mask"].dtype == torch.int64
    assert item["filename"].endswith(".png")
    assert set(torch.unique(item["mask"]).tolist()) <= {0, 1, 2}


def test_no_transform_still_returns_mask_tensor(fake_root):
    """Không truyền transform vẫn phải trả mask là tensor long (không phải numpy)."""
    item = PhenoBenchDataset(fake_root, "train")[0]
    assert isinstance(item["mask"], torch.Tensor)
    assert item["mask"].dtype == torch.int64
    assert item["mask"].shape == (128, 128)


def test_merge_mode_pixel_counts(fake_root, expected):
    ds = PhenoBenchDataset(fake_root, "train", transform=TO_TENSOR, mask_mode="merge")
    counts = np.bincount(ds[0]["mask"].numpy().ravel(), minlength=3)
    for class_id, n in expected["merged_per_image"].items():
        assert counts[class_id] == n


def test_ignore_mode_marks_partial_as_255(fake_root, expected):
    ds = PhenoBenchDataset(fake_root, "train", transform=TO_TENSOR, mask_mode="ignore")
    mask = ds[0]["mask"].numpy()
    assert set(np.unique(mask).tolist()) == {0, 1, 2, 255}
    assert (mask == 255).sum() == expected["ignored_per_image"]


def test_image_and_mask_stay_aligned_after_geometric_transform(fake_root):
    """Lỗi hay gặp nhất: ảnh và mask bị crop/lật/xoay lệch nhau.

    Kênh R của ảnh giả = nhãn * 80, nên sau transform phải còn khớp mask.
    """
    transform = A.Compose([
        A.RandomCrop(64, 64),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        ToTensorV2(),
    ])
    ds = PhenoBenchDataset(fake_root, "train", transform=transform)
    for _ in range(20):
        for i in range(len(ds)):
            item = ds[i]
            red = item["image"][0].numpy().astype(int)
            assert (red == item["mask"].numpy() * 80).all()


def test_dataloader_batches(fake_root):
    transform = A.Compose([A.RandomCrop(64, 64), ToTensorV2()])
    ds = PhenoBenchDataset(fake_root, "train", transform=transform)
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert batch["image"].shape == (2, 3, 64, 64)
    assert batch["mask"].shape == (2, 64, 64)
    assert batch["mask"].dtype == torch.int64
    assert len(batch["filename"]) == 2


def test_empty_or_missing_split_raises(fake_root):
    with pytest.raises(FileNotFoundError):
        PhenoBenchDataset(fake_root, "does_not_exist")


def test_missing_mask_raises_on_access(tmp_path):
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "semantics").mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
        tmp_path / "train" / "images" / "a.png"
    )
    ds = PhenoBenchDataset(tmp_path, "train")
    with pytest.raises(FileNotFoundError):
        ds[0]