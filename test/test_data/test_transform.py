"""Test cho src/datasets/transforms.py. Đặt tại: tests/test_transforms.py

Giả định file transforms.py có: train_transform(crop), val_transform(), MEAN, STD
(đúng như bản mình đã viết). Nếu bạn đặt tên khác thì sửa dòng import.
"""
import numpy as np
import pytest
import torch

from src.datasets.augmentation import MEAN, STD, train_transform, val_transform

CROP = 64


def _sample(h=128, w=128, labels=(0, 1, 2), seed=0):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = rng.choice(labels, size=(h, w)).astype(np.uint8)
    return image, mask


def test_train_output_shapes_and_dtypes():
    image, mask = _sample()
    out = train_transform(crop=CROP)(image=image, mask=mask)
    assert out["image"].shape == (3, CROP, CROP)
    assert out["image"].dtype == torch.float32
    assert out["mask"].shape == (CROP, CROP)


def test_train_keeps_mask_values_valid():
    """Augmentation không được tạo ra nhãn mới (không nội suy mask)."""
    for seed in range(20):
        image, mask = _sample(labels=(0, 2), seed=seed)
        out = train_transform(crop=CROP)(image=image, mask=mask)
        assert set(torch.unique(out["mask"]).tolist()) <= {0, 2}


def test_train_is_random():
    image, mask = _sample()
    t = train_transform(crop=CROP)
    outs = [t(image=image, mask=mask)["image"] for _ in range(10)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])


def test_val_keeps_full_size_and_mask():
    image, mask = _sample()
    out = val_transform()(image=image, mask=mask)
    assert out["image"].shape == (3, 128, 128)
    assert out["image"].dtype == torch.float32
    assert np.array_equal(out["mask"].numpy(), mask)


def test_val_is_deterministic():
    image, mask = _sample()
    t = val_transform()
    assert torch.equal(t(image=image, mask=mask)["image"], t(image=image, mask=mask)["image"])


def test_val_normalization_matches_imagenet_formula():
    value = 124
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    out = val_transform()(image=image, mask=mask)["image"].numpy()
    for c in range(3):
        expected = (value / 255.0 - MEAN[c]) / STD[c]
        assert out[c] == pytest.approx(expected, abs=1e-4)


def test_mean_std_are_imagenet():
    assert tuple(MEAN) == pytest.approx((0.485, 0.456, 0.406))
    assert tuple(STD) == pytest.approx((0.229, 0.224, 0.225))