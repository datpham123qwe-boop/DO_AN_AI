"""Test cho src/visualization/visualize.py. Đặt tại: tests/test_visualize.py"""
import albumentations as A
import matplotlib

matplotlib.use("Agg")  # không mở cửa sổ khi chạy test

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from albumentations.pytorch import ToTensorV2  # noqa: E402

from src.datasets.phenobench import PhenoBenchDataset  # noqa: E402
from src.visualization.visualize import (  # noqa: E402
    PALETTE,
    colorize_mask,
    denormalize,
    overlay,
    plot_samples,
)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def _random_image(h=16, w=16, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_colorize_mask_shape_and_dtype():
    mask = np.random.default_rng(0).integers(0, 3, (10, 12))
    out = colorize_mask(mask)
    assert out.shape == (10, 12, 3)
    assert out.dtype == np.uint8


def test_colorize_mask_uses_palette_per_class():
    for class_id in range(3):
        out = colorize_mask(np.full((4, 4), class_id))
        assert (out == PALETTE[class_id]).all()


def test_palette_has_three_distinct_colors():
    assert PALETTE.shape == (3, 3)
    assert len({tuple(c) for c in PALETTE.tolist()}) == 3


def test_overlay_alpha_zero_returns_image():
    image = _random_image()
    mask = np.random.default_rng(1).integers(0, 3, (16, 16))
    assert np.array_equal(overlay(image, mask, alpha=0.0), image)


def test_overlay_alpha_one_returns_mask_colors():
    image = _random_image()
    mask = np.random.default_rng(1).integers(0, 3, (16, 16))
    assert np.array_equal(overlay(image, mask, alpha=1.0), colorize_mask(mask))


def test_overlay_half_is_between_image_and_color():
    image = _random_image()
    mask = np.ones((16, 16), dtype=int)
    out = overlay(image, mask, alpha=0.5).astype(float)
    expected = (image.astype(float) + colorize_mask(mask).astype(float)) / 2
    assert np.abs(out - expected).max() <= 1.0
    assert out.shape == image.shape


def test_denormalize_inverts_imagenet_normalization():
    image = _random_image()
    normalized = (image / 255.0 - MEAN) / STD
    tensor = torch.tensor(normalized, dtype=torch.float32).permute(2, 0, 1)
    out = denormalize(tensor)
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.uint8
    assert np.abs(out.astype(int) - image.astype(int)).max() <= 1


def test_denormalize_clips_to_valid_range():
    out = denormalize(torch.full((3, 4, 4), 100.0))
    assert out.min() >= 0 and out.max() <= 255


def test_plot_samples_draws_grid(fake_root, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    transform = A.Compose([A.Normalize(), ToTensorV2()])
    ds = PhenoBenchDataset(fake_root, "train", transform=transform)
    plt.close("all")
    plot_samples(ds, n=2)
    fig = plt.gcf()
    assert len(fig.axes) == 6  # 2 hàng x 3 cột: RGB / mask / overlay
    plt.close("all")