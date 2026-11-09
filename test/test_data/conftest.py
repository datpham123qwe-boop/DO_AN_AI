"""Fixture dùng chung cho mọi test. Đặt tại: tests/conftest.py

Tạo một bộ dữ liệu PhenoBench GIẢ (nhỏ, chạy trong vài giây) với cấu trúc:
    <root>/train/images/*.png, <root>/train/semantics/*.png   (3 ảnh)
    <root>/val/images/*.png,   <root>/val/semantics/*.png     (2 ảnh)

Mỗi mask 128x128 (uint16, giống PhenoBench) chia theo CỘT nên biết trước
số pixel của từng ID:
    cột   0- 79 -> ID 0 (Background)
    cột  80-103 -> ID 1 (Crop)
    cột 104-111 -> ID 2 (Weed)
    cột 112-119 -> ID 3 (Partial Crop)
    cột 120-127 -> ID 4 (Partial Weed)

Kênh R của ảnh = (nhãn sau mapping) * 80, để test được ảnh và mask
có còn khớp nhau sau crop/lật/xoay hay không.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

H = W = 128
COLUMNS = [(0, 80, 0), (80, 104, 1), (104, 112, 2), (112, 120, 3), (120, 128, 4)]

# Bảng mapping viết lại độc lập, để không dùng chính code đang được test làm đáp án
MERGE_LUT = np.array([0, 1, 2, 1, 2], dtype=np.uint8)

TRAIN_FILES = ["05-15_00001_P001.png", "05-26_00002_P002.png", "06-05_00003_P003.png"]
VAL_FILES = ["05-15_00010_P010.png", "06-05_00011_P011.png"]

RAW_PER_IMAGE = {0: 10240, 1: 3072, 2: 1024, 3: 1024, 4: 1024}
MERGED_PER_IMAGE = {0: 10240, 1: 4096, 2: 2048}   # 3 -> Crop, 4 -> Weed
IGNORED_PER_IMAGE = 2048                          # ID 3 + ID 4 khi mode="ignore"


def _make_raw_mask():
    mask = np.zeros((H, W), dtype=np.uint16)
    for c0, c1, value in COLUMNS:
        mask[:, c0:c1] = value
    return mask


def _make_image(raw_mask, rng):
    image = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    image[..., 0] = MERGE_LUT[raw_mask] * 80
    return image


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("PhenoBench")
    rng = np.random.default_rng(0)
    for split, names in (("train", TRAIN_FILES), ("val", VAL_FILES)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "semantics").mkdir(parents=True)
        for name in names:
            raw = _make_raw_mask()
            Image.fromarray(_make_image(raw, rng)).save(root / split / "images" / name)
            Image.fromarray(raw).save(root / split / "semantics" / name)  # uint16
    return root


@pytest.fixture(scope="session")
def expected():
    return {
        "size": (H, W),
        "n_train": len(TRAIN_FILES),
        "n_val": len(VAL_FILES),
        "train_files": TRAIN_FILES,
        "raw_per_image": RAW_PER_IMAGE,
        "merged_per_image": MERGED_PER_IMAGE,
        "ignored_per_image": IGNORED_PER_IMAGE,
        "pixels_per_image": H * W,
    }