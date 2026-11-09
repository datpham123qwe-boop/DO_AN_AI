"""Test cho src/datasets/label_mapping.py. Đặt tại: tests/test_label_mapping.py"""
import numpy as np
import pytest

from src.datasets.label_mapping import (
    BACKGROUND,
    CLASS_NAMES,
    CROP,
    IGNORE_INDEX,
    NUM_CLASSES,
    WEED,
    map_labels,
)


def test_constants():
    assert (BACKGROUND, CROP, WEED) == (0, 1, 2)
    assert NUM_CLASSES == 3
    assert CLASS_NAMES == ["Background", "Crop", "Weed"]
    assert IGNORE_INDEX == 255


@pytest.mark.parametrize("raw_id, expected_id", [(0, 0), (1, 1), (2, 2), (3, 1), (4, 2)])
def test_merge_mode_each_id(raw_id, expected_id):
    """Option A: partial crop -> crop, partial weed -> weed."""
    out = map_labels(np.full((4, 4), raw_id, dtype=np.uint16), mode="merge")
    assert (out == expected_id).all()


@pytest.mark.parametrize("raw_id, expected_id", [(0, 0), (1, 1), (2, 2), (3, 255), (4, 255)])
def test_ignore_mode_each_id(raw_id, expected_id):
    """Option B: partial -> ignore_index (255)."""
    out = map_labels(np.full((4, 4), raw_id, dtype=np.uint16), mode="ignore")
    assert (out == expected_id).all()


def test_default_mode_is_merge():
    raw = np.array([[3, 4]], dtype=np.uint16)
    assert map_labels(raw).tolist() == [[1, 2]]


def test_mixed_mask_position_preserved():
    raw = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint16)
    assert map_labels(raw).tolist() == [[0, 1, 2], [1, 2, 0]]


def test_shape_and_dtype():
    raw = np.random.default_rng(0).integers(0, 5, (32, 48)).astype(np.uint16)
    out = map_labels(raw)
    assert out.shape == raw.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.int64])
def test_accepts_common_integer_dtypes(dtype):
    out = map_labels(np.array([[0, 3, 4]], dtype=dtype))
    assert out.tolist() == [[0, 1, 2]]


def test_input_is_not_modified():
    raw = np.array([[0, 3, 4]], dtype=np.uint16)
    before = raw.copy()
    map_labels(raw)
    assert np.array_equal(raw, before)


def test_output_only_contains_project_classes():
    raw = np.random.default_rng(1).integers(0, 5, (64, 64)).astype(np.uint16)
    assert set(np.unique(map_labels(raw)).tolist()) <= {0, 1, 2}


@pytest.mark.parametrize("bad_id", [5, 10, 255])
def test_unknown_id_raises(bad_id):
    with pytest.raises(ValueError):
        map_labels(np.array([[0, bad_id]], dtype=np.uint16))


def test_non_2d_mask_raises():
    with pytest.raises(ValueError):
        map_labels(np.zeros((2, 4, 4), dtype=np.uint16))


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        map_labels(np.zeros((4, 4), dtype=np.uint16), mode="wrong")