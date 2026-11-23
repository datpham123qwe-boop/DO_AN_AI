from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from scripts.package_submission import validate_predictions, write_submission


def create_pair(tmp_path):
    images, predictions = tmp_path / "test/images", tmp_path / "predictions"
    images.mkdir(parents=True)
    predictions.mkdir()
    Image.new("RGB", (1024, 1024)).save(images / "a.png")
    labels = np.zeros((1024, 1024), dtype=np.uint8)
    labels[:, 300:600] = 1
    labels[:, 600:] = 2
    Image.fromarray(labels).save(predictions / "a.png")
    return images, predictions, labels


def test_submission_zip_preserves_class_ids_and_uses_uint16_png(tmp_path):
    images, predictions, labels = create_pair(tmp_path)
    files = validate_predictions(predictions, images, expected_count=1)
    output = write_submission(files, tmp_path / "submission.zip")
    with ZipFile(output) as archive:
        assert archive.namelist() == ["semantics/", "semantics/a.png"]
        payload = archive.read("semantics/a.png")
        assert payload[24] == 16  # PNG IHDR bit depth.
        with Image.open(BytesIO(payload)) as image:
            np.testing.assert_array_equal(np.array(image), labels)
    with pytest.raises(FileExistsError):
        write_submission(files, output)


def test_partial_test_input_is_rejected_by_default(tmp_path):
    images, predictions, _ = create_pair(tmp_path)
    with pytest.raises(ValueError, match="Expected 693"):
        validate_predictions(predictions, images)


@pytest.mark.parametrize("problem", ["missing", "extra", "rgb", "spray", "wrong_size"])
def test_invalid_submission_masks_are_rejected(tmp_path, problem):
    images, predictions, labels = create_pair(tmp_path)
    path = predictions / "a.png"
    if problem == "missing":
        path.rename(predictions / "different.png")
    elif problem == "extra":
        Image.fromarray(labels).save(predictions / "extra.png")
    elif problem == "rgb":
        Image.new("RGB", (1024, 1024)).save(path)
    elif problem == "spray":
        Image.fromarray((labels == 2).astype(np.uint8) * 255).save(path)
    else:
        Image.new("L", (512, 512)).save(path)
    with pytest.raises(ValueError):
        validate_predictions(predictions, images, expected_count=1)
