"""Validate and package semantic predictions for the PhenoBench test benchmark."""

import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image

EXPECTED_TEST_IMAGES = 693
EXPECTED_SIZE = (1024, 1024)


def validate_predictions(predictions, test_images, expected_count=EXPECTED_TEST_IMAGES):
    predictions, test_images = Path(predictions), Path(test_images)
    images = sorted(test_images.glob("*.png"))
    if len(images) != expected_count:
        raise ValueError(f"Expected {expected_count} test images; found {len(images)} in {test_images}")
    expected = {path.name for path in images}
    actual = {path.name for path in predictions.glob("*.png")}
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"Prediction filenames differ: missing={missing[:5]}, extra={extra[:5]}")
    for image_path in images:
        with Image.open(image_path) as image:
            if image.size != EXPECTED_SIZE:
                raise ValueError(f"Test image must be {EXPECTED_SIZE}: {image_path.name}")
        path = predictions / image_path.name
        with Image.open(path) as image:
            labels = np.array(image)
            if image.format != "PNG" or image.size != EXPECTED_SIZE or labels.ndim != 2:
                raise ValueError(f"Prediction must be a single-channel {EXPECTED_SIZE} PNG: {path.name}")
        if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, [0, 1, 2]).all():
            raise ValueError(f"Prediction must contain only class IDs 0/1/2: {path.name}")
    return [predictions / path.name for path in images]


def write_submission(files, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Submission exists; choose another output filename: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, allowZip64=True) as archive:
        # The official validator checks for an explicit directory entry.
        archive.writestr("semantics/", b"")
        for path in files:
            with Image.open(path) as image:
                labels = np.array(image).astype(np.uint16)
            buffer = BytesIO()
            Image.fromarray(labels).save(buffer, format="PNG")
            archive.writestr("semantics/" + path.name, buffer.getvalue())
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--test-images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".zip":
        parser.error("--output must end with .zip")
    files = validate_predictions(args.predictions, args.test_images)
    output = write_submission(files, args.output)
    print(f"Validated {len(files)} semantic masks: IDs 0/1/2, 1024x1024, uint16 PNG")
    print(f"Submission ZIP: {output}")


if __name__ == "__main__":
    main()
