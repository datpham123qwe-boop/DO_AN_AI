"""Package project source and optional train/validation data for Kaggle/Colab."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def source_files():
    files = []
    for folder in ("src", "scripts", "configs", "test"):
        files.extend(p for p in (REPO / folder).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".ps1")
                     and "__pycache__" not in p.parts)
    for relative in ("README.md", "requirements.txt", "requirements-cloud.txt", "pytest.ini",
                     "notebooks/train_cloud.ipynb", "notebooks/evaluate_cloud.ipynb",
                     "notebooks/result_analysis.ipynb", "notebooks/predict_test.ipynb"):
        path = REPO / relative
        if path.is_file():
            files.append(path)
    return sorted(set(files))


def write_source_zip(destination):
    files = source_files()
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "sha256": {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in files},
    }
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in files:
            archive.write(path, "DO_AN_AI/" + path.relative_to(REPO).as_posix())
        archive.writestr("DO_AN_AI/source_manifest.json", json.dumps(manifest, indent=2))
    print(f"Source: {destination} ({len(files)} files)", flush=True)


def write_dataset_zip(root, destination):
    entries = []
    for split in ("train", "val"):
        images = sorted((root / split / "images").glob("*.png"))
        if not images:
            raise FileNotFoundError(f"No images in {root / split / 'images'}")
        for image in images:
            mask = root / split / "semantics" / image.name
            if not mask.is_file():
                raise FileNotFoundError(mask)
            entries.extend((image, mask))
    # PNG already compresses its data. Store files directly to avoid slow recompression.
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for index, path in enumerate(entries, 1):
            archive.write(path, "PhenoBench/" + path.relative_to(root).as_posix())
            if index % 500 == 0 or index == len(entries):
                print(f"Dataset: {index}/{len(entries)} files", flush=True)
        readme = root / "README.MD"
        if readme.is_file():
            archive.write(readme, "PhenoBench/README.MD")
    print(f"Dataset: {destination} ({destination.stat().st_size / 1024**3:.2f} GiB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-data", action="store_true")
    parser.add_argument("--data-root", default="../PhenoBench-v110/PhenoBench")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = REPO / "cloud_upload"
    output.mkdir(exist_ok=True)
    code_zip, data_zip = output / "DO_AN_AI_cloud.zip", output / "PhenoBench_trainval.zip"
    targets = [code_zip] + ([data_zip] if args.include_data else [])
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Package exists; use --overwrite to rebuild the selected archives")
    root = Path(args.data_root)
    root = root.resolve() if root.is_absolute() else (REPO / root).resolve()
    if args.include_data and not (root / "train" / "images").is_dir():
        raise FileNotFoundError(root)
    write_source_zip(code_zip)
    if args.include_data:
        write_dataset_zip(root, data_zip)


if __name__ == "__main__":
    main()
