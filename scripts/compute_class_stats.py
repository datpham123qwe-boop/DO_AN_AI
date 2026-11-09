"""Tính thống kê lớp (class statistics) cho PhenoBench và lưu ra JSON.

Đặt file này tại: scripts/compute_class_stats.py
Chạy từ thư mục gốc của repo:

    python scripts/compute_class_stats.py
    python scripts/compute_class_stats.py --data_root data/raw/PhenoBench-v110/PhenoBench --split train

Kết quả: data/processed/class_stats.json

File JSON chứa:
  - số pixel và tỉ lệ của 5 ID gốc (trước khi mapping)
  - số pixel và tỉ lệ của 3 lớp project (sau khi mapping)
  - class weights theo 3 cách tính (B dùng cho Weighted CE)

Class weights chỉ nên tính từ tập TRAIN, không dùng val/test.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# Cho phép chạy trực tiếp từ thư mục gốc của repo
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.label_mapping import (  # noqa: E402
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    map_labels,
)

RAW_NAMES = ["Background", "Crop", "Weed", "Partial Crop", "Partial Weed"]
NUM_RAW = len(RAW_NAMES)


def compute_counts(sem_dir: Path, mode: str):
    """Đếm pixel của từng ID gốc và từng lớp sau mapping trên toàn bộ split."""
    files = sorted(sem_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy mask nào trong {sem_dir}")

    raw_counts = np.zeros(NUM_RAW, dtype=np.int64)
    mapped_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    ignored = 0

    for i, path in enumerate(files, start=1):
        raw = np.array(Image.open(path))
        if raw.max() >= NUM_RAW:
            raise ValueError(
                f"{path.name} có ID ngoài 0-{NUM_RAW - 1}: {np.unique(raw)}"
            )
        raw_counts += np.bincount(raw.ravel(), minlength=NUM_RAW)[:NUM_RAW]

        mapped = map_labels(raw, mode)
        bc = np.bincount(mapped.ravel(), minlength=IGNORE_INDEX + 1)
        mapped_counts += bc[:NUM_CLASSES]
        ignored += int(bc[IGNORE_INDEX])

        if i % 100 == 0 or i == len(files):
            print(f"  đã xử lý {i}/{len(files)} ảnh")

    return len(files), raw_counts, mapped_counts, ignored


def compute_weights(freq: np.ndarray) -> dict:
    """Ba cách tính class weights. Tất cả được chuẩn hóa để trung bình = 1."""

    def normalize(w):
        return w / w.sum() * len(w)

    inverse = 1.0 / freq
    inverse_sqrt = 1.0 / np.sqrt(freq)
    median_freq = np.median(freq) / freq

    return {
        "inverse_sqrt": normalize(inverse_sqrt).tolist(),
        "inverse": normalize(inverse).tolist(),
        "median_frequency": normalize(median_freq).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Class statistics cho PhenoBench")
    parser.add_argument("--data_root", default="data/raw/PhenoBench-v110/PhenoBench")
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", default="merge", choices=["merge", "ignore"],
                        help="merge: partial -> crop/weed; ignore: partial -> 255")
    parser.add_argument("--out", default="data/processed/class_stats.json")
    parser.add_argument("--sem_folder", default="semantics",
                        help="tên thư mục chứa mask (sửa theo cây thư mục thật)")
    args = parser.parse_args()

    if args.split != "train":
        print("CẢNH BÁO: class weights nên tính từ tập train, không phải "
              f"'{args.split}'.")

    sem_dir = Path(args.data_root) / args.split / args.sem_folder
    print(f"Đọc mask từ: {sem_dir} (mode={args.mode})")

    num_images, raw_counts, counts, ignored = compute_counts(sem_dir, args.mode)

    total_pixels = int(raw_counts.sum())
    counted = int(counts.sum())
    if counted == 0:
        raise RuntimeError("Không có pixel nào thuộc 3 lớp, kiểm tra lại mapping.")
    freq = counts / counted  # tỉ lệ tính trên các pixel không bị ignore

    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "split": args.split,
        "label_mode": args.mode,
        "num_images": num_images,
        "total_pixels": total_pixels,
        "ignored_pixels": ignored,
        "raw_ids": {
            RAW_NAMES[i]: {
                "id": i,
                "pixels": int(raw_counts[i]),
                "fraction": float(raw_counts[i] / total_pixels),
            }
            for i in range(NUM_RAW)
        },
        "classes": {
            CLASS_NAMES[c]: {
                "id": c,
                "pixels": int(counts[c]),
                "frequency": float(freq[c]),
            }
            for c in range(NUM_CLASSES)
        },
        "class_names": CLASS_NAMES,
        "class_weights": compute_weights(freq),
        "recommended_weights": "inverse_sqrt",
        "weight_formulas": {
            "inverse_sqrt": "w_c = 1/sqrt(freq_c), chuẩn hóa để mean(w) = 1",
            "inverse": "w_c = 1/freq_c, chuẩn hóa để mean(w) = 1",
            "median_frequency": "w_c = median(freq)/freq_c, chuẩn hóa để mean(w) = 1",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # In tóm tắt ra màn hình
    print(f"\n{num_images} ảnh, {total_pixels:,} pixel")
    print("\nID gốc:")
    for name, info in stats["raw_ids"].items():
        print(f"  {info['id']} {name:<13} {info['pixels']:>15,}  "
              f"{info['fraction'] * 100:6.2f}%")
    print("\nSau mapping:")
    for name, info in stats["classes"].items():
        print(f"  {info['id']} {name:<13} {info['pixels']:>15,}  "
              f"{info['frequency'] * 100:6.2f}%")
    if ignored:
        print(f"  (ignore) {ignored:,} pixel")
    print("\nClass weights:")
    for method, w in stats["class_weights"].items():
        print(f"  {method:<17} {[round(x, 3) for x in w]}")
    print(f"\nĐã lưu: {out_path}")


if __name__ == "__main__":
    main()