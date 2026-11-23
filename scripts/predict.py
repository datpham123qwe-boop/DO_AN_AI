"""Predict class/spray masks for RGB images, including test images without GT."""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
from PIL import Image

from src.datasets.augmentation import val_transform
from src.evaluation.figures import save_qualitative_figure
from src.evaluation.inference import predict_logits
from src.evaluation.io import (choose_device, inference_protocol, load_model_bundle,
                               prepare_output, resolve_path, runtime_info, write_json)
from src.spraying.spray_mask import make_spray_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image")
    inputs.add_argument("--input-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--model-name", "--model_name", choices=["deeplab", "segformer"])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--inference-mode", choices=["auto", "full", "tiled"], default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-figures", type=int, default=10)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.max_figures < 0 or (args.limit is not None and args.limit < 1):
        parser.error("Invalid limit/figure count")
    files = [resolve_path(args.image)] if args.image else sorted(
        p for p in resolve_path(args.input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    if not files:
        raise FileNotFoundError("No input RGB images")
    available = len(files)
    files = files[:args.limit] if args.limit else files
    if len({p.stem for p in files}) != len(files):
        raise ValueError("Input images have duplicate stems; avoid output filename collisions")
    device = choose_device(args.device)
    model, config, info = load_model_bundle(args.checkpoint, device, args.config,
                                           args.model_name, args.allow_smoke_checkpoint)
    protocol = inference_protocol(config, args.inference_mode, args.tile_size, not args.no_amp)
    output = prepare_output(args.output_dir)
    for folder in ("predictions", "spray_masks", "figures"):
        (output / folder).mkdir(exist_ok=True)
    transform = val_transform()
    for index, path in enumerate(files):
        with Image.open(path) as file:
            image = np.array(file.convert("RGB"))
        tensor = transform(image=image)["image"].unsqueeze(0).to(device)
        pred = predict_logits(model, tensor, protocol["tile_size"], protocol["amp_requested"]).argmax(1)[0].cpu().numpy().astype(np.uint8)
        name = path.stem + ".png"
        Image.fromarray(pred).save(output / "predictions" / name)
        Image.fromarray(make_spray_mask(pred).astype(np.uint8) * 255).save(output / "spray_masks" / name)
        if index < args.max_figures:
            save_qualitative_figure(output / "figures" / name, image, pred, title=path.name)
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(files):
            print(f"Predicted {index + 1}/{len(files)} images", flush=True)
    write_json(output / "predictions_manifest.json", {
        "checkpoint": info, "protocol": protocol, "runtime": runtime_info(device),
        "available_images": available, "predicted_images": len(files),
        "input_files": [str(path) for path in files],
        "notes": "No ground truth supplied: no segmentation/spraying performance metrics computed.",
    })


if __name__ == "__main__":
    main()
