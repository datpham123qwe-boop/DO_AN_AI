"""Full-resolution or non-overlapping tiled inference; no image downscaling."""

import torch
import torch.nn.functional as F


def _forward(model, images, amp):
    height, width = images.shape[-2:]
    padded = F.pad(images, (0, (-width) % 32, 0, (-height) % 32))
    with torch.autocast(device_type=images.device.type,
                        enabled=bool(amp and images.device.type == "cuda")):
        logits = model(padded)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[:2] != (images.shape[0], 3):
        raise ValueError("Model must return [B,3,H,W] logits")
    if logits.shape[-2:] != padded.shape[-2:]:
        logits = F.interpolate(logits, size=padded.shape[-2:], mode="bilinear", align_corners=False)
    return logits[..., :height, :width]


@torch.inference_mode()
def predict_logits(model, images, tile_size=None, amp=True):
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Images must be normalized [B,3,H,W] tensors")
    if tile_size is None:
        return _forward(model, images, amp)
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError("tile_size must be a positive integer")
    height, width = images.shape[-2:]
    stitched = None
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            patch = images[..., top:top + tile_size, left:left + tile_size]
            ph, pw = patch.shape[-2:]
            patch = F.pad(patch, (0, tile_size - pw, 0, tile_size - ph))
            logits = _forward(model, patch, amp)[..., :ph, :pw]
            if stitched is None:
                stitched = logits.new_empty((images.shape[0], 3, height, width))
            stitched[..., top:top + ph, left:left + pw] = logits
    return stitched
