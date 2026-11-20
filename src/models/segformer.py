import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

class SegFormerWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=3,
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values):
        output = self.backbone(pixel_values=pixel_values)
        logits = F.interpolate(
            output.logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits


def modelSegformer():
    return SegFormerWrapper()