import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegFormerWrapper(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            self.backbone = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512",
                num_labels=3,
                ignore_mismatched_sizes=True,
                id2label={0: "Background", 1: "Crop", 2: "Weed"},
                label2id={"Background": 0, "Crop": 1, "Weed": 2},
            )
        else:
            # B0 architecture; checkpoint loading supplies all learned weights.
            self.backbone = SegformerForSemanticSegmentation(SegformerConfig(
                num_labels=3, hidden_sizes=[32, 64, 160, 256],
                depths=[2, 2, 2, 2], decoder_hidden_size=256,
                id2label={0: "Background", 1: "Crop", 2: "Weed"},
                label2id={"Background": 0, "Crop": 1, "Weed": 2},
            ))

    def forward(self, pixel_values):
        output = self.backbone(pixel_values=pixel_values)
        logits = F.interpolate(
            output.logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits


def modelSegformer(pretrained=True):
    return SegFormerWrapper(pretrained=pretrained)
