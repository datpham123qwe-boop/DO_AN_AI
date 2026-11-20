import torch
from torch import nn

def build_ce_loss(ignore_index = 255):
    return nn.CrossEntropyLoss(ignore_index=ignore_index)

def build_weight_ce_loss(class_weight, ignore_index=255):
    if len(class_weight) != 3:
        raise ValueError(
            f"Weighted CE requires 3 class weights, received {len(class_weight)}"
        )
    weight = torch.tensor(class_weight,
                       dtype= torch.float32)
    return nn.CrossEntropyLoss(weight=weight,ignore_index=ignore_index)

def build_loss(name="ce", class_weight=None, ignore_index = 255):
    name = name.lower()
    if name in ("ce","cross_entropy"):
        return build_ce_loss(ignore_index=ignore_index)
    if name in ("wce", "weighted_cross_entropy"):
        if class_weight is None:
            raise ValueError("Weight must be passed!")
        return build_weight_ce_loss(
            class_weight,
            ignore_index
        )
    raise ValueError(f"Unsupported loss: {name}")
