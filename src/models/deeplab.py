
import segmentation_models_pytorch as smp


def modelDeepLabV3(pretrained=True):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=3,
        activation=None,
    )
    return model
