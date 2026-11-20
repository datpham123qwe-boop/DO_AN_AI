
import segmentation_models_pytorch as smp
def modelDeepLabV3():
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3, 
        classes=3,
        activation = None)
    return model
