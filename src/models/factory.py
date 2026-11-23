def build_model(name, pretrained=True):
    if name == "deeplab":
        from src.models.deeplab import modelDeepLabV3
        return modelDeepLabV3(pretrained=pretrained)
    if name == "segformer":
        from src.models.segformer import modelSegformer
        return modelSegformer(pretrained=pretrained)
    raise ValueError(f"Unsupported model: {name}")
