import segmentation_models_pytorch as smp


def get_smp_model(
    arch: str = "unetplusplus",
    backbone: str = "efficientnet-b0",
    weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
):
    return smp.create_model(
        arch=arch,
        encoder_name=backbone,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=classes,
    )
