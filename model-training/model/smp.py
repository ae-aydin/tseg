import segmentation_models_pytorch as smp


def get_smp_model(
    arch: str = "unetplusplus",
    backbone: str = "efficientnet-b0",
    weights: str = "imagenet",
    encoder_depth: int = 5,
    decoder_channels: list[int] = [256, 128, 64, 32, 16],
    in_channels: int = 3,
    classes: int = 1,
):
    return smp.create_model(
        arch=arch,
        encoder_name=backbone,
        encoder_weights=weights,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
        in_channels=in_channels,
        classes=classes,
    )
