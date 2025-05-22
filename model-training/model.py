import segmentation_models_pytorch as smp
import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self):
        super().__init__()


class SMPUnet(nn.Module):
    def __init__(
        self,
        arch: str = "unetplusplus",
        backbone: str = "mobilenet_v2",
        weights: str = "imagenet",
    ) -> None:
        super().__init__()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=3,
            classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
