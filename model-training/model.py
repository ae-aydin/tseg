import segmentation_models_pytorch as smp


def get_smp_model(
    arch: str = "unet",
    backbone: str = "mobilenet_v2",
    weights: str = "imagenet",
):
    return smp.create_model(
        arch=arch,
        encoder_name=backbone,
        encoder_weights=weights,
        encoder_depth=3,
        decoder_channels=[128, 64, 16],
        in_channels=3,
        classes=1,
    )


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, epoch: int, value: float) -> bool:
        if self.mode == "min":
            is_better = value < self.best_value - self.min_delta
        else:
            is_better = value > self.best_value + self.min_delta

        if is_better:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(
                        f"\nEarly stopping triggered after {epoch + 1} epochs. "
                        f"Best value: {self.best_value:.4f} at epoch {self.best_epoch + 1}"
                    )

        return self.should_stop
