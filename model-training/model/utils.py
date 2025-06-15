from loguru import logger


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
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
                    logger.success(f"Early stopping triggered after {epoch + 1} epochs")
                    logger.success(f"Best value: {self.best_value:.4f} at epoch {self.best_epoch + 1}")
        return self.should_stop
