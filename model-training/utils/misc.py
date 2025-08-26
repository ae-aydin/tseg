import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


@dataclass
class ExperimentDirectory:
    experiment_name: str
    parent: Path

    _checkpoints: str = "checkpoints"
    _logs: str = "logs"
    _predictions: str = "predictions"
    _samples: str = "samples"

    def __post_init__(self):
        self.timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        self.create_dirs()

    @property
    def experiment_dir(self):
        return self.parent / f"{self.experiment_name}_{self.timestamp}"

    @property
    def checkpoints(self):
        return self.experiment_dir / self._checkpoints

    @property
    def logs(self):
        return self.experiment_dir / self._logs

    @property
    def predictions(self):
        return self.experiment_dir / self._predictions

    @property
    def samples(self):
        return self.experiment_dir / self._samples

    @property
    def paths(self) -> dict:
        return {
            "checkpoints": self.checkpoints,
            "logs": self.logs,
            "predictions": self.predictions,
            "samples": self.samples,
        }

    def create_dirs(self):
        for val in self.paths.values():
            val.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        return str(self.experiment_dir)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
