import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from datetime import datetime
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
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
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
            "samples": self.samples
        }

    def create_dirs(self):
        for val in self.paths.values():
            val.mkdir(parents=True, exist_ok=True)
            
    def __str__(self):
        return str(self.experiment_dir)


class PrintableArgs:
    def print_args(self) -> None:
        print(f"\n{self.__class__.__name__}:")
        print("-" * 50)
        for key, value in vars(self).items():
            if isinstance(value, Path):
                print(f"{key:25} = {value.resolve()}")
            else:
                print(f"{key:25} = {value}")
        print("-" * 50 + "\n")
        
    def __str__(self):
        return f"{self.__class__.__name__}: " + ", ".join(f"{k}={v}" for k, v in vars(self).items())


@dataclass
class TrainingArguments(PrintableArgs):
    source: Path
    target: ExperimentDirectory

    batch_size: int = 32
    img_size: int = 256
    conf: float = 0.5
    seed: int = 42

    lr: float = 1e-4
    warmup_epochs: int = 3
    weight_decay: float = 1e-4
    epochs: int = 100
    es_patience: int = 10
    es_delta: float = 1e-4
    arch: Literal["unet", "unetplusplus", "linknet"] = "unetplusplus"
    backbone: Literal[
        "resnet18", "resnet34", "resnet50", "mobilenet_v2", "efficientnet-b0"
    ] = "efficientnet-b0"
    weights: Literal["imagenet", None] = "imagenet"


@dataclass
class TestArguments(PrintableArgs):
    model_path: Path
    source: Path
    target: ExperimentDirectory

    batch_size: int = 32
    img_size: int = 256
    conf: float = 0.5
    seed: int = 42


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
