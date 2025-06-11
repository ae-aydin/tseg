import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch


class PrintableArgs:
    def print_args(self) -> None:
        print(f"\n{self.__class__.__name__}:")
        print("-" * 50)
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                print(f"{key:25} = {value.resolve()}")
            else:
                print(f"{key:25} = {value}")
        print("-" * 50 + "\n")


@dataclass
class TrainingArguments(PrintableArgs):
    model_path: Path
    source: Path
    target: Path

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
    target: Path

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
