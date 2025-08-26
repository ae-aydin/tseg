from pathlib import Path

import yaml
from pydantic import BaseModel


class LoggableModel(BaseModel):
    def __str__(self) -> str:
        parts = []
        for k, v in self.model_dump().items():
            if isinstance(v, Path):
                v = v.resolve()
            parts.append(f"{k}={v}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class DataConfig(LoggableModel):
    source: Path
    target: str
    img_size: int
    seed: int


class ModelConfig(LoggableModel):
    arch: str
    backbone: str
    weights: str


class TrainConfig(LoggableModel):
    batch_size: int
    lr: float
    warmup_epochs: int
    weight_decay: float
    epochs: int
    es_patience: int
    es_delta: float
    conf: float


class TestConfig(LoggableModel):
    batch_size: int
    conf: float
    num_samples: int


class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    test: TestConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def print(self) -> None:
        print("\nConfiguration loaded from settings:")
        print("-" * 50)
        for section_name in ("data", "model", "train", "test"):
            section = getattr(self, section_name)
            print(f"[{section_name}]")
            for key, value in section.model_dump().items():
                if isinstance(value, Path):
                    value = value.resolve()
                print(f"  {key:25} = {value}")
        print("-" * 50 + "\n")

    def log(self, logger):
        logger.info(self.data)
        logger.info(self.model)
        logger.info(self.train)
        logger.info(self.test)
