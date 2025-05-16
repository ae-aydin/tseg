from pathlib import Path

import typer
from loguru import logger

from tseg import train_test_split


def main(
    source: str,  # path containing tile folders
    target: str,  # path to create dataset folder
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    yolo_format: bool = False,  # create yolo dataset
):
    logger.info("Dataset Preparation for tseg")
    train_test_split(
        Path(source),
        Path(target),
        train_ratio,
        val_ratio,
        yolo_format,
    )
    logger.info("Finished")


if __name__ == "__main__":
    typer.run(main)
