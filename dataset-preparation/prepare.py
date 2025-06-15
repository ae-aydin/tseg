import random
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
    seed: int = -1,  # random seed for reproducibility
):
    logger.info("Dataset Preparation for tseg")
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"Using random seed: {seed}")
    else:
        logger.info(f"Using seed: {seed}")

    train_test_split(
        Path(source),
        Path(target),
        train_ratio,
        val_ratio,
        yolo_format,
        seed,
    )
    logger.info("Finished")


if __name__ == "__main__":
    typer.run(main)
