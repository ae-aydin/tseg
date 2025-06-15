import random
from pathlib import Path

import typer
from loguru import logger

from tseg import split


def main(
    source: str,  # path containing tile folders
    target: str,  # path to create dataset folder
    train_ratio: float = 0.7,
    hpa_train_only: bool = True,  # use all hpa slides as train
    create_dev: bool = True,  # create validation set
    dev_test_ratio: float = 0.5,
    generate_cv: bool = True,  # generate k fold cross validation
    k_folds: int = 5,
    use_yolo_format: bool = False,  # create yolo dataset
    seed: int = -1,  # random seed for reproducibility
):
    logger.info("Dataset Preparation for tseg")
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"Using random seed: {seed}")
    else:
        logger.info(f"Using seed: {seed}")

    split(
        Path(source),
        Path(target),
        train_ratio,
        hpa_train_only,
        create_dev,
        dev_test_ratio,
        generate_cv,
        k_folds,
        use_yolo_format,
        seed,
    )
    logger.info("Finished")


if __name__ == "__main__":
    typer.run(main)
