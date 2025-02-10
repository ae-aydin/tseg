from pathlib import Path

import typer
from loguru import logger

from tseg import split


def main(
    tiles_path: str,  # path containing tile folders
    export_path: str,  # path to create dataset folder
    ratio: float = 0.85,  # train-test split
    visualize: bool = False,  # visualize yolo annotations
):
    logger.info("Dataset Preparation for tseg")
    split.train_test_split(Path(tiles_path), Path(export_path), ratio, visualize)
    logger.info("Finished.")


if __name__ == "__main__":
    typer.run(main)
