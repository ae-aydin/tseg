from pathlib import Path

import typer
from loguru import logger

from tseg.split import train_test_split


def main(
    tiles_path: str,  # path containing tile folders
    export_path: str,  # path to create dataset folder
    tile_ratio: float = 1.0,  # ratio of tiles to copy
    ratio: float = 0.85,  # train-test split
    visualize: bool = False,  # visualize yolo annotations
):
    logger.info("Dataset Preparation for tseg")
    train_test_split(
        Path(tiles_path), Path(export_path), tile_ratio, ratio, visualize
    )
    logger.info("Finished.")


if __name__ == "__main__":
    typer.run(main)
