from pathlib import Path

import typer
from loguru import logger

from tseg import train_test_split


def main(
    tiles_path: str,  # path containing tile folders
    export_path: str,  # path to create dataset folder
    tile_count: int = -1,  # maximum tile count in a wsi
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    yolo_format: bool = False,  # create yolo dataset
    visualize: bool = False,  # visualize yolo annotations
):
    logger.info("Dataset Preparation for tseg")
    train_test_split(
        Path(tiles_path),
        Path(export_path),
        tile_count,
        train_ratio,
        val_ratio,
        yolo_format,
        visualize,
    )
    logger.info("Finished.")


if __name__ == "__main__":
    typer.run(main)
