import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from tqdm import tqdm

from .data_ops import categorize_tiles, extract_slide_info, extract_tile_info
from .utils import add_suffix_to_dir_items, pad_str, save_csv, save_yaml


@dataclass
class DatasetDirectory:
    parent: Path
    _slides: str = "slides"
    _metadata: str = "metadata"

    def __post_init__(self):
        logger.info("Creating dataset folders")
        self.create_dirs()

    @property
    def slides(self):
        return self.parent / self._slides

    @property
    def metadata(self):
        return self.parent / self._metadata

    @property
    def paths(self) -> dict:
        return {
            "slides": self.slides,
            "metadata": self.metadata,
        }

    def create_dirs(self):
        for val in self.paths.values():
            val.mkdir(parents=True, exist_ok=True)


def _construct_dataset_list(split: str, tiles: list) -> list:
    """Helper function for split_tiles."""
    return [
        {"full_path": path, "slide_name": path.name.split("|")[1], "split": split}
        for path in tiles
    ]


def split_tiles(
    source: Path, train_ratio: float, val_ratio: float, save_dir: Path, seed: int = 42
) -> pl.DataFrame:
    """Split tiled WSI slides into train-val-test sets virtually. Extract slide and split metadata."""
    np.random.seed(seed)

    categories = categorize_tiles(source)
    logger.info("Creating slide and train-test-split metadata")
    create_test_set = False if np.isclose(train_ratio + val_ratio, 1) else True

    split_entries, slide_info_entries = [], []
    for tile_folders in categories.values():
        np.random.shuffle(tile_folders)
        slide_info_entries.extend(extract_slide_info(tile_folders))

        n_train = int(round(len(tile_folders) * train_ratio))
        n_val = int(round(len(tile_folders) * val_ratio))

        split_entries.extend(_construct_dataset_list("train", tile_folders[:n_train]))

        if create_test_set:
            split_entries.extend(
                _construct_dataset_list("val", tile_folders[n_train : n_train + n_val])
            )
            split_entries.extend(
                _construct_dataset_list("test", tile_folders[n_train + n_val :])
            )
        else:
            split_entries.extend(_construct_dataset_list("val", tile_folders[n_train:]))

    # Save slide information
    save_csv(slide_info_entries, save_dir / "slide_info.csv")
    return pl.DataFrame(split_entries)


def construct_dataset(
    source: Path, split_info_df: pl.DataFrame, dirs: DatasetDirectory
) -> None:
    """Copy files according to CSV file containing split information. Extracts tile metadata."""
    logger.info(f"Started constructing dataset at {dirs.parent}")
    all_tile_info_entries = []

    pbar = tqdm(
        split_info_df.iter_rows(named=True),
        total=len(split_info_df),
        position=0,
        leave=True,
        ncols=100,
    )
    for row_dict in pbar:
        current_slide_name = row_dict["slide_name"]
        pbar.set_description(f"Processing Slides | {pad_str(current_slide_name)}")

        target_path = dirs.slides / current_slide_name
        shutil.copytree(row_dict["full_path"], target_path)

        add_suffix_to_dir_items(target_path / "masks")
        tile_info_entries = extract_tile_info(target_path)
        all_tile_info_entries.extend(tile_info_entries)

    # Remove temporary full path
    split_info_df.drop_in_place("full_path")

    # Save split information
    split_info_df.write_csv(dirs.metadata / "split_info.csv")

    # Save tile information
    save_csv(all_tile_info_entries, dirs.metadata / "tile_info.csv")

    logger.info(f"Dataset constructed at {dirs.parent}")
    logger.info(f"Metadata saved at {dirs.metadata}")


def train_test_split(
    source: Path,
    target: Path,
    train_ratio: float,
    val_ratio: float,
    use_yolo_format: bool,
    seed: int = 42,
):
    if use_yolo_format:
        raise NotImplementedError

    dirs = DatasetDirectory(target)
    save_yaml({"seed": seed}, dirs.metadata, "seed.yaml")
    split_info_df = split_tiles(source, train_ratio, val_ratio, dirs.metadata, seed)
    construct_dataset(source, split_info_df, dirs)
