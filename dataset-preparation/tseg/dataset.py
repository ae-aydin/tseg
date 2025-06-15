import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from .data_ops import extract_slide_info, extract_tile_info
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


def construct_dataset(source: Path, dirs: DatasetDirectory) -> None:
    """Copy files according to CSV file containing split information. Extracts tile metadata."""
    logger.info(f"Started constructing dataset at {dirs.parent}")
    slide_info_entries, all_tile_info_entries = [], []
    slide_folders = list(source.iterdir())
    pbar = tqdm(
        slide_folders,
        total=len(slide_folders),
        position=0,
        leave=True,
        ncols=100,
    )

    for slide_folder_path in pbar:
        slide_info_entries.append(extract_slide_info(slide_folder_path))
        current_slide_name = str(slide_folder_path.name).split("|")[1]
        pbar.set_description(f"Processing Slides | {pad_str(current_slide_name)}")

        target_path = dirs.slides / current_slide_name
        shutil.copytree(slide_folder_path, target_path)

        add_suffix_to_dir_items(target_path / "masks")
        tile_info_entries = extract_tile_info(target_path)
        all_tile_info_entries.extend(tile_info_entries)

    # Save tile information
    save_csv(all_tile_info_entries, dirs.metadata / "tile_info.csv")
    save_csv(slide_info_entries, dirs.metadata / "slide_info.csv")

    logger.info(f"Dataset constructed at {dirs.parent}")
    logger.info(f"Slide and tile metadata saved at {dirs.metadata}")


def split_tiles(
    dirs: DatasetDirectory,
    train_ratio: float,
    hpa_train_only: bool = True,
    create_dev: bool = True,
    dev_test_ratio: float = 0.5,
    generate_cv: bool = True,
    n_folds: int = 5,
    seed: int = 42,
):
    np.random.seed(seed)

    info = pl.read_csv(dirs.metadata / "slide_info.csv")
    wsi = info.filter(pl.col("category") == "wsi_tiled")["slide_name"].to_list()
    hpa = info.filter(pl.col("category") == "img_tiled")["slide_name"].to_list()
    base = wsi if hpa_train_only else wsi + hpa

    train, remainder = train_test_split(base, train_size=train_ratio, random_state=seed)
    val, test = (
        train_test_split(remainder, train_size=dev_test_ratio, random_state=seed)
        if create_dev
        else ([], remainder)
    )

    split_dict = {}
    split_dict.update({s: "train" for s in train})
    split_dict.update({s: "val" for s in val})
    split_dict.update({s: "test" for s in test})
    if hpa_train_only:
        split_dict.update({s: "train" for s in hpa})

    # base split
    def write_split(path: Path, mapping: dict[str, str]) -> None:
        pl.DataFrame(
            {"slide_name": list(mapping), "split": [mapping[s] for s in mapping]}
        ).write_csv(path)

    write_split(dirs.metadata / "split_info.csv", split_dict)
    logger.info(f"Split metadata saved at {dirs.metadata}")

    if not generate_cv:
        return

    cv_dir = dirs.metadata / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    eligible = [
        s
        for s in split_dict
        if split_dict[s] != "test" and (s in wsi or not hpa_train_only)
    ]

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(kfold.split(eligible)):
        fold_split = {
            s: ("test" if split_dict[s] == "test" else "train") for s in split_dict
        }
        for s in (eligible[i] for i in val_idx):
            fold_split[s] = "val"
        write_split(cv_dir / f"split_fold_{fold}.csv", fold_split)
    logger.info(f"K-fold cross-validation metadata saved at {cv_dir}")


def split(
    source: Path,
    target: Path,
    train_ratio: float,
    hpa_train_only: bool = True,
    create_dev: bool = True,
    dev_test_ratio: float = 0.5,
    generate_cv: bool = float,
    k_folds: int = 5,
    use_yolo_format: bool = False,
    seed: int = 42,
):
    if use_yolo_format:
        raise NotImplementedError

    dirs = DatasetDirectory(target)
    construct_dataset(source, dirs)
    save_yaml({"seed": seed}, dirs.metadata, "seed.yaml")
    split_tiles(
        dirs,
        train_ratio,
        hpa_train_only,
        create_dev,
        dev_test_ratio,
        generate_cv,
        k_folds,
        seed,
    )
