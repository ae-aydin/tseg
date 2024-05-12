import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_STD_THRESHOLD = 1

# Used for filtering empty images (full white tiles) according to std values.
# Not used for final dataset.


def filter_tiles(main_path: Path):
    """
    Filter tiles based on standard deviation.

    Args:
        main_path (Path): Main folder containing images, masks, etc.

    Returns:
        list: Suitable images.
        list: Empty images.
    """
    images_path = main_path / "images"
    filtered = list()
    empty = list()
    for img_name in tqdm(
        os.listdir(images_path), desc="Filtering empty tiles", ncols=150
    ):
        image = cv2.imread(str(images_path / img_name))
        if np.std(image) > IMAGE_STD_THRESHOLD:
            filtered.append(img_name)
        else:
            empty.append(img_name)
    return filtered, empty


def get_filtered_tiles(filtered: list, main_path: Path, subfolder: str, subtype: str):
    """
    Getting filtered tiles for given arguments.

    Args:
        filtered (list): Given filter type image list (empty or filtered).
        main_path (Path): Main folder containing images, masks, etc.
        subfolder (str): Subfolder type (images or masks).
        subtype (str): Filter type (empty or filtered).
    """
    filtered_path = main_path / str(subfolder + "_" + subtype)
    os.makedirs(filtered_path, exist_ok=True)
    for file in tqdm(filtered, desc=f"Getting {subtype} {subfolder}", ncols=150):
        src = os.path.join(main_path / subfolder, file)
        target = os.path.join(filtered_path, file)
        shutil.copy(src, target)


def filter_empty(main_path: Path):
    """
    Filtering all dataset (splitting empty and suitable).

    Args:
        main_path (Path): Main folder containing images, masks, etc.
    """
    filtered, empty = filter_tiles(main_path)
    get_filtered_tiles(filtered, main_path, "images", "filtered")
    get_filtered_tiles(filtered, main_path, "masks", "filtered")
    get_filtered_tiles(empty, main_path, "images", "empty")
    get_filtered_tiles(empty, main_path, "masks", "empty")
