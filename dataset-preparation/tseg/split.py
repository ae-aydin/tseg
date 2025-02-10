import shutil
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from tseg.yolo_formatter import convert_to_yolo_format

DATA_CATEGORIES = ["wsi_tiled", "img_tiled"]


def _fix_mask(mask_path: Path, target_path: Path, binarify_mask: bool = False, kernel_size: int = 5):
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    grayscale_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if binarify_mask:
        grayscale_mask = (grayscale_mask > 0).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    binary_mask = cv2.morphologyEx(grayscale_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(str(target_path / mask_path.name), binary_mask)
    

def _count_data(tiles_path: Path) -> dict:
    """
    Categorizes the tiles according to WSI type.

    Args:
        tiles_path (Path): Path containing all tile folders.

    Returns:
        dict: Categorized WSI tiles.
    """
    tile_folders_categorized = {
        category: list(
            filter(
                lambda x: str(x.stem).split("|")[0] == category, tiles_path.iterdir()
            )
        )
        for category in DATA_CATEGORIES
    }
    return tile_folders_categorized


def _copy_tiles(
    tiles_path: Path,
    subset_tile_folders: list,
    subset_images_path: Path,
    subset_masks_path: Path,
):
    """
    Helper function to copy tiles.

    Args:
        tiles_path (Path): Path containing all tile folders.
        subset_tile_folders (list): Train or test tile folder list.
        subset_images_path (Path): train/images or test/images path.
        subset_masks_path (Path): train/masks or test/masks path.
    """
    for folder in tqdm(subset_tile_folders, desc=f"Copying {subset_images_path.parent.name} images", ncols=100):
        source_images = tiles_path / folder / "images"
        source_masks = tiles_path / folder / "masks"

        for file in source_images.iterdir():
            shutil.copy(file, subset_images_path)

        for file in source_masks.iterdir():
            _fix_mask(file, subset_masks_path)


def _accumulate_tiles(
    tiles_path: Path,
    export_folder_path: Path,
    train_tile_folders: list,
    test_tile_folders: list,
):
    """
    Create the dataset folder and copy the files according to the train-test split.

    Args:
        tiles_path (Path): Path containing all tile folders.
        export_folder_path (Path): Path to export (dataset) folder.
        train_tile_folders (list): Tile folder names selected for train set.
        test_tile_folders (list): Tile folder names selected for test set.
    """
    # Define dataset folder structure
    train_images_path = export_folder_path / "train" / "images"
    train_masks_path = export_folder_path / "train" / "masks"
    test_images_path = export_folder_path / "test" / "images"
    test_masks_path = export_folder_path / "test" / "masks"

    # Create the folder structure
    for path in [
        train_images_path,
        train_masks_path,
        test_images_path,
        test_masks_path,
    ]:
        path.mkdir(parents=True, exist_ok=True)
        
    logger.info("Copying tile images according to subset.")

    _copy_tiles(tiles_path, train_tile_folders, train_images_path, train_masks_path)
    _copy_tiles(tiles_path, test_tile_folders, test_images_path, test_masks_path)


def _prepare_yolo_dataset(export_folder_path: Path):
    """
    Prepare the YOLO dataset according to the format.

    Args:
        export_folder_path (Path): Path to export (dataset) folder.
    """
    yolo_dataset_path = export_folder_path / "yolo-dataset"
    train_path = export_folder_path / "train"
    test_path = export_folder_path / "test"

    yolo_dataset_path.mkdir(parents=True, exist_ok=True)

    img_train = yolo_dataset_path / "images" / "train"
    img_val = yolo_dataset_path / "images" / "val"
    label_train = yolo_dataset_path / "labels" / "train"
    label_val = yolo_dataset_path / "labels" / "val"

    img_train.mkdir(parents=True, exist_ok=True)
    img_val.mkdir(parents=True, exist_ok=True)
    label_train.mkdir(parents=True, exist_ok=True)
    label_val.mkdir(parents=True, exist_ok=True)
    
    logger.info("Arranging dataset in YOLO format.")

    for file in tqdm(list((train_path / "images").iterdir()), desc=f"Copying files to {img_train}", ncols=150):
        shutil.copy(file, img_train)

    for file in tqdm(list((test_path / "images").iterdir()), desc=f"Copying files to {img_val}", ncols=150):
        shutil.copy(file, img_val)

    for file in tqdm(list((train_path / "annotations").iterdir()), desc=f"Copying files to {label_train}", ncols=150):
        shutil.copy(file, label_train)

    for file in tqdm(list((test_path / "annotations").iterdir()), desc=f"Copying files to {label_val}", ncols=150):
        shutil.copy(file, label_val)


def train_test_split(
    tiles_path: Path, export_path: Path, ratio: float, visualize: bool
):
    """
    Train-test split the data WSI-level, and create a dataset in YOLO format.

    Args:
        tiles_path (Path): Path containing all tile folders.
        export_path (Path): Path where dataset folder will be created.
        ratio (float): Train-test split ratio.
        visualize (bool): Whether to visualize YOLO format to check consistency.
    """
    tile_folders_categorized = _count_data(Path(tiles_path))
    train_tile_folders, test_tile_folders = [], []
    for tile_folders in tile_folders_categorized.values():
        temp_n_train = int(len(tile_folders) * ratio)
        np.random.shuffle(tile_folders)
        train_tile_folders.extend(tile_folders[:temp_n_train])
        test_tile_folders.extend(tile_folders[temp_n_train:])
    export_folder_path = export_path / "dataset"
    logger.info(f"Train tiles ({len(train_tile_folders)} total)\n\t{'\n\t'.join(map(lambda x: str(x.name), train_tile_folders))}")
    logger.info(f"Test tiles ({len(test_tile_folders)} total)\n\t{'\n\t'.join(map(lambda x: str(x.name), test_tile_folders))}")
    _accumulate_tiles(
        tiles_path, export_folder_path, train_tile_folders, test_tile_folders
    )
    convert_to_yolo_format(export_folder_path / "train", visualize)
    convert_to_yolo_format(export_folder_path / "test", visualize)
    _prepare_yolo_dataset(export_folder_path)
