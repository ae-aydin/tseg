import shutil
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from tseg.convert_yolo import convert_to_yolo_format
from tseg.dataset_yolo import prepare_yolo_dataset

DATA_CATEGORIES = ["wsi_tiled", "img_tiled"]


def _fix_mask(
    mask_path: Path,
    target_path: Path,
    kernel_size: int = 5,
):
    """
    Fix mask imperfections with morphological operations.

    Args:
        mask_path (Path): Path to read the mask from.
        target_path (Path): Path to save the mask to.
        binarify_mask (bool, optional): Whether to convert pixel value 255 to 1. Defaults to False.
        kernel_size (int, optional): Kernel size for morphological operations. Defaults to 3.
    """
    mask = cv2.imread(str(mask_path))
    grayscale_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Fixing annotation glitches
    binary_mask = cv2.morphologyEx(grayscale_mask, cv2.MORPH_CLOSE, kernel)

    # Slightly enlarging the masks
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
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
    tile_count: int,
):
    """
    Helper function to copy tiles.

    Args:
        tiles_path (Path): Path containing all tile folders.
        subset_tile_folders (list): Train or test tile folder list.
        subset_images_path (Path): train/images, val/images, or test/images path.
        subset_masks_path (Path): train/masks, val/masks, or test/masks path.
        tile_count (int): Maximum number of tiles to copy.
    """
    for folder in tqdm(
        subset_tile_folders,
        desc=f"Copying {subset_images_path.parent.name} images",
        ncols=100,
    ):
        source_images = tiles_path / folder / "images"
        source_masks = tiles_path / folder / "masks"

        source_images_list = list(source_images.iterdir())

        # Apply tile count limit only to "wsi_tiled" folders if specified
        if tile_count != -1 and source_images.parent.name.split("|")[0] == "wsi_tiled":
            tiles_per_wsi = min(tile_count, len(source_images_list))
            np.random.shuffle(source_images_list)
            source_files = source_images_list[:tiles_per_wsi]
        else:
            # Copy all tiles for non-wsi_tiled folders
            source_files = source_images_list

        for file in source_files:
            shutil.copy(file, subset_images_path)

            mask_path = source_masks / file.name
            _fix_mask(mask_path, subset_masks_path)


def _accumulate_tiles(
    tiles_path: Path,
    export_path: Path,
    train_tile_folders: list,
    val_tile_folders: list,
    test_tile_folders: list,
    tile_count: int,
):
    """
    Create the dataset folder and copy the files according to the train-test split.

    Args:
        tiles_path (Path): Path containing all tile folders.
        export_path (Path): Path to export (dataset) folder.
        train_tile_folders (list): Tile folder names selected for train set.
        val_tile_folders (list): Tile folder names selected for validation set.
        test_tile_folders (list): Tile folder names selected for test set.
        tile_count (int): Maximum tile count in a wsi.
    """
    # Define dataset folder structure
    train_images_path = export_path / "train" / "images"
    train_masks_path = export_path / "train" / "masks"
    val_images_path = export_path / "val" / "images"
    val_masks_path = export_path / "val" / "masks"
    test_images_path = export_path / "test" / "images"
    test_masks_path = export_path / "test" / "masks"

    # Create the folder structure
    for path in [
        train_images_path,
        train_masks_path,
        val_images_path,
        val_masks_path,
        test_images_path,
        test_masks_path,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    logger.info("Copying tile images according to subset.")

    _copy_tiles(
        tiles_path, train_tile_folders, train_images_path, train_masks_path, tile_count
    )
    _copy_tiles(
        tiles_path, val_tile_folders, val_images_path, val_masks_path, tile_count
    )
    _copy_tiles(
        tiles_path, test_tile_folders, test_images_path, test_masks_path, tile_count
    )


def train_test_split(
    tiles_path: Path,
    export_path: Path,
    tile_count: int,
    train_ratio: float,
    val_ratio: float,
    yolo_format: bool,
    visualize: bool,
):
    """
    Train-test split the data WSI-level, and create a dataset in YOLO format.

    Args:
        tiles_path (Path): Path containing all tile folders.
        export_path (Path): Path where dataset folder will be created.
        tile_count (float): Maximum tile count in a wsi.
        train_ratio (float): Train set ratio.
        val_ratio (float): Validation set ratio.
        yolo_format (bool): Create YOLO format.
        visualize (bool): Whether to visualize YOLO format to check consistency.
    """
    # If sum of train ratio and test ratio close to 1, dont use test set
    create_test_set = False if np.isclose(train_ratio + val_ratio, 1) else True
    tile_folders_categorized = _count_data(tiles_path)
    train_tiles, val_tiles, test_tiles = [], [], []
    for tile_folders in tile_folders_categorized.values():
        n_train = int(round(len(tile_folders) * train_ratio))
        n_val = int(round(len(tile_folders) * val_ratio))
        np.random.shuffle(tile_folders)
        train_tiles.extend(tile_folders[:n_train])
        if create_test_set:
            val_tiles.extend(tile_folders[n_train : n_train + n_val])
            test_tiles.extend(tile_folders[n_train + n_val :])
        else:
            val_tiles.extend(tile_folders[n_train:])

    logger.info(
        f"Train tiles ({len(train_tiles)} total)\n\t{'\n\t'.join(map(lambda x: str(x.name), train_tiles))}"
    )
    logger.info(
        f"Validation tiles ({len(val_tiles)} total)\n\t{'\n\t'.join(map(lambda x: str(x.name), val_tiles))}"
    )
    logger.info(
        f"Test tiles ({len(test_tiles)} total)\n\t{'\n\t'.join(map(lambda x: str(x.name), test_tiles))}"
    )

    _accumulate_tiles(
        tiles_path, export_path, train_tiles, val_tiles, test_tiles, tile_count
    )

    if yolo_format:
        convert_to_yolo_format(export_path / "train", visualize)
        convert_to_yolo_format(export_path / "val", visualize)
        convert_to_yolo_format(export_path / "test", visualize)
        prepare_yolo_dataset(export_path)
