from pathlib import Path

from loguru import logger

from tseg import data_ops


def create_dataset_folders(target: Path):
    logger.info("Creating dataset folders")
    dirs = {
        "parent": target,
        "train": target / "train",
        "val": target / "val",
        "test": target / "test",
        "metadata": target / "metadata",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def train_test_split(
    source: Path,
    target: Path,
    train_ratio: float,
    val_ratio: float,
    use_yolo_format: bool,
):
    """
    Train-test split the data WSI-level, and create a dataset in YOLO format.

    Args:
        source (Path): Path containing all tile folders.
        target (Path): Path where dataset folder will be created.
        train_ratio (float): Train set ratio.
        val_ratio (float): Validation set ratio.
        use_yolo_format (bool): Include YOLO compatible format.
    """
    dirs = create_dataset_folders(target)
    split_info_path = data_ops.split_tiles(
        source, train_ratio, val_ratio, dirs["metadata"]
    )
    data_ops.construct_dataset(source, split_info_path, dirs)
    if use_yolo_format:
        raise NotImplementedError
