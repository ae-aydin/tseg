import shutil
from pathlib import Path

from loguru import logger
from tqdm import tqdm

# Used for preparing dataset in YOLO compatible format.


def _create_yolo_dataset_structure(yolo_dataset_path: Path):
    """Create folders using YOLO dataset format.

    Args:
        yolo_dataset_path (Path): Path to YOLO dataset.
    """
    yolo_dataset_path.mkdir(parents=True, exist_ok=True)

    img_train = yolo_dataset_path / "images" / "train"
    img_val = yolo_dataset_path / "images" / "val"
    img_test = yolo_dataset_path / "images" / "test"
    label_train = yolo_dataset_path / "labels" / "train"
    label_val = yolo_dataset_path / "labels" / "val"
    label_test = yolo_dataset_path / "labels" / "test"

    img_train.mkdir(parents=True, exist_ok=True)
    img_val.mkdir(parents=True, exist_ok=True)
    img_test.mkdir(parents=True, exist_ok=True)
    label_train.mkdir(parents=True, exist_ok=True)
    label_val.mkdir(parents=True, exist_ok=True)
    label_test.mkdir(parents=True, exist_ok=True)

    return [img_train, img_val, img_test, label_train, label_val, label_test]


def copy_to_yolo_folder(source_path: Path, target_path: Path):
    """Copy files to target YOLO subfolder.

    Args:
        source_path (Path): Source path for target subfolder.
        target_path (Path): YOLO subfolder path.
    """
    for file in tqdm(
        list(source_path.iterdir()),
        desc=f"Copying files to {target_path}",
        ncols=150,
    ):
        shutil.copy(file, target_path)


def prepare_yolo_dataset(export_path: Path):
    """
    Prepare the YOLO dataset according to the format.

    Args:
        export_path (Path): Path to export (dataset) folder.
    """
    yolo_dataset_path = export_path / "yolo-dataset"

    source_paths = [
        export_path / "train" / "images",
        export_path / "val" / "images",
        export_path / "test" / "images",
        export_path / "train" / "annotations",
        export_path / "val" / "annotations",
        export_path / "test" / "annotations",
    ]

    target_subfolder_paths = _create_yolo_dataset_structure(yolo_dataset_path)
    correspondance_table = dict(zip(source_paths, target_subfolder_paths))

    logger.info("Arranging dataset in YOLO format.")

    for source_path, target_path in correspondance_table.items():
        copy_to_yolo_folder(source_path, target_path)
