import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm

# Used for creating dataset according to YOLO dataset structure.


def copy(filelist: list, source_path: Path, target_path: Path, ext: str):
    """
    Copy file from source to target.

    Args:
        filelist (list): Filenames (stems, same for image and annotation).
        source_path (Path): Source directory path.
        target_path (Path): Target directory path.
        ext (str): File extension.
    """
    description = "/".join(str(target_path).split(os.path.sep)[-3:])
    for file in tqdm(filelist, desc=f"Copying files for {description}", ncols=150):
        tpath = target_path / str(file + ext)
        spath = source_path / str(file + ext)
        shutil.copy(spath, tpath)


def prepare_yolo_dataset(main_path: Path, filtered: bool, ratio: float):
    """
    Preparing files according to YOLO dataset structure.

    Args:
        main_path (Path): Main folder containing images, masks, etc.
        filtered (bool): Whether images are filtered (different folder).
        ratio (float): Train-Test split ratio.
    """
    image_folder = "images_filtered" if filtered else "images"
    images_path = main_path / image_folder
    annotations_path = main_path / "annotations"
    yolo_dataset_path = main_path / "yolo_dataset"
    yolo_dataset_path.mkdir(exist_ok=True)

    stems = [Path(p).stem for p in os.listdir(images_path)]
    n_train = int(len(stems) * ratio)
    random.shuffle(stems)
    train_filelist = stems[:n_train]
    val_filelist = stems[n_train:]
    im_train = yolo_dataset_path / "images" / "train"
    im_val = yolo_dataset_path / "images" / "val"
    label_train = yolo_dataset_path / "labels" / "train"
    label_val = yolo_dataset_path / "labels" / "val"
    im_train.mkdir(parents=True, exist_ok=True)
    im_val.mkdir(parents=True, exist_ok=True)
    label_train.mkdir(parents=True, exist_ok=True)
    label_val.mkdir(parents=True, exist_ok=True)
    copy(train_filelist, images_path, im_train, ".png")
    copy(val_filelist, images_path, im_val, ".png")
    copy(train_filelist, annotations_path, label_train, ".txt")
    copy(val_filelist, annotations_path, label_val, ".txt")
