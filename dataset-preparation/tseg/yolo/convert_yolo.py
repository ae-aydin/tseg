import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .polygon_ops import mask_to_polygon

TUMOR_COLOR_BGR = (255, 255, 255)

# Used for converting binary tile mask to YOLO annotation format.


def mask_to_yolo(mask_shape: tuple, polygons: list, label_path: Path):
    """
    Convert extracted polygons to YOLO format.

    Args:
        mask_shape (tuple): Mask dimensions.
        polygons (dict): Polygon list.
        label_path (Path): Where YOLO annotation will be saved.
    """
    height_y, width_x = mask_shape
    yolo_label_list = list()
    for obj in polygons:
        line_list = [0]
        np_obj = np.array(obj, dtype=np.float32)
        if np_obj.shape[0] < 3:
            continue
        np_obj[:, 0] /= width_x
        np_obj[:, 1] /= height_y
        np_obj = np.round(np_obj, 7)
        line_list.extend(np_obj.flatten())
        yolo_label_list.append(line_list)
    random.shuffle(yolo_label_list)
    with open(label_path, "w") as f:
        num_lines = len(yolo_label_list)
        for i, inner_list in enumerate(yolo_label_list):
            line = " ".join(map(str, inner_list))
            f.write(line)
            if i < num_lines - 1:
                f.write("\n")


def extract_polygons(mask_path: Path):
    """
    Extract segmented polygons from given mask.

    Args:
        mask_path (Path): RGB mask image path.

    Returns:
        tuple: Shape of the mask.
        list: List containing polygons.
    """
    rgb_mask = cv2.imread(str(mask_path))
    mask = np.all(rgb_mask == TUMOR_COLOR_BGR, axis=-1).astype(np.uint8) * 255
    polygons = mask_to_polygon(mask)
    return rgb_mask.shape[:-1], polygons


def visualize_annotations(source: Path, target: Path, imgsz: int = 640):
    """
    Convert YOLO formats to image masks for comparison.

    Args:
        source (Path): Path where YOLO labels located.
        target (Path): Path where resulting mask will be saved.
        imgsz (int): Image size.
    """
    for label_path in tqdm(
        list(source.iterdir()), desc="Visualizing YOLO labels", ncols=100
    ):
        image = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        with open(label_path, "r") as f:
            objects = f.readlines()
            for o in objects:
                points = list(map(float, o.split()[1:]))
                points = np.array(points, dtype=np.float32).reshape((-1, 1, 2)) * imgsz
                cv2.fillPoly(
                    image, [np.array(points, dtype=np.int32)], color=TUMOR_COLOR_BGR
                )
        visualized_filename = label_path.stem + ".png"
        cv2.imwrite(str(target / visualized_filename), image)


def convert_to_yolo_format(source: Path, target: Path, visualize: bool):
    """
    Convert RGB masks obtained from QuPath script to YOLO format.

    Args:
        subset_path (Path): Subset (train, val, test) folder containing images, masks, etc.
        visualize (bool): Whether to visualize YOLO annotations.
    """
    for mask_path in tqdm(
        list(source.iterdir()),
        desc=f"Converting {source} to YOLO format",
        ncols=100,
    ):
        annot_path = target / f"{mask_path.stem}.txt"
        mask_shape, polygons = extract_polygons(mask_path)
        mask_to_yolo(mask_shape, polygons, annot_path)

    if visualize:
        visualization_path = target.parent.parent / "visualization"
        visualization_path.mkdir(exist_ok=True)
        visualize_annotations(target, visualization_path)
