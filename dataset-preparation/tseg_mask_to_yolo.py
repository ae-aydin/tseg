import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from extract_polygons import mask_to_polygon

TUMOR_COLOR_BGR = (0, 0, 200)

# Used for converting tile masks to YOLO annotations format.


def mask_to_yolo(mask_shape: tuple, polygons: list, annot_path: Path):
    """
    Convert extracted polygons to YOLO format.

    Args:
        mask_shape (tuple): Mask dimensions.
        polygons (dict): Polygon list.
        annot_path (Path): Where YOLO annotation will be saved.
    """
    height_y, width_x = mask_shape
    yolo_annot_list = list()
    for obj in polygons:
        line_list = [0]
        np_obj = np.array(obj, dtype=np.float32)
        if np_obj.shape[0] < 3:
            continue
        np_obj[:, 0] /= width_x
        np_obj[:, 1] /= height_y
        np_obj = np.round(np_obj, 7)
        line_list.extend(np_obj.flatten())
        yolo_annot_list.append(line_list)
    random.shuffle(yolo_annot_list)
    with open(annot_path, "w") as f:
        num_lines = len(yolo_annot_list)
        for i, inner_list in enumerate(yolo_annot_list):
            line = " ".join(map(str, inner_list))
            f.write(line)
            if i < num_lines - 1:
                f.write("\n")


def extract_polygons(rgb_mask_path: Path):
    """
    Extract segmented polygons from given mask.

    Args:
        rgb_mask_path (Path): Mask image path.

    Returns:
        tuple: Shape of the mask.
        list: List containing polygons.
    """
    rgb_mask = cv2.imread(str(rgb_mask_path))
    mask = np.all(rgb_mask == TUMOR_COLOR_BGR, axis=-1).astype(np.uint8) * 255
    polygons = mask_to_polygon(mask)
    return rgb_mask.shape[:-1], polygons


def visualize_annotations(annotations_path: Path, visualized_path: Path):
    """
    Convert YOLO format to mask for comparison.

    Args:
        annotations_path (Path): Path where YOLO annotations located.
        visualized_path (Path): Path where resulting mask will be saved.
    """
    for annot_txt in tqdm(
        os.listdir(annotations_path), desc="Visualizing YOLO annotations", ncols=150
    ):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        with open(os.path.join(annotations_path, annot_txt), "r") as f:
            objects = f.readlines()
            for o in objects:
                points = list(map(float, o.split()[1:]))
                points = np.array(points, np.float32).reshape((-1, 1, 2)) * 640
                cv2.fillPoly(image, np.int32([points]), color=TUMOR_COLOR_BGR)
        cv2.imwrite(os.path.join(visualized_path, Path(annot_txt).stem + ".png"), image)


def convert_to_yolo_format(main_path: Path, filtered: bool, visualize: bool):
    """
    Convert RGB masks obtained from QuPath script to YOLO format.

    Args:
        main_path (Path): Main folder containing images, masks, etc.
        filtered (bool): Whether images are filtered (different folder).
        visualize (bool): Whether to visualize YOLO annotations.
    """
    masks_folder = "masks_filtered" if filtered else "masks"
    masks_path = main_path / masks_folder
    annotations_path = main_path / "annotations"
    annotations_path.mkdir(exist_ok=True)
    for mask_name in tqdm(
        os.listdir(masks_path), desc="Converting masks to YOLO format", ncols=150
    ):
        mask_path = os.path.join(masks_path, mask_name)
        mask_basename = os.path.splitext(mask_name)[0]
        annot_path = os.path.join(annotations_path, f"{mask_basename}.txt")
        mask_shape, polygons = extract_polygons(mask_path)
        mask_to_yolo(mask_shape, polygons, annot_path)
    if visualize:
        visualized_path = main_path / "annotations_visualized"
        visualized_path.mkdir(exist_ok=True)
        visualize_annotations(annotations_path, visualized_path)
