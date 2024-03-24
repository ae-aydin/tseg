import argparse

import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def mask2yolo(mask, annotations: dict, annot_path: Path):
    height_y, width_x = mask.shape[:-1]
    yolo_annot_list = list()
    for obj in annotations:
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


def extract(mask_path: Path, annot_path: Path, bg: int):
    mask = cv2.imread(mask_path)
    class_polygons = list()
    class_mask = np.all(mask == (0, 0, 200), axis=-1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        polygon = contour.squeeze().tolist()
        class_polygons.append(polygon)
    mask2yolo(mask, class_polygons, annot_path)


def loop_folder(args: argparse.Namespace):
    mask_list = os.listdir(args.spath)
    for mask_filename in tqdm(mask_list):
        mask_path = os.path.join(args.spath, mask_filename)
        mask_basename = os.path.splitext(mask_filename)[0]
        annot_path = os.path.join(args.tpath, f"{mask_basename}.txt")
        extract(mask_path, annot_path, args.bg)


def main(args: argparse.Namespace):
    loop_folder(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spath", type=Path, help="Source mask path")
    parser.add_argument("tpath", type=Path, help="Target annotation path")
    parser.add_argument("--bg", type=int, default=0, help="background color")
    args = parser.parse_args()
    main(args)
