from pathlib import Path

import cv2
import numpy as np
import torch
import typer
import yaml
from loguru import logger
from tqdm import tqdm
from ultralytics import YOLO


def calculate_iou_dice(true_mask: np.ndarray, pred_mask: np.ndarray):
    true_mask = true_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    true_area = true_mask.sum()
    pred_area = pred_mask.sum()

    if true_area == 0 and pred_area == 0:
        return 1.0, 1.0

    if true_area == 0 or pred_area == 0:
        return 0.0, 0.0

    intersection = float(np.logical_and(true_mask, pred_mask).sum())
    union = float(true_area + pred_area - intersection)

    iou = intersection / union if union > 0 else 0.0
    dice = (
        (2.0 * intersection) / (true_area + pred_area)
        if (true_area + pred_area) > 0
        else 0.0
    )

    return iou, dice


def get_true_mask(label_path: Path, height: float, width: float):
    true_mask = np.zeros((height, width), dtype=np.uint8)

    if not label_path.exists():
        return true_mask

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        points = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                x = float(parts[i]) * width
                y = float(parts[i + 1]) * height
                points.append([x, y])

        if len(points) >= 3:
            cv2.fillPoly(true_mask, [np.array(points, dtype=np.int32)], 1)

    return true_mask


def evaluate_model(model_type, yaml_path: str, batch_size: int, conf: float):
    if isinstance(model_type, str):
        model = YOLO(model_type)
    else:
        model = model_type

    with open(yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    val_images_path = Path(data_cfg.get("path", ".")) / data_cfg.get("val", "")
    val_labels_path = Path(str(val_images_path).replace("images", "labels"))

    if not val_labels_path.exists():
        logger.error(f"Labels directory not found: {val_labels_path}")
        return 0, 0, 0

    image_files = list(val_images_path.glob("*.jpg")) + list(
        val_images_path.glob("*.png")
    )
    logger.info(f"Found {len(image_files)} image files.")

    total_iou = 0.0
    total_dice = 0.0
    count = 0

    for image_path in tqdm(image_files, desc="Calculating metrics", ncols=100):
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]

        label_path = val_labels_path / f"{image_path.stem}.txt"

        true_mask = get_true_mask(label_path, h, w)
        results = model(image, conf=conf, verbose=False)

        if not hasattr(results[0], "masks") or results[0].masks is None:
            count += 1
            total_iou += 0
            total_dice += 0
            continue

        pred_mask = np.zeros((h, w), dtype=bool)

        for mask in results[0].masks.data:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            if mask.shape[:2] != (h, w):
                mask_resized = np.zeros((h, w), dtype=np.uint8)
                cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    dst=mask_resized,
                    interpolation=cv2.INTER_NEAREST,
                )
                mask = mask_resized

            pred_mask = np.logical_or(pred_mask, mask > 0)

        iou, dice = calculate_iou_dice(true_mask, pred_mask)
        total_iou += iou
        total_dice += dice
        count += 1

    logger.info("Validation on ultralytics framework to calculate more metrics.")
    val_results = model.val(
        data="data.yaml", imgsz=640, batch=batch_size, conf=conf, project="tseg", plots=True
    )

    logger.info(f"Extra metrics On {count} images")

    p = val_results.seg.p[0]
    logger.info(f"Precision: {p:.4f}")

    r = val_results.seg.r[0]
    logger.info(f"Recall: {r:.4f}")

    f1 = val_results.seg.f1[0]
    logger.info(f"F1 Score: {f1:.4f}")

    fnr = 1 - r
    logger.info(f"False Negative Rate: {fnr:.4f}")

    avg_iou = total_iou / count
    logger.info(f"IoU: {avg_iou:.4f}")

    avg_dice = total_dice / count
    logger.info(f"Dice Score: {avg_dice:.4f}")


def main(model_path: str, yaml_path: str, batch_size: int = 24, conf: float = 0.001):
    evaluate_model(
        model_type=model_path, yaml_path=yaml_path, batch_size=batch_size, conf=conf
    )


if __name__ == "__main__":
    typer.run(main)
