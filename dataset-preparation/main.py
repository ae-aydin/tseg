import argparse
import logging
from pathlib import Path

from tseg_filter import filter_empty
from tseg_mask_to_yolo import convert_to_yolo_format
from tseg_merge import accumulate_all_tiles
from tseg_split import prepare_yolo_dataset

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def main(args: argparse.Namespace):
    logger.info(f"Accumulating all tiles at {args.target}")
    main_path = accumulate_all_tiles(args.src, args.tiles, args.target)
    if args.filter:
        logger.info("Filtering empty tiles, creating new directory for filtered ones")
        filter_empty(main_path)
    convert_log = " + visualizing annotations for comparison" if args.visualize else ""
    logger.info("Converting RGB masks to YOLO format annotations" + convert_log)
    convert_to_yolo_format(main_path, args.filter, args.visualize)
    logger.info("Preparing dataset (train-test split) for YOLOv8 training")
    prepare_yolo_dataset(main_path, args.filter, args.ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src", type=Path, help="Path containing all QuPath project folders"
    )
    parser.add_argument(
        "tiles", type=str, help="Tile folders' name inside QuPath projects"
    )
    parser.add_argument(
        "target", type=Path, help="Where new dataset folder will be created"
    )
    parser.add_argument(
        "--filter", action="store_true", help="Filtering background images"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualization of mask to yolo conversion",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.85, help="Train-test split ratio"
    )
    args = parser.parse_args()
    main(args)
