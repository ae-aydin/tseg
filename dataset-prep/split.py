import argparse

import os
import random
from pathlib import Path
import shutil


def move(filelist: list, source_path: Path, target_path: Path, ext: str):
    for file in filelist:
        tpath = target_path / str(file + ext)
        spath = source_path / str(file + ext)
        shutil.copy(spath, tpath)


def main(args: argparse.Namespace):
    filenames = [Path(p).stem for p in os.listdir(args.impath)]
    n_train = int(len(filenames) * args.ratio)
    random.shuffle(filenames)
    train_filelist = filenames[:n_train]
    val_filelist = filenames[n_train:]
    im_train = args.tpath / "images" / "train"
    im_val = args.tpath / "images" / "val"
    label_train = args.tpath / "labels" / "train"
    label_val = args.tpath / "labels" / "val"
    im_train.mkdir(parents=True, exist_ok=True)
    im_val.mkdir(parents=True, exist_ok=True)
    label_train.mkdir(parents=True, exist_ok=True)
    label_val.mkdir(parents=True, exist_ok=True)
    move(train_filelist, args.impath, im_train, ".jpg")
    move(val_filelist, args.impath, im_val, ".jpg")
    move(train_filelist, args.anpath, label_train, ".txt")
    move(val_filelist, args.anpath, label_val, ".txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("impath", type=Path, help="Source image path")
    parser.add_argument("anpath", type=Path, help="Source annotation path")
    parser.add_argument("tpath", type=Path, help="Target path")
    parser.add_argument("--ratio", type=float, default=0.85, help="train split ratio")
    args = parser.parse_args()
    main(args)
