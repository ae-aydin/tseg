# YOLO-based Model Training for Tumor Segmentation

> This directory was used for initial iterations of model training. It is no longer in use but has been kept in the repository.

## Setup

- Update `yolo-dataset.yaml` file accordingly.
  
- Train `YOLO<model_suffix>-seg`.

  - For *uv*, `uv run model-training/train.py <model_suffix> <task-name> <--from-scratch>`.

## Requirements

> Check requirements.txt

- torch
- ultralytics
  - albumentations
- opencv-python
- loguru
- typer
- wandb
