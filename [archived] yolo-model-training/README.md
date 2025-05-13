# YOLO-based Model Training for Tumor Segmentation

> This directory was used for initial iterations of model training. It is no longer in use but has been kept in the repository.

## Setup

- Move prepared `data.yaml` file to this folder.
  
- Train `YOLO<model_suffix>-seg`.
  
  - Run `py model-training/train.py <model_suffix>` or `py model-training/aug_train.py <model_suffix>`.

  - For *uv*, `uv run model-training/train.py <model_suffix>`.
