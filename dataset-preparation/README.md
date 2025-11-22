# Dataset Preparation

1. Create tiles for all QuPath projects using the tiling scripts.

    ```plain
    tiles folder
    ├───tiled_image1 folder
    │   ├───images
    │   └───masks
    ├───tiled_image2 folder
    │   ├───images
    │   └───masks
    ├───tiled_image3 folder
    │   ├───images
    │   └───masks
    .
    .
    .
    ```

2. Run `prepare.py`.
   - For *uv*: `uv run dataset-preparation/prepare.py source target [--train-ratio RATIO] [--val-ratio RATIO] [--yolo-format]`
   - Arguments:
     - `source` (required): Path containing all tile folders.

     - `target` (required): Where the new dataset folder will be created.

     - `--train-ratio RATIO` (optional): Train set ratio. Default is 0.6.

     - `--hpa-train-only` (optional): Use all HPA TMA cores as train. Default is True.

     - `--create-dev` (optional): Create validation set. Default is True.

     - `--dev-test-ratio RATIO` (optional): Validation/test set ratio. Default is 0.5.

     - `--generate-cv` (optional): Generate cross-validation folds. Default is True.

     - `--k-folds K` (optional): Cross-validation fold count. Default is 5.

     - `--use-yolo-format` (optional): Whether to create YOLO dataset.
       - Currently not implemented.

     - `seed SEED` (optional): Seed for reproducibility. Default is -1.
