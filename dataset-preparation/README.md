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

     - `--val-ratio RATIO` (optional): Validations set ratio. Default is 0.2.
       - If `train-ratio + val-ratio == 1`, then no test set created. Otherwise test set created from remaining fraction.

     - `--yolo-format` (optional): Whether to create YOLO dataset.
       - All generated YOLO annotations are automatically visualized for sanity check.

     - All created/prepared files will be saved into `target`.

     - All masks will be converted to yolo format and saved into `target/yolo_dataset` if `--yolo-format` provided.
       - Change the `[archived]-yolo-model-training/yolo_dataset.yaml` file accordingly (used for YOLO model training).

## Future Improvements

- Add K-fold cross-validation scheme.
