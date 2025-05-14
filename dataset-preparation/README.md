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
   - For *uv*: `uv run dataset-preparation/prepare.py tiles_path export_path [--tile-count TILE_COUNT] [--train-ratio RATIO] [--val-ratio RATIO] [--yolo-format] [--visualize]`
   - For *pip*: `py dataset-preparation/prepare.py tiles_path export_path [--tile-count TILE_COUNT] [--train-ratio RATIO] [--val-ratio RATIO] [--yolo-format] [--visualize]`
   - Arguments:
     - `tiles_path` (required): Path containing all tile folders.

     - `export_path` (required): Where the new dataset folder will be created.

     - `--tile-count TILE_COUNT` (optional): Max tile count per slide.

     - `--train-ratio RATIO` (optional): Train set ratio. Default is 0.6.

     - `--val-ratio RATIO` (optional): Validations set ratio. Default is 0.2.
       - If `train-ratio + val-ratio == 1`, then no test set created. Otherwise test set created from remaining fraction.

     - `--yolo-format` (optional): Whether to create YOLO dataset.

     - `--visualize` (optional): Whether to visualize mask to YOLO conversion.

     - All created/prepared files will be saved into `export_path`.

     - All masks will be converted to yolo format and saved into `export_path/yolo_dataset` if `--yolo-format` provided.
       - Change the `data.yaml` file accordingly (used for YOLO model training).

## Future Improvements

- Add K-fold cross-validation scheme.
