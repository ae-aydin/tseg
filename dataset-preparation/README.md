# Dataset Preparation

- Create tiles for all QuPath projects using the tiling scripts.

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

- Run `main.py`.

- `uv run dataset-preparation/main.py tiles_path export_path [--visualize] [--ratio RATIO]` for *uv*, `py dataset-preparation/main.py tiles_path export_path [--visualize] [--ratio RATIO]` for default.

  - `tiles_path` (required): Path containing all tile folders.

  - `export_path` (required): Where the new dataset folder will be created.

  - `--ratio RATIO` (optional): Train-test split ratio. Default is 0.85.

  - `--visualize` (optional): Whether to visualize mask to YOLO conversion.

  - All created/prepared files will be saved into `export_path/dataset`.

  - All masks will be converted to yolo format and saved into `export_path/dataset/yolo_dataset`.

- Change the `data.yaml` file accordingly (used for YOLO model training).

## Future Improvements

- Add K-fold cross-validation scheme.

- Improve performance.

- Improve code readability.

- Improve logging.
