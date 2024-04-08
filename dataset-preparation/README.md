# Dataset Preparation

- Prepare a folder containing all QuPath projects.
  
  ```
  qupath_projects folder
  ├───project1 folder
  │   └───tile_folder (created with QuPath script)
  │       ├───images
  │       └───masks
  ├───project2 folder
  │   └───tile_folder (created with QuPath script)
  │       ├───images
  │       └───masks
  ├───project3 folder
  │   └───tile_folder (created with QuPath script)
  │       ├───images
  │       └───masks
  ```
  

- Install python requirements: `py -m pip install -r requirements.txt`

- Run `main.py`.

- `py main.py src tiles target [--filter] [--visualize] [--ratio RATIO]`

  - `src` (required): Path containing all QuPath project folders.
  
  - `tiles` (required): Tile folders' name inside QuPath projects.
  
  - `target` (required): Where the new dataset folder will be created.
  
  - `--filter` (optional): Whether to filter background images.
  
  - `--visualize` (optional): Whether to visualize mask to YOLO conversion.
  
  - `--ratio RATIO` (optional): Train-test split ratio. Default is 0.85.

  - All created/prepared files will be saved into `target/tiles`.
  
  - All masks will be converted to yolo format and saved into `target/tiles/yolo_dataset`.

- Change the `data.yaml` file accordingly (used for YOLO model training).
