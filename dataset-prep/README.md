# Dataset Preparation

- Accumulate all tiles and all masks in seperate folders.

- Install python requirements: `py -m pip install -r requirements.txt`

- Run `mask2yolo.py`.
  
  - `py mask2yolo.py <masks-folder-path> <target-annotations-folder-path> --<background-color>`
    
  - All masks will be converted to yolo format and saved into `target-annotations-folder-path`.
    
- Run `split.py` to prepare final dataset.
  
  - Create an empty folder for dataset.
    
  -    `py split.py <images-folder-path> <annotations-folder-path> <created-folder-path> --<train-val-split>`
    
- Dataset suitable for YOLO will be created at `created-folder-path`.

- Change the `data.yaml` file accordingly (required for model training).
