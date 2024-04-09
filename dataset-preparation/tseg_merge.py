import glob
import os
import shutil
from pathlib import Path

from tqdm import tqdm


def get_all_paths(path_template: str, subfolder: str):
    """_summary_

    Args:
        path_template (str): path template used in glob (subfolder search)
        subfolder (str): QuPath tile export folder name

    Returns:
        all_paths: All found tile folder paths
    """
    folders = glob.glob(path_template + "/" + subfolder, recursive=True)
    all_paths = list()
    for folder_path in folders:
        all_paths.extend(
            [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        )
    return all_paths


def create_directories(main_dir_name: str, target_path: Path):
    """Creating main folder and subfolders

    Args:
        main_dir_name (str): Name for main folder
        target_path (Path): Where to save main folder

    Returns:
        main_folder_path: Main folder path
        main_images_path: <main-folder>/images path
        main_masks_path: <main-folder>/masks path
    """
    target_folder_name = "dataset_" + "_".join(main_dir_name.split("_")[1:])
    main_folder_path = target_path / target_folder_name
    main_images_path = main_folder_path / "images"
    main_masks_path = main_folder_path / "masks"
    main_images_path.mkdir(parents=True, exist_ok=True)
    main_masks_path.mkdir(parents=True, exist_ok=True)
    return main_folder_path, main_images_path, main_masks_path


def copy(path_list: list, target_path: Path):
    """Copy files to given target path

    Args:
        path_list (list): Filenames
        target_path (Path): Target path
    """
    for path in tqdm(
        path_list, desc=f"Accumulating {os.path.basename(target_path)}", ncols=150
    ):
        target_file_path = target_path / os.path.basename(path)
        shutil.copy(path, target_file_path)


def accumulate_all_tiles(projects_path: Path, tile_folder_name: str, target_path: Path):
    """Accumulate all QuPath project tiles at given target path

    Args:
        projects_path (Path): Folder containing all QuPath projects
        tile_folder_name (str): Tile folder name inside project directories
        target_path (Path): Where to save main folder

    Returns:
        main_paths[0]: Main folder path
    """
    path_template = str(projects_path) + "/**/" + tile_folder_name
    image_paths = get_all_paths(path_template, "images")
    masks_paths = get_all_paths(path_template, "masks")

    main_paths = create_directories(tile_folder_name, target_path)
    copy(image_paths, main_paths[1])
    copy(masks_paths, main_paths[2])
    return main_paths[0]
