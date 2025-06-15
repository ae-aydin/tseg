from pathlib import Path

import polars as pl
import yaml

CONFIG_PATH = Path("./dataset-preparation/config.yaml")


def load_yaml(source: Path = CONFIG_PATH) -> dict:
    with open(source, "r") as f:
        data = yaml.safe_load(f)
    return data


def save_yaml(data: dict, target: Path, filename: str):
    with open(target / filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def save_csv(entries: list, target: Path) -> None:
    df = pl.DataFrame(entries)
    df.write_csv(target)


def add_suffix_to_dir_items(source: Path, suffix="label") -> None:
    for item in source.iterdir():
        if item.is_file():
            new_name = f"{item.stem}_{suffix}{item.suffix}"
            item.rename(source / new_name)


def pad_str(s: str, max_len: int = 15, pad: str = "."):
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."

    n_space = max_len - len(s)
    return f"{pad * n_space}{s}"
