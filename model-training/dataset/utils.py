from pathlib import Path

import pandas as pd
import polars as pl


def bin_tumor_fracs(
    df: pd.DataFrame, left: float = 0.05, right: float = 0.25
) -> pd.DataFrame:
    bin_edges = [0.0, left, right, 1.0]
    bin_labels = ["small", "medium", "large"]

    return df.assign(
        tumor_bin=pd.cut(
            df["tumor_frac"],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
            right=True,
        )
    )


def read_tile_metadata(source: Path, split_file: str = "split_info.csv") -> pd.DataFrame:
    split_df = pl.read_csv(source / split_file)
    slide_df = pl.read_csv(source / "slide_info.csv")
    tiles_df = pl.read_csv(source / "tile_info.csv")

    slide_df = slide_df[["slide_name", "category", "tile_count"]]
    tiles_df = tiles_df[
        [
            "slide_name",
            "parent_dir_path",
            "relative_image_path",
            "relative_mask_path",
            "tumor_frac",
        ]
    ]

    combined_metadata_df = tiles_df.join(slide_df, on="slide_name").join(
        split_df, on="slide_name"
    )

    combined_metadata_df = combined_metadata_df.to_pandas()
    combined_metadata_df = bin_tumor_fracs(combined_metadata_df)
    return combined_metadata_df


def save(df: pd.DataFrame, directory: Path, filename: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / filename
    df.to_csv(file_path, index=False)
