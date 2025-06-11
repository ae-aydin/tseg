from pathlib import Path

import albumentations as A

from .base import BaseDataset


class SlideTileDataset(BaseDataset):
    def __init__(
        self,
        source: Path,
        df,
        category: str | None = None,
        min_tumor_frac: float = 0.01,
        transform: A.Compose | None = None,
        img_size: int = 512,
    ):
        if category is not None:
            filtered_df = (
                df[(df["category"] == category) & (df["tumor_frac"] >= min_tumor_frac)]
                .copy()
                .reset_index(drop=True)
            )
        else:
            filtered_df = (
                df[df["tumor_frac"] >= min_tumor_frac].copy().reset_index(drop=True)
            )

        super().__init__(source, filtered_df, transform, img_size)
        self.category = category
        self.min_tumor_frac = min_tumor_frac
        self.slide_counts = self.df.groupby("slide_name").size()

    def get_min_slide_tile_count(self) -> int:
        return self.slide_counts.min()

    def print_tiles_per_slide(self):
        for slide_name, count in self.slide_counts.items():
            print(f"Slide: {slide_name}, Tiles remaining: {count}")
        print()
