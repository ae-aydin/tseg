from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import polars as pl
import torch
from PIL import Image
from torch.utils.data import Dataset


class TileDataset(Dataset):
    def __init__(
        self,
        source: Path,
        df: pd.DataFrame,
        transform: A.Compose | None = None,
        img_size: int = 512,
        domain_weight: float | None = None,
        gamma: float = 0.2,
        eps: float = 1.0,
    ):
        self.source = source
        self.df = df.reset_index(drop=True)

        self.img_size = img_size
        self.area = img_size * img_size
        self.transform = transform

        counts = self.df["category"].value_counts()
        if domain_weight is None:
            domain_weight = counts["wsi_tiled"] / counts["img_tiled"]
        domain_weight = float(domain_weight or 1.0)

        self.df["domain_weight"] = self.df["category"].map(
            lambda x: domain_weight if x == "img_tiled" else 1.0
        )

        tumor_pixels = self.df["tumor_frac"] * self.area
        self.df["tumor_weight"] = (tumor_pixels + eps) ** (-gamma)

        tumor_bin_weights = {"small": 0.25, "medium": 0.5, "large": 0.25}
        self.df["tumor_bin_weight"] = self.df["tumor_bin"].map(tumor_bin_weights)

        self.df["weight"] = (
            self.df["domain_weight"].astype(float)
            * self.df["tumor_weight"].astype(float)
            * self.df["tumor_bin_weight"].astype(float)
        )
        self.df["weight"] = self.df["weight"] / self.df["weight"].sum()

        self.weights = torch.DoubleTensor(self.df["weight"].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.source / row["parent_dir_path"] / row["relative_image_path"]
        mask_path = self.source / row["parent_dir_path"] / row["relative_mask_path"]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            mask = mask.float().div(255).unsqueeze(0)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "slide_name": row["slide_name"],
            "category": row["category"],
            "tumor_frac": row["tumor_frac"],
            "tumor_bin": row["tumor_bin"],
            "idx": idx,
        }


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


def read_tile_metadata(source: Path) -> pd.DataFrame:
    split_df = pl.read_csv(source / "split_info.csv")
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


def get_train_augmentations(img_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            # Geometric transforms - more aggressive
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                shift_limit=0.2, scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.7
            ),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.OpticalDistortion(distort_limit=0.3, p=0.5),
            # Color and staining variations - more aggressive
            A.OneOf(
                [
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0,
                    ),
                    A.RGBShift(
                        r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0
                    ),
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0
                    ),
                ],
                p=0.8,
            ),
            # Quality variations - more aggressive
            A.OneOf(
                [
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, p=1.0),
                ],
                p=0.5,
            ),
            # Tissue artifacts and variations
            A.OneOf(
                [
                    A.CoarseDropout(
                        num_holes_range=(4, 8),
                        hole_height_range=(16, 32),
                        hole_width_range=(16, 32),
                        p=1.0,
                    ),
                    A.GridDropout(
                        ratio=0.3,
                        unit_size_range=(32, 64),
                        holes_number_xy=(4, 4),
                        shift_xy=(0, 0),
                        random_offset=True,
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),
            # Additional histopathology-specific augmentations
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                ],
                p=0.3,
            ),
            # Normalize and convert to tensor
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2(),
        ],
        p=1.0,
    )


def get_val_augmentations(img_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2(),
        ]
    )


def get_sampler_stats(dataset: TileDataset) -> dict:
    df = dataset.df

    domain_stats = (
        df.groupby("category")
        .agg(
            {
                "weight": ["sum", "mean", "std"],
                "slide_name": "nunique",
            }
        )
        .round(4)
    )
    domain_stats[("weight", "count")] = df.groupby("category").size()

    tumor_stats = (
        df.groupby("tumor_bin")
        .agg(
            {
                "weight": ["sum", "mean", "std"],
            }
        )
        .round(4)
    )
    tumor_stats[("weight", "count")] = df.groupby("tumor_bin").size()

    return {
        "domain_stats": domain_stats,
        "tumor_stats": tumor_stats,
        "total_samples": len(dataset),
        "weight_sum": df["weight"].sum(),
        "weight_mean": df["weight"].mean(),
        "weight_std": df["weight"].std(),
    }
