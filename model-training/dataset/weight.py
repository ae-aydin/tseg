from pathlib import Path

import torch

from .base import BaseDataset


class WeightedDomainDataset(BaseDataset):
    def __init__(
        self,
        source: Path,
        df,
        transform=None,
        img_size: int = 512,
        domain_weight: float = None,
        gamma: float = 0.2,
        eps: float = 1.0,
    ):
        super().__init__(source, df, transform, img_size)
        self.area = img_size * img_size

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

    def print_stats(self):
        domain_stats = (
            self.df.groupby("category")
            .agg(
                {
                    "weight": ["sum", "mean", "std"],
                    "slide_name": "nunique",
                }
            )
            .round(4)
        )
        domain_stats[("weight", "count")] = self.df.groupby("category").size()

        tumor_stats = (
            self.df.groupby("tumor_bin", observed=False)
            .agg(
                {
                    "weight": ["sum", "mean", "std"],
                }
            )
            .round(4)
        )
        tumor_stats[("weight", "count")] = self.df.groupby(
            "tumor_bin", observed=False
        ).size()

        print("\nTraining Dataset Sampling Statistics:")
        print("-" * 50)

        print("\nDomain-level Statistics:")
        print("Category Distribution:")
        for category in domain_stats.index:
            count = domain_stats.loc[category, ("weight", "count")]
            weight_sum = domain_stats.loc[category, ("weight", "sum")]
            print(f"{category:12} - Count: {count:5d}, Weight Sum: {weight_sum:.4f}")

        print("\nTumor Fraction Distribution:")
        for tumor_bin in tumor_stats.index:
            count = tumor_stats.loc[tumor_bin, ("weight", "count")]
            weight_sum = tumor_stats.loc[tumor_bin, ("weight", "sum")]
            print(f"{tumor_bin:12} - Count: {count:5d}, Weight Sum: {weight_sum:.4f}")

        print(f"\nTotal samples: {len(self.df)}")
        print(f"Weight sum: {self.df['weight'].sum():.4f} (should be close to 1.0)")
        print(f"Weight mean: {self.df['weight'].mean():.4f}")
        print(f"Weight std: {self.df['weight'].std():.4f}")
        print("-" * 50 + "\n")
