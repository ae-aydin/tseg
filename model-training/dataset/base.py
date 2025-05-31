from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        source: Path,
        df,
        transform=None,
        img_size: int = 512,
    ):
        self.source = source
        self.transform = transform
        self.img_size = img_size
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def load_image_mask(self, row):
        img_path = self.source / row["parent_dir_path"] / row["relative_image_path"]
        mask_path = self.source / row["parent_dir_path"] / row["relative_mask_path"]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        return image, mask

    def apply_transform(self, image, mask):
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            mask = mask.float().div(255).unsqueeze(0)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image, mask = self.load_image_mask(row)
        image, mask = self.apply_transform(image, mask)
        return {
            "image": image,
            "mask": mask,
            "slide_name": row["slide_name"],
            "category": row["category"],
            "tumor_frac": row["tumor_frac"],
            "tumor_bin": row["tumor_bin"],
            "idx": idx,
        }
