from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .image import unnormalize, post_process_image


def save_training_samples(
    dataset: Dataset,
    target: Path,
    suffix: str,
    num_samples: int = 4,
    num_images: int = 3,
) -> None:
    target.mkdir(exist_ok=True)

    for i in range(num_images):
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
        fig.patch.set_facecolor("beige")

        indices = np.random.choice(len(dataset), num_samples, replace=False)

        for j, idx in enumerate(indices):
            sample = dataset[idx]
            img = sample["image"].cpu().numpy()
            img = unnormalize(img)
            mask = sample["mask"].cpu().numpy().squeeze()

            axes[j, 0].imshow(img)
            axes[j, 0].set_title(
                f"Sample {j + 1}\n"
                f"Slide: {sample['slide_name']}\n"
                f"Category: {sample['category']}"
            )
            axes[j, 0].axis("off")

            axes[j, 1].imshow(mask, cmap="gray")
            axes[j, 1].set_title(
                f"Mask {j + 1}\n"
                f"Tumor: {sample['tumor_frac']:.2f}\n"
                f"Tumor Bin: {sample['tumor_bin']}"
            )
            axes[j, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            target / f"sample_{suffix}_{i}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()


def save_predictions(
    model: torch.nn.Module,
    dataset,
    target: Path,
    device: torch.device,
    confidence: float = 0.5,
    num_samples: int | None = 5,
    post_process: bool = False,
):
    target.mkdir(parents=True, exist_ok=True)

    if num_samples:
        all_indices = np.arange(len(dataset))
        selected_indices = np.random.choice(all_indices, num_samples, replace=False)
        subset = Subset(dataset, selected_indices)
        loader = DataLoader(subset, batch_size=1, shuffle=False)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type):
        for i, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            slide_names = batch["slide_name"]

            preds = torch.sigmoid(model(images))
            preds = (preds > confidence).float()

            img = images[0].cpu().numpy()
            img = unnormalize(img)
            mask = masks[0].cpu().numpy().squeeze()
            pred = preds[0].cpu().numpy().squeeze()

            post_processed = post_process_image(pred) if post_process else None
            fig_cols = 4 if post_process else 3

            fig, axes = plt.subplots(1, fig_cols, figsize=(5 * fig_cols, 5))
            fig.patch.set_facecolor("beige")

            axes[0].imshow(img)
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(pred, cmap="gray")
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            if post_process:
                axes[3].imshow(post_processed, cmap="gray")
                axes[3].set_title("Post-Process")
                axes[3].axis("off")

            plt.suptitle(f"Slide: {slide_names[0]}")
            plt.tight_layout()
            plt.savefig(
                target / f"sample_{i + 1:04d}.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close()
