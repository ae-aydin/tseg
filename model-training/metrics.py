from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from dataset import TileDataset
from torch.utils.data import DataLoader


def get_bce_dice_loss(alpha: float = 1.0, beta: float = 1.0):
    bce = smp.losses.SoftBCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary")
    return lambda pred, target: alpha * bce(pred, target) + beta * dice(pred, target)


def get_segmentation_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    pred = (torch.sigmoid(pred) > 0.5).int()
    tp, fp, fn, tn = smp.metrics.get_stats(
        pred, target.int(), mode="binary", threshold=0.5
    )

    dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }


def save_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    test: bool = False,
) -> None:
    model.eval()
    output_path.mkdir(exist_ok=True)
    num_samples = 10 if test else 3

    with torch.no_grad(), torch.autocast(device_type=device.type):
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break

            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            slide_names = batch["slide_name"]

            preds = torch.sigmoid(model(images))
            preds = (preds > 0.5).float()

            for j in range(min(images.shape[0], 2)):
                img = images[j].cpu().numpy().transpose(1, 2, 0)
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                mask = masks[j].cpu().numpy().squeeze()
                pred = preds[j].cpu().numpy().squeeze()

                if test:
                    sample_metrics = get_segmentation_metrics(
                        torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(device),
                        masks[j : j + 1],
                    )

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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

                plt.suptitle(f"Slide: {slide_names[j]}")
                plt.tight_layout()
                plt.savefig(
                    output_path / f"sample_{i + 1}_{j + 1}.png",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                plt.close()

                if test:
                    pd.DataFrame(
                        [
                            {
                                "slide": slide_names[j],
                                "sample_idx": f"{i + 1}_{j + 1}",
                                **sample_metrics,
                            }
                        ]
                    ).to_csv(
                        output_path / f"sample_{i + 1}_{j + 1}_metrics.csv", index=False
                    )


def print_sampling_stats(dataset: TileDataset) -> None:
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
        df.groupby("tumor_bin", observed=False)
        .agg(
            {
                "weight": ["sum", "mean", "std"],
            }
        )
        .round(4)
    )
    tumor_stats[("weight", "count")] = df.groupby("tumor_bin", observed=False).size()

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

    print(f"\nTotal samples: {len(dataset)}")
    print(f"Weight sum: {df['weight'].sum():.4f} (should be close to 1.0)")
    print(f"Weight mean: {df['weight'].mean():.4f}")
    print(f"Weight std: {df['weight'].std():.4f}")
    print("-" * 50 + "\n")
