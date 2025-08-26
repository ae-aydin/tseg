import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import typer
from augment import BasicAugment
from dataset import SlideTileDataset, read_tile_metadata, save
from loguru import logger
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *

warnings.filterwarnings("ignore")


@torch.no_grad()
def get_predictions_with_slides(
    model, loader, device
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Get predictions and ground truth with slide information."""
    model.eval()
    all_outputs = []
    all_masks = []
    all_slides = []

    pbar = tqdm(loader, desc="Getting predictions", ncols=100)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        slides = batch["slide_name"]

        with torch.autocast(device_type=device.type):
            outputs = model(images)

        all_outputs.append(outputs.cpu().detach())
        all_masks.append(masks.cpu().detach())
        all_slides.extend(slides)

    return torch.cat(all_outputs), torch.cat(all_masks), all_slides


def calculate_metrics_at_threshold(
    y_pred: np.ndarray, y_true: np.ndarray
) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score at a threshold."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision, recall, f1


def calculate_per_slide_metrics(
    outputs: torch.Tensor, masks: torch.Tensor, slides: List[str], threshold: float
) -> pd.DataFrame:
    """Calculate metrics for each slide at a given threshold using micro averaging."""
    slide_metrics = []

    # Apply sigmoid and threshold
    probs = torch.sigmoid(outputs).numpy()
    y_pred = (probs > threshold).astype(int)
    y_true = masks.numpy()

    # Group by slide
    unique_slides = list(set(slides))

    for slide_name in unique_slides:
        # Get indices for this slide
        slide_indices = [i for i, s in enumerate(slides) if s == slide_name]

        # Combine all tiles from this slide
        pred_slide = y_pred[slide_indices].flatten()
        true_slide = y_true[slide_indices].flatten()

        precision, recall, f1 = calculate_metrics_at_threshold(pred_slide, true_slide)

        slide_metrics.append(
            {
                "slide_name": slide_name,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "num_tiles": len(slide_indices),
            }
        )

    return pd.DataFrame(slide_metrics)


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    avg_precision: float,
    save_path: Path,
    title: str = "Precision-Recall Curve",
):
    """Plot and save PR curve."""
    plt.figure(figsize=(10, 8))

    plt.plot(
        recall,
        precision,
        "b-",
        linewidth=2,
        label=f"PR curve (AP = {avg_precision:.3f})",
    )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.text(
        0.05,
        0.95,
        f"Average Precision: {avg_precision:.3f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    save_path: Path,
    title: str = "ROC Curve",
):
    """Plot and save ROC curve."""
    plt.figure(figsize=(10, 8))

    plt.plot(fpr, tpr, "r-", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.text(
        0.05,
        0.95,
        f"ROC AUC: {roc_auc:.3f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "Confusion Matrix",
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.colorbar()

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_vs_confidence(
    thresholds: List[float],
    metrics: Dict[str, List[float]],
    save_path: Path,
    title: str = "Metrics vs Confidence Threshold",
):
    """Plot metrics vs confidence threshold."""
    plt.figure(figsize=(12, 8))

    colors = ["blue", "red", "green"]
    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.plot(
            thresholds,
            values,
            "o-",
            color=colors[i],
            linewidth=2,
            markersize=6,
            label=metric_name.capitalize(),
        )

    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(
    model_path: Path = typer.Argument(..., help="Path to trained model directory"),
    config_path: Path = typer.Argument(
        "config.yaml", help="Path to the configuration YAML file"
    ),
    split: str = typer.Option("val", help="Dataset split to evaluate (val/test)"),
):
    try:
        config = Config.from_yaml(config_path)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise typer.Exit(code=1)

    exp_dir = ExperimentDirectory("tumorseg_val", Path(config.data.target))

    logger.add(exp_dir.logs / "val.log")
    config.log(logger)

    set_seed(config.data.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_df = read_tile_metadata(config.data.source / "metadata")
    split_df = metadata_df[metadata_df["split"] == split]
    save(split_df, exp_dir.logs, f"{split}.csv")

    logger.info(f"Found {len(split_df)} samples for {split} split")
    logger.info(f"Number of slides: {split_df['slide_name'].nunique()}")
    logger.info(f"Slides: {split_df['slide_name'].unique()}")

    split_dataset = SlideTileDataset(
        source=config.data.source,
        df=split_df,
        transform=BasicAugment(config.data.img_size),
        img_size=config.data.img_size,
    )

    split_loader = DataLoader(
        split_dataset,
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    logger.info(f"Loading model from {model_path}")
    model = smp.from_pretrained(model_path).to(device)

    model.eval()
    logger.info(f"Using device: {device}")

    # Get predictions with slide information
    logger.info("Getting predictions...")
    outputs, masks, slides = get_predictions_with_slides(model, split_loader, device)

    # Convert to numpy for analysis
    probs = torch.sigmoid(outputs).numpy()
    y_pred_overall = probs.flatten()
    y_true_overall = masks.numpy().flatten()

    logger.info(f"Predictions shape: {outputs.shape}")
    logger.info(
        f"Positive pixels: {np.sum(y_true_overall > 0.5)} / {len(y_true_overall)} ({np.sum(y_true_overall > 0.5) / len(y_true_overall) * 100:.2f}%)"
    )

    # Calculate ROC curve and AUC
    logger.info("Calculating ROC curve and AUC...")
    fpr, tpr, roc_thresholds = roc_curve(y_true_overall, y_pred_overall)
    roc_auc = auc(fpr, tpr)
    logger.info(f"ROC AUC: {roc_auc:.4f}")

    # Calculate PR curve and AP
    logger.info("Calculating PR curve and AP...")
    precision_overall, recall_overall, pr_thresholds = precision_recall_curve(
        y_true_overall, y_pred_overall
    )
    avg_precision = average_precision_score(y_true_overall, y_pred_overall)
    logger.info(f"Average Precision: {avg_precision:.4f}")

    # Plot ROC curve
    roc_curve_path = exp_dir.logs / f"roc_curve_{split}.png"
    plot_roc_curve(
        fpr, tpr, roc_auc, roc_curve_path, f"ROC Curve - {split.upper()} Set"
    )
    logger.info(f"ROC curve saved to: {roc_curve_path}")

    # Plot PR curve
    pr_curve_path = exp_dir.logs / f"pr_curve_{split}.png"
    plot_pr_curve(
        precision_overall,
        recall_overall,
        avg_precision,
        pr_curve_path,
        f"Precision-Recall Curve - {split.upper()} Set",
    )
    logger.info(f"PR curve saved to: {pr_curve_path}")

    # Calculate metrics at different thresholds
    logger.info("Calculating metrics at different thresholds...")
    threshold_list = np.arange(0.1, 0.95, 0.05)
    overall_metrics = []
    per_slide_metrics_list = []

    for threshold in tqdm(threshold_list, desc="Processing thresholds"):
        # Overall metrics
        y_pred_binary = (y_pred_overall > threshold).astype(int)
        precision, recall, f1 = calculate_metrics_at_threshold(
            y_pred_binary, y_true_overall
        )

        overall_metrics.append(
            {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1}
        )

        # Per-slide metrics
        slide_metrics_df = calculate_per_slide_metrics(
            outputs, masks, slides, threshold
        )
        per_slide_metrics_list.append(slide_metrics_df)

    overall_metrics_df = pd.DataFrame(overall_metrics)

    # Find best threshold based on per-slide F1 mean
    logger.info("Finding best threshold based on per-slide metrics...")
    slide_f1_means = []
    for slide_metrics_df in per_slide_metrics_list:
        slide_f1_means.append(slide_metrics_df["f1"].mean())

    best_slide_idx = np.argmax(slide_f1_means)
    best_threshold = threshold_list[best_slide_idx]
    best_slide_f1 = slide_f1_means[best_slide_idx]

    logger.info(
        f"Best threshold based on per-slide F1: {best_threshold:.3f} (F1: {best_slide_f1:.4f})"
    )

    # Plot confusion matrix at best threshold
    y_pred_best = (y_pred_overall > best_threshold).astype(int)
    cm_path = exp_dir.logs / f"confusion_matrix_{split}.png"
    plot_confusion_matrix(
        y_true_overall,
        y_pred_best,
        cm_path,
        f"Confusion Matrix - {split.upper()} Set (Threshold: {best_threshold:.3f})",
    )
    logger.info(f"Confusion matrix saved to: {cm_path}")

    # Plot metrics vs confidence
    metrics_vs_conf_path = exp_dir.logs / f"metrics_vs_confidence_{split}.png"
    plot_metrics_vs_confidence(
        threshold_list.tolist(),
        {
            "precision": overall_metrics_df["precision"].tolist(),
            "recall": overall_metrics_df["recall"].tolist(),
            "f1": overall_metrics_df["f1"].tolist(),
        },
        metrics_vs_conf_path,
        f"Metrics vs Confidence Threshold - {split.upper()} Set",
    )
    logger.info(f"Metrics vs confidence plot saved to: {metrics_vs_conf_path}")

    # Get per-slide metrics at best threshold
    best_slide_metrics_df = per_slide_metrics_list[best_slide_idx]

    # Summary statistics
    mean_f1 = best_slide_metrics_df["f1"].mean()
    std_f1 = best_slide_metrics_df["f1"].std()
    min_f1 = best_slide_metrics_df["f1"].min()
    max_f1 = best_slide_metrics_df["f1"].max()

    logger.info("Per-slide F1 summary at best threshold:")
    logger.info(f"  Mean: {mean_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"  Min: {min_f1:.4f}")
    logger.info(f"  Max: {max_f1:.4f}")

    # Save results
    logger.info("Saving results...")

    # Save overall metrics
    overall_csv_path = exp_dir.logs / f"overall_metrics_{split}.csv"
    overall_metrics_df.to_csv(overall_csv_path, index=False)
    logger.info(f"Overall metrics saved to: {overall_csv_path}")

    # Save per-slide metrics at best threshold
    slide_csv_path = exp_dir.logs / f"per_slide_metrics_{split}.csv"
    best_slide_metrics_df.to_csv(slide_csv_path, index=False)
    logger.info(f"Per-slide metrics saved to: {slide_csv_path}")

    # Save summary
    summary = {
        "split": split,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "best_threshold": best_threshold,
        "best_slide_f1": best_slide_f1,
        "mean_slide_f1": mean_f1,
        "std_slide_f1": std_f1,
        "min_slide_f1": min_f1,
        "max_slide_f1": max_f1,
        "num_slides": len(best_slide_metrics_df),
        "num_tiles": len(split_df),
    }

    summary_df = pd.DataFrame([summary])
    summary_path = exp_dir.logs / f"summary_{split}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to: {summary_path}")

    logger.success(
        f"Validation analysis complete! All results saved to: {exp_dir.logs}"
    )


if __name__ == "__main__":
    typer.run(main)
