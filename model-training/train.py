import shutil
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import typer
from augment import BasicAugment, HeavyAugment
from dataset import FilteredDataset, read_tile_metadata
from losses import tversky_loss
from metrics import get_segmentation_metrics
from model import EarlyStopping, get_smp_model
from sampler import LimitedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from utils import (
    TrainingArguments,
    save_predictions,
    set_seed,
    visualize_training_samples,
)


def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training", ncols=100)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model, loader, criterion, device, confidence
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    pbar = tqdm(loader, desc="Validation", ncols=100)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, masks)

            batch_metrics = get_segmentation_metrics(outputs, masks, confidence)
            for k, v in batch_metrics.items():
                metrics_sum[k] += v

        total_loss += loss.item()
        pbar.set_postfix(
            {"loss": f"{loss.item():.3f}", "dice": f"{batch_metrics['dice']:.3f}"}
        )

    num_batches = len(loader)
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    return total_loss / num_batches, avg_metrics


def main(
    source: Path = typer.Argument(..., help="Path to dataset directory"),
    target: Path = typer.Argument(..., help="Path to output directory"),
    batch_size: int = typer.Option(32, help="Training batch size"),
    img_size: int = typer.Option(256, help="Input image size for training"),
    conf: float = typer.Option(0.5, help="Confidence"),
    lr: float = typer.Option(1e-4, help="Learning rate"),
    warmup_epochs: int = typer.Option(3, help="Number of warmup epochs"),
    weight_decay: float = typer.Option(1e-4, help="Weight decay"),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    es_patience: int = typer.Option(10, help="Early stopping patience"),
    es_delta: float = typer.Option(1e-4, help="Early stopping min delta"),
    arch: str = typer.Option("unet", help="Model architecture"),
    backbone: str = typer.Option("efficientnet-b0", help="Backbone network"),
    weights: str = typer.Option("imagenet", help="Pretrained weights"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
):
    args = TrainingArguments(
        source=source,
        target=target,
        batch_size=batch_size,
        img_size=img_size,
        conf=conf,
        lr=lr,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        epochs=epochs,
        es_patience=es_patience,
        es_delta=es_delta,
        arch=arch,
        backbone=backbone,
        weights=weights,
        seed=seed,
    )

    args.print_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("tumor-segmentation")

    metadata_df = read_tile_metadata(args.source / "metadata")
    train_df = metadata_df[metadata_df["split"] == "train"]
    val_df = metadata_df[metadata_df["split"] == "val"]

    # Training Dataset
    train_dataset = FilteredDataset(
        source=args.source,
        df=train_df,
        category="wsi_tiled",
        min_tumor_frac=0.01,
        transform=HeavyAugment(args.img_size),
        img_size=args.img_size,
    )
    print("Training Set:")
    train_dataset.print_tiles_per_slide()

    visualize_training_samples(
        dataset=train_dataset,
        target=args.target / "training_visualizations",
    )

    # Validation Dataset
    val_dataset = FilteredDataset(
        source=args.source,
        df=val_df,
        category="wsi_tiled",
        min_tumor_frac=0.01,
        transform=BasicAugment(args.img_size),
        img_size=args.img_size,
    )
    print("Validation Set:")
    val_dataset.print_tiles_per_slide()

    # Sampler
    train_sampler = LimitedSampler(
        dataset=train_dataset,
        max_tiles_per_slide=train_dataset.get_min_slide_tile_count(),
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(
            {
                "batch_size": args.batch_size,
                "img_size": args.img_size,
                "conf": args.conf,
                "lr": args.lr,
                "warmup_epochs": args.warmup_epochs,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "arch": args.arch,
                "backbone": args.backbone,
                "weights": args.weights,
                "es_patience": args.es_patience,
                "es_delta": args.es_delta,
                "seed": args.seed,
            }
        )

        model = get_smp_model(
            arch=args.arch, backbone=args.backbone, weights=args.weights
        ).to(device)

        print(f"Using device: {device}")
        print("Model Summary:")
        summary(model, (args.batch_size, 3, args.img_size, args.img_size), depth=1)

        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.epochs - args.warmup_epochs,
            T_mult=1,
            eta_min=args.lr * 0.01,
        )

        def get_lr(epoch):
            if epoch < args.warmup_epochs:
                return args.lr * (epoch + 1) / args.warmup_epochs
            return scheduler.get_last_lr()[0]

        criterion = tversky_loss()
        scaler = torch.GradScaler()
        early_stopping = EarlyStopping(
            patience=args.es_patience, min_delta=args.es_delta, mode="min", verbose=True
        )

        # Training loop
        best_val_loss = float("inf")
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

            current_lr = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
            )
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, args.conf
            )

            if epoch >= args.warmup_epochs:
                scheduler.step()

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "val_dice": val_metrics["dice"],
                    "val_iou": val_metrics["iou"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                },
                step=epoch,
            )

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print("Validation Metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                model_path = args.target / "best_model"
                shutil.rmtree(model_path, ignore_errors=True)
                model.save_pretrained(model_path)

                val_pred_path = args.target / "val_predictions"
                save_predictions(
                    model, val_dataset, val_pred_path, device, confidence=args.conf
                )

            if early_stopping(epoch, val_loss):
                print("\nTraining stopped early due to no improvement in val loss")
                break


if __name__ == "__main__":
    typer.run(main)
