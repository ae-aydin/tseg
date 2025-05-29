import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import mlflow
import torch
import typer
from dataset import (TileDataset, get_train_augmentations,
                     get_val_augmentations, read_tile_metadata)
from metrics import (get_bce_dice_loss, get_segmentation_metrics,
                     print_sampling_stats, save_predictions)
from model import EarlyStopping, get_smp_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchinfo import summary
from tqdm import tqdm


@dataclass
class TrainingArguments:
    dataset_path: Path
    output_path: Path
    batch_size: int = 16
    num_workers: int = 4
    img_size: int = 512

    arch: Literal["unet", "unetplusplus", "linknet"] = "unet"
    backbone: Literal[
        "resnet18", "resnet34", "resnet50", "mobilenet_v2", "efficientnet-b0"
    ] = "mobilenet_v2"
    weights: Literal["imagenet", None] = "imagenet"

    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    epochs: int = 50
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    def print_args(self):
        print("\nTraining Arguments:")
        print("-" * 50)
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                print(f"{key:25} = {value.resolve()}")
            else:
                print(f"{key:25} = {value}")
        print("-" * 50 + "\n")


def train_epoch(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc="Training", ncols=100)
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

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion, device) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    pbar = tqdm(val_loader, desc="Validation", ncols=100)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, masks)

            batch_metrics = get_segmentation_metrics(outputs, masks)
            for k, v in batch_metrics.items():
                metrics_sum[k] += v

        total_loss += loss.item()
        pbar.set_postfix(
            {"loss": f"{loss.item():.3f}", "dice": f"{batch_metrics['dice']:.3f}"}
        )

    num_batches = len(val_loader)
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    return total_loss / num_batches, avg_metrics


def main(
    dataset_path: Path = typer.Argument(..., help="Path to dataset directory"),
    output_path: Path = typer.Argument(..., help="Path to output directory"),
    batch_size: int = typer.Option(16, help="Training batch size"),
    img_size: int = typer.Option(512, help="Input image size for training"),
    learning_rate: float = typer.Option(1e-4, help="Learning rate"),
    weight_decay: float = typer.Option(1e-3, help="Weight decay"),
    epochs: int = typer.Option(30, help="Number of training epochs"),
    arch: str = typer.Option("unet", help="Model architecture"),
    backbone: str = typer.Option("mobilenet_v2", help="Backbone network"),
    weights: str = typer.Option("imagenet", help="Pretrained weights"),
    early_stopping_patience: int = typer.Option(3, help="Early stopping patience"),
    early_stopping_min_delta: float = typer.Option(
        0.0, help="Early stopping minimum delta"
    ),
    domain_weight: float = typer.Option(
        None, help="Weight for img_tiled domain (None for auto)"
    ),
):
    args = TrainingArguments(
        dataset_path=dataset_path,
        output_path=output_path,
        batch_size=batch_size,
        img_size=img_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        arch=arch,
        backbone=backbone,
        weights=weights,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    args.print_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("tumor-segmentation")

    metadata_df = read_tile_metadata(args.dataset_path / "metadata")
    train_df = metadata_df[metadata_df["split"] == "train"]
    val_df = metadata_df[metadata_df["split"] == "val"]

    train_dataset = TileDataset(
        source=args.dataset_path,
        df=train_df,
        img_size=args.img_size,
        transform=get_train_augmentations(args.img_size),
        domain_weight=domain_weight,
    )

    val_dataset = TileDataset(
        source=args.dataset_path,
        df=val_df,
        img_size=args.img_size,
        transform=get_val_augmentations(args.img_size),
    )

    print_sampling_stats(train_dataset)

    train_sampler = WeightedRandomSampler(
        weights=train_dataset.weights, num_samples=len(train_dataset), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(
            {
                "batch_size": args.batch_size,
                "img_size": args.img_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "arch": args.arch,
                "backbone": args.backbone,
                "weights": args.weights,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
            }
        )

        model = get_smp_model(
            arch=args.arch, backbone=args.backbone, weights=args.weights
        ).to(device)

        print(f"Using device: {device}")
        print("Model Summary:")
        summary(model, (args.batch_size, 3, args.img_size, args.img_size), depth=1)

        optimizer = AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = get_bce_dice_loss()
        scaler = torch.GradScaler()
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode="min",
            verbose=True,
        )

        # Training loop
        best_val_loss = float("inf")
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
            )
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            scheduler.step()

            # Log only training metrics
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "val_dice": val_metrics["dice"],
                    "val_iou": val_metrics["iou"],
                },
                step=epoch,
            )

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print("Validation Metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                model_path = args.output_path / "best_model"
                shutil.rmtree(model_path, ignore_errors=True)
                model.save_pretrained(model_path)

                val_pred_path = args.output_path / "val_predictions"
                save_predictions(model, val_loader, device, val_pred_path)

            if early_stopping(epoch, val_loss):
                print(
                    "\nTraining stopped early due to no improvement in validation loss"
                )
                break


if __name__ == "__main__":
    typer.run(main)
