import shutil
import warnings
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import typer
from augment import BasicAugment, HeavyAugment
from dataset import SlideTileDataset, read_tile_metadata, save
from loguru import logger
from losses import dice_focal_loss
from metrics import get_segmentation_metrics
from model import EarlyStopping, get_smp_model
from sampler import SlideBalancedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from utils import (
    ExperimentDirectory,
    TrainingArguments,
    save_predictions,
    save_training_samples,
    set_seed,
)

warnings.filterwarnings("ignore")


def train_epoch(model, loader1, loader2, optimizer, criterion, device, scaler):
    assert len(loader1) == len(loader2)
    total_len = len(loader1)

    model.train()
    total_loss = 0
    pbar = tqdm(
        zip(loader1, loader2),
        desc="Training",
        ncols=100,
        total=total_len,
    )
    for batch1, batch2 in pbar:
        # Combine batches
        images = torch.cat([batch1["image"], batch2["image"]], dim=0).to(
            device, non_blocking=True
        )
        masks = torch.cat([batch1["mask"], batch2["mask"]], dim=0).to(
            device, non_blocking=True
        )

        # Shuffle
        batch_size = images.size(0)
        shuffle_indices = torch.randperm(batch_size, device=device)
        images = images[shuffle_indices]
        masks = masks[shuffle_indices]

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    return total_loss / total_len


@torch.no_grad()
def validate(
    model, loader, criterion, device, confidence
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0

    all_outputs = []
    all_masks = []

    pbar = tqdm(loader, desc="Validation", ncols=100)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, masks)

        total_loss += loss.item()

        all_outputs.append(outputs.cpu().detach())
        all_masks.append(masks.cpu().detach())

    all_outputs = torch.cat(all_outputs)
    all_masks = torch.cat(all_masks)

    epoch_metrics = get_segmentation_metrics(all_outputs, all_masks, confidence)
    avg_loss = total_loss / len(loader)

    return avg_loss, epoch_metrics


def main(
    source: Path = typer.Argument(..., help="Path to dataset directory"),
    target: str = typer.Option("experiments", help="Path to output directory"),
    batch_size: int = typer.Option(32, help="Training batch size"),
    img_size: int = typer.Option(256, help="Input image size for training"),
    conf: float = typer.Option(0.5, help="Confidence"),
    lr: float = typer.Option(1e-4, help="Learning rate"),
    warmup_epochs: int = typer.Option(3, help="Number of warmup epochs"),
    weight_decay: float = typer.Option(1e-4, help="Weight decay"),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    es_patience: int = typer.Option(10, help="Early stopping patience"),
    es_delta: float = typer.Option(1e-4, help="Early stopping min delta"),
    arch: str = typer.Option("unetplusplus", help="Model architecture"),
    backbone: str = typer.Option("efficientnet-b0", help="Backbone network"),
    weights: str = typer.Option("imagenet", help="Pretrained weights"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
):
    exp = ExperimentDirectory("tumorseg", Path(target))
    args = TrainingArguments(
        source=source,
        target=exp,
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

    logger.add(args.target.logs / "train.log")
    logger.info(args)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment(args.target.experiment_name)

    metadata_df = read_tile_metadata(args.source / "metadata")
    save(metadata_df, args.target.logs, "metadata.csv")

    train_df = metadata_df[metadata_df["split"] == "train"]
    save(train_df, args.target.logs, "train.csv")

    train_wsi_df = train_df[train_df["category"] == "wsi_tiled"]
    save(train_wsi_df, args.target.logs, "train_hospital.csv")

    train_hpa_df = train_df[train_df["category"] == "img_tiled"]
    save(train_hpa_df, args.target.logs, "train_hpa.csv")

    val_df = metadata_df[metadata_df["split"] == "val"]
    save(val_df, args.target.logs, "val.csv")

    test_df = metadata_df[metadata_df["split"] == "test"]
    save(test_df, args.target.logs, "test.csv")
    logger.info(f"Individual sets saved at {args.target.logs}")

    # Training Dataset
    train_wsi_dataset = SlideTileDataset(
        source=args.source,
        df=train_wsi_df,
        transform=HeavyAugment(args.img_size),
        img_size=args.img_size,
    )

    save_training_samples(
        dataset=train_wsi_dataset, target=args.target.samples / "train", suffix="wsi"
    )

    train_hpa_dataset = SlideTileDataset(
        source=args.source,
        df=train_hpa_df,
        transform=HeavyAugment(args.img_size),
        img_size=args.img_size,
    )

    save_training_samples(
        dataset=train_hpa_dataset, target=args.target.samples / "train", suffix="hpa"
    )

    # Validation Dataset
    val_dataset = SlideTileDataset(
        source=args.source,
        df=val_df,
        transform=BasicAugment(args.img_size),
        img_size=args.img_size,
    )

    # Sampler
    q_n_sample = 1536
    train_sampler = SlideBalancedSampler(
        dataset=train_wsi_dataset, samples_per_epoch=q_n_sample * 3, seed=args.seed
    )

    train_hpa_sampler = SlideBalancedSampler(
        dataset=train_hpa_dataset, samples_per_epoch=q_n_sample, seed=args.seed
    )

    # Loader
    q_batch_size = args.batch_size // 4
    train_loader = DataLoader(
        train_wsi_dataset,
        batch_size=q_batch_size * 3,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    train_hpa_loader = DataLoader(
        train_hpa_dataset,
        batch_size=q_batch_size,
        sampler=train_hpa_sampler,
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

    with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d%H%M%S')}"):
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

        logger.info(f"Using device: {device}")
        dummy_input_size = (args.batch_size, 3, args.img_size, args.img_size)
        model_summary = summary(model, input_size=dummy_input_size, depth=1, verbose=0)
        logger.info(f"Model Summary\n {str(model_summary)}")

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
                return args.lr * (0.1 + 0.9 * (epoch + 1) / args.warmup_epochs)
            return scheduler.get_last_lr()[0]

        criterion = dice_focal_loss(gamma=1.0)
        scaler = torch.GradScaler()
        early_stopping = EarlyStopping(
            patience=args.es_patience, min_delta=args.es_delta, mode="max", verbose=True
        )

        # Training loop
        logger.info("Training started")
        best_val_loss = float("inf")
        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch + 1}/{args.epochs}")

            current_lr = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            train_loss = train_epoch(
                model,
                train_loader,
                train_hpa_loader,
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

            logger.info(
                f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )
            logger.info(
                "Validation Metrics: "
                + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                best_model_path = args.target.checkpoints / "best"
                shutil.rmtree(best_model_path, ignore_errors=True)
                model.save_pretrained(best_model_path)

                val_pred_path = args.target.predictions / "val"
                save_predictions(
                    model, val_dataset, val_pred_path, device, confidence=args.conf
                )

            if early_stopping(epoch, val_metrics["dice"]):
                logger.success(
                    "Training stopped early due to no improvement in val dice score"
                )
                break

        last_model_path = args.target.checkpoints / "last"
        model.save_pretrained(last_model_path)
        logger.success(f"Model saved at {last_model_path}")


if __name__ == "__main__":
    typer.run(main)
