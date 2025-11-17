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
from utils import *

warnings.filterwarnings("ignore")

app = typer.Typer(pretty_exceptions_show_locals=False)

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


def train(
    config: Config,
    fold: int = None,
):
    experiment_name = "tumorseg" if fold is None else f"tumorseg_fold_{fold}"
    exp_dir = ExperimentDirectory(experiment_name, Path(config.data.target))

    mlflow_uri = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    logger.info(f"MLflow tracking data will be saved to: {mlflow_uri}")

    log_file = logger.add(exp_dir.logs / "train.log")
    config.log(logger)

    set_seed(config.data.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if fold is not None:
        logger.info(f"Using cross-validation fold {fold}")
        split_file = f"cv/split_fold_{fold}.csv"
        metadata_df = read_tile_metadata(config.data.source / "metadata", split_file)
    else:
        logger.info("Using standard train/val/test split")
        metadata_df = read_tile_metadata(config.data.source / "metadata")
    logger.info(f"Loaded with {len(metadata_df)} samples")

    save(metadata_df, exp_dir.logs, "metadata.csv")

    train_df = metadata_df[metadata_df["split"] == "train"]
    save(train_df, exp_dir.logs, "train.csv")

    train_wsi_df = train_df[train_df["category"] == "wsi_tiled"]
    save(train_wsi_df, exp_dir.logs, "train_hospital.csv")

    train_hpa_df = train_df[train_df["category"] == "img_tiled"]
    save(train_hpa_df, exp_dir.logs, "train_hpa.csv")

    val_df = metadata_df[metadata_df["split"] == "val"]
    save(val_df, exp_dir.logs, "val.csv")

    test_df = metadata_df[metadata_df["split"] == "test"]
    save(test_df, exp_dir.logs, "test.csv")
    logger.info(f"Individual sets saved at {exp_dir.logs}")

    # Check if validation dataset is empty
    has_validation = len(val_df) > 0
    has_test = len(test_df) > 0

    if not has_validation:
        logger.warning("Validation dataset is empty - training without validation")
    if not has_test:
        logger.warning("Test dataset is empty")

    # Training Dataset
    train_wsi_dataset = SlideTileDataset(
        source=config.data.source,
        df=train_wsi_df,
        transform=HeavyAugment(config.data.img_size),
        img_size=config.data.img_size,
    )

    save_training_samples(
        dataset=train_wsi_dataset, target=exp_dir.samples / "train", suffix="wsi"
    )

    train_hpa_dataset = SlideTileDataset(
        source=config.data.source,
        df=train_hpa_df,
        transform=HeavyAugment(config.data.img_size),
        img_size=config.data.img_size,
    )

    save_training_samples(
        dataset=train_hpa_dataset, target=exp_dir.samples / "train", suffix="hpa"
    )

    # Validation Dataset (only create if validation data exists)
    val_dataset = None
    if has_validation:
        val_dataset = SlideTileDataset(
            source=config.data.source,
            df=val_df,
            transform=BasicAugment(config.data.img_size),
            img_size=config.data.img_size,
        )

    # Sampler
    q_n_sample = 1536
    train_sampler = SlideBalancedSampler(
        dataset=train_wsi_dataset,
        samples_per_epoch=q_n_sample * 3,
        seed=config.data.seed,
    )

    train_hpa_sampler = SlideBalancedSampler(
        dataset=train_hpa_dataset, samples_per_epoch=q_n_sample, seed=config.data.seed
    )

    # Loader
    batch_size = config.train.batch_size
    quarter_batch_size = batch_size // 4
    train_loader = DataLoader(
        train_wsi_dataset,
        batch_size=quarter_batch_size * 3,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    train_hpa_loader = DataLoader(
        train_hpa_dataset,
        batch_size=quarter_batch_size,
        sampler=train_hpa_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = None
    if has_validation:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    mlflow.set_experiment(experiment_name)
    run_cv_name = "" if fold is None else f"fold_{fold}"
    with mlflow.start_run(
        run_name=f"train_{run_cv_name}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    ):
        mlflow.log_dict(config.model_dump(), "config.json")
        mlflow.log_artifacts(exp_dir.logs, artifact_path="data_splits")

        model = get_smp_model(
            arch=config.model.arch,
            backbone=config.model.backbone,
            weights=config.model.weights,
        ).to(device)

        logger.info(f"Using device: {device}")
        dummy_input_size = (batch_size, 3, config.data.img_size, config.data.img_size)
        model_summary = summary(model, input_size=dummy_input_size, depth=1, verbose=0)
        logger.info(f"Model Summary\n{str(model_summary)}")

        optimizer = AdamW(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.train.epochs - config.train.warmup_epochs,
            T_mult=1,
            eta_min=config.train.lr * 0.01,
        )

        def get_lr(epoch):
            if epoch < config.train.warmup_epochs:
                return config.train.lr * (
                    0.1 + 0.9 * (epoch + 1) / config.train.warmup_epochs
                )
            return scheduler.get_last_lr()[0]

        criterion = dice_focal_loss(gamma=1.0)
        scaler = torch.GradScaler()
        early_stopping = EarlyStopping(
            patience=config.train.es_patience,
            min_delta=config.train.es_delta,
            mode="max",
            verbose=True,
        )

        # Training loop
        logger.info("Training started")
        best_val_loss = float("inf")
        for epoch in range(config.train.epochs):
            logger.info(f"Epoch {epoch + 1}/{config.train.epochs}")

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

            if has_validation:
                val_loss, val_metrics = validate(
                    model, val_loader, criterion, device, config.train.conf
                )
            else:
                val_loss = float("inf")
                val_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

            if epoch >= config.train.warmup_epochs:
                scheduler.step()

            metrics_to_log = {
                "train_loss": train_loss,
                "learning_rate": current_lr,
            }

            if has_validation:
                metrics_to_log.update(
                    {
                        "val_loss": val_loss,
                        "val_dice": val_metrics["dice"],
                        "val_iou": val_metrics["iou"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                    }
                )

            mlflow.log_metrics(metrics_to_log, step=epoch)

            if has_validation:
                logger.info(
                    f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
                )
                logger.info(
                    "Validation Metrics: "
                    + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    best_model_path = exp_dir.checkpoints / "best"
                    shutil.rmtree(best_model_path, ignore_errors=True)
                    model.save_pretrained(best_model_path)

                    val_pred_path = exp_dir.predictions / "val"
                    save_predictions(
                        model,
                        val_dataset,
                        val_pred_path,
                        device,
                        confidence=config.train.conf,
                    )
            else:
                logger.info(f"Train Loss: {train_loss:.4f} (No validation)")

            if has_validation and early_stopping(epoch, val_metrics["dice"]):
                logger.success(
                    "Training stopped early due to no improvement in val dice score"
                )
                break

        last_model_path = exp_dir.checkpoints / "last"
        model.save_pretrained(last_model_path)
        logger.success(f"Model saved at {last_model_path}")
        logger.remove(log_file)


@app.command()
def main(
    config_path: Path = typer.Argument(
        "config.yaml", help="Path to the configuration YAML file"
    ),
    cv: bool = typer.Option(
        False, "--cv", help="Train all folds back-to-back using CV"
    ),
):
    try:
        config = Config.from_yaml(config_path)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise typer.Exit(code=1)

    config.data.source = config.data.source.expanduser()
    n_folds = 1
    if cv:
        cv_path = config.data.source / "metadata" / "cv"
        if not cv_path.exists():
            logger.error(f"Cross-validation directory not found: {cv_path}")
            raise typer.Exit(code=1)
        n_folds = len(list(cv_path.glob("*.csv")))
    logger.info(f"Training {'all' if cv else 'single'} {n_folds} fold(s)")

    for fold_idx in range(n_folds):
        current_fold = fold_idx if cv else None
        logger.info(
            f"--- Starting {'Fold ' + str(current_fold) if cv else 'Training'} ---"
        )
        train(config, fold=current_fold)
    logger.success(f"{'All folds' if cv else 'Training'} completed")


if __name__ == "__main__":
    app()
