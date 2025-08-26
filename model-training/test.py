import warnings
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import typer
from augment import BasicAugment
from dataset import SlideTileDataset, read_tile_metadata, save
from loguru import logger
from losses import dice_focal_loss
from metrics import get_segmentation_metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *

warnings.filterwarnings("ignore")


@torch.no_grad()
def test(
    model, loader, criterion, device, confidence
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0

    all_outputs = []
    all_masks = []

    pbar = tqdm(loader, desc="Testing", ncols=100)
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
    model_path: Path = typer.Argument(..., help="Path to trained model directory"),
    config_path: Path = typer.Argument(
        "config.yaml", help="Path to the configuration YAML file"
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

    exp_dir = ExperimentDirectory("tumorseg_test", Path(config.data.target))

    logger.add(exp_dir.logs / "test.log")
    config.log(logger)

    set_seed(config.data.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_df = read_tile_metadata(config.data.source / "metadata")
    test_df = metadata_df[metadata_df["split"] == "test"]
    save(test_df, exp_dir.logs, "test.csv")

    test_dataset = SlideTileDataset(
        source=config.data.source,
        df=test_df,
        transform=BasicAugment(config.data.img_size),
        img_size=config.data.img_size,
    )

    test_loader = DataLoader(
        test_dataset,
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

    criterion = dice_focal_loss(gamma=1.0)

    logger.info("Running inference on test set...")
    test_loss, test_metrics = test(
        model, test_loader, criterion, device, config.test.conf
    )

    logger.success(f"Test Loss: {test_loss:.4f}")
    logger.success(
        "Test Metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
    )
    logger.info("Saving post-processed predictions for test samples...")

    test_pred_path = exp_dir.predictions / "test"
    save_predictions(
        model=model,
        dataset=test_dataset,
        target=test_pred_path,
        device=device,
        confidence=config.test.conf,
        num_samples=config.test.num_samples,
        post_process=True,
    )

    logger.success(f"Saved {config.test.num_samples} predictions at {test_pred_path}")


if __name__ == "__main__":
    typer.run(main)
