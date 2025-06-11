import warnings
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import typer
from augment import BasicAugment
from dataset import SlideTileDataset, read_tile_metadata, save
from losses import dice_focal_loss
from metrics import get_segmentation_metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import TestArguments, save_predictions, set_seed

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

        all_outputs.append(outputs.cpu())
        all_masks.append(masks.cpu())

    all_outputs = torch.cat(all_outputs)
    all_masks = torch.cat(all_masks)

    epoch_metrics = get_segmentation_metrics(all_outputs, all_masks, confidence)
    avg_loss = total_loss / len(loader)

    return avg_loss, epoch_metrics


def main(
    model_path: Path = typer.Argument(..., help="Path to trained model directory"),
    source: Path = typer.Argument(..., help="Path to dataset directory"),
    target: Path = typer.Argument(..., help="Path to output directory"),
    batch_size: int = typer.Option(32, help="Test batch size"),
    img_size: int = typer.Option(256, help="Input image size for testing"),
    conf: float = typer.Option(0.5, help="Confidence threshold"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    num_samples: int = typer.Option(25, help="Number of samples to visualize"),
):
    args = TestArguments(
        model_path=model_path,
        source=source,
        target=target,
        batch_size=batch_size,
        img_size=img_size,
        conf=conf,
        seed=seed,
    )

    args.print_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_df = read_tile_metadata(args.source / "metadata")
    test_df = metadata_df[metadata_df["split"] == "test"]
    test_df = test_df[test_df["category"] == "wsi_tiled"]
    save(test_df, args.target / "sheets", "test.csv")

    test_dataset = SlideTileDataset(
        source=args.source,
        df=test_df,
        transform=BasicAugment(args.img_size),
        img_size=args.img_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"Loading model from {model_path}")
    model = smp.from_pretrained(args.model_path).to(device)

    model.eval()
    print(f"Using device: {device}")

    criterion = dice_focal_loss()

    print("\nRunning inference on test set...")
    test_loss, test_metrics = test(model, test_loader, criterion, device, args.conf)

    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print("Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nSaving post-processed predictions for test samples...")

    save_predictions(
        model=model,
        dataset=test_dataset,
        target=args.target / "test_predictions",
        device=device,
        confidence=args.conf,
        num_samples=num_samples,
        post_process=True,
    )

    total_saved_samples = num_samples if num_samples else len(test_dataset)
    print(
        f"Saved {total_saved_samples} predictions at {args.target / 'test_predictions'}"
    )


if __name__ == "__main__":
    typer.run(main)
