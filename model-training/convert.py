import warnings
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import typer
from loguru import logger

warnings.filterwarnings("ignore")


def main(
    source: Path = typer.Argument(..., help="Path to saved model directory"),
    target: Path = typer.Argument(..., help="Output path to create models directory"),
    model_name: str = typer.Argument(..., help="Descriptive model name"),
    img_size: int = typer.Option(256, help="Input image size for export"),
    batch_size: int = typer.Option(1, help="Batch size for export"),
):
    logger.info(f"Source model: {source}")
    logger.info(f"Target directory: {target}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Image size: {img_size}x{img_size}")
    logger.info(f"Batch size: {batch_size}")

    output_path = target / "models"
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Currently only for smp based models
    logger.info("Loading original model...")
    model = smp.from_pretrained(source)
    model.eval()
    logger.success("Original model loaded successfully")

    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    logger.info("Converting to ONNX...")
    onnx_path = output_path / f"{model_name}.onnx"
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,  # doesnt work for smp unet
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    logger.success(f"ONNX model saved to: {onnx_path}")

    logger.info("Converting to TorchScript...")
    torchscript_path = output_path / f"{model_name}.pt"
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(torchscript_path)
    logger.success(f"TorchScript model saved to: {torchscript_path}")
    logger.success("All conversions completed successfully")


if __name__ == "__main__":
    typer.run(main)
