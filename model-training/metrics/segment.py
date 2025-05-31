import segmentation_models_pytorch as smp
import torch


def get_segmentation_metrics(
    pred: torch.Tensor, target: torch.Tensor, confidence: float = 0.5
) -> dict[str, float]:
    pred = (torch.sigmoid(pred) > confidence).int()
    tp, fp, fn, tn = smp.metrics.get_stats(
        pred, target.int(), mode="binary", threshold=confidence
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
