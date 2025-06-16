import segmentation_models_pytorch as smp
import torch


def get_segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    confidence: float = 0.5,
    reduction: str = "micro-imagewise",
) -> dict[str, float]:
    pred = (torch.sigmoid(pred) > confidence).int()
    target = target.int()
    tp, fp, fn, tn = smp.metrics.get_stats(pred, target, mode="binary")

    dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction=reduction)
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=reduction)
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction=reduction)
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction=reduction)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }
