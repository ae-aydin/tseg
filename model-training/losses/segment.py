import segmentation_models_pytorch as smp
from torch import Tensor


def dice_loss():
    return smp.losses.DiceLoss(mode="binary")


def focal_loss(gamma: float = 2.0):
    return smp.losses.FocalLoss(mode="binary", gamma=gamma)


def tversky_loss(alpha: float = 0.5, beta: float = 0.5):
    return smp.losses.TverskyLoss(mode="binary", alpha=alpha, beta=beta)


def bce_loss(pos_weight: Tensor | None = None):
    return smp.losses.SoftBCEWithLogitsLoss(pos_weight=pos_weight)


def bce_dice_loss(
    alpha: float = 0.5, beta: float = 0.5, pos_weight: Tensor | None = None
):
    bce = smp.losses.SoftBCEWithLogitsLoss(pos_weight=pos_weight)
    dice = smp.losses.DiceLoss(mode="binary")
    return lambda pred, target: alpha * bce(pred, target) + beta * dice(pred, target)


def bce_focal_loss(
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 2.0,
    pos_weight: Tensor | None = None,
):
    bce = smp.losses.SoftBCEWithLogitsLoss(pos_weight=pos_weight)
    focal = smp.losses.FocalLoss(mode="binary", gamma=gamma)
    return lambda pred, target: alpha * bce(pred, target) + beta * focal(pred, target)


def dice_focal_loss(alpha: float = 0.5, beta: float = 0.5, gamma: float = 2.0):
    dice = smp.losses.DiceLoss(mode="binary")
    focal = smp.losses.FocalLoss(mode="binary", gamma=gamma)
    return lambda pred, target: alpha * dice(pred, target) + beta * focal(pred, target)


def tversky_dice_loss(
    alpha: float = 0.5,
    beta: float = 0.5,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
):
    tversky = smp.losses.TverskyLoss(
        mode="binary", alpha=tversky_alpha, beta=tversky_beta
    )
    dice = smp.losses.DiceLoss(mode="binary")
    return lambda pred, target: alpha * tversky(pred, target) + beta * dice(
        pred, target
    )


def tversky_focal_loss(
    alpha: float = 0.5,
    beta: float = 0.5,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
    gamma: float = 2.0,
):
    tversky = smp.losses.TverskyLoss(
        mode="binary", alpha=tversky_alpha, beta=tversky_beta
    )
    focal = smp.losses.FocalLoss(mode="binary", gamma=gamma)
    return lambda pred, target: alpha * tversky(pred, target) + beta * focal(
        pred, target
    )
