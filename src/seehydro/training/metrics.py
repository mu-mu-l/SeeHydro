"""评估指标."""

import numpy as np
import torch


def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict[int, float]:
    """计算每个类别的IoU.

    Args:
        pred: (H, W) 预测掩膜
        target: (H, W) 真值掩膜
        num_classes: 类别数

    Returns:
        {class_id: iou_value} 字典
    """
    ious = {}
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        ious[cls] = float(intersection / union) if union > 0 else float("nan")
    return ious


def compute_miou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    """计算mIoU（忽略NaN类别）."""
    ious = compute_iou(pred, target, num_classes)
    valid = [v for v in ious.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def compute_pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """计算像素精度."""
    return float((pred == target).sum() / target.size)


def compute_dice(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict[int, float]:
    """计算每个类别的Dice系数."""
    dices = {}
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = np.logical_and(pred_mask, target_mask).sum()
        total = pred_mask.sum() + target_mask.sum()
        dices[cls] = float(2 * intersection / total) if total > 0 else float("nan")
    return dices


class SegmentationMetrics:
    """分割评估指标累积器."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """更新混淆矩阵."""
        mask = (target >= 0) & (target < self.num_classes)
        indices = target[mask] * self.num_classes + pred[mask]
        counts = np.bincount(indices, minlength=self.num_classes**2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """计算所有指标."""
        cm = self.confusion_matrix
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection

        iou_per_class = np.where(union > 0, intersection / union, np.nan)
        valid_ious = iou_per_class[~np.isnan(iou_per_class)]

        return {
            "miou": float(np.mean(valid_ious)) if len(valid_ious) > 0 else 0.0,
            "iou_per_class": {i: float(v) for i, v in enumerate(iou_per_class)},
            "pixel_accuracy": float(intersection.sum() / cm.sum()) if cm.sum() > 0 else 0.0,
        }

    def reset(self) -> None:
        """重置混淆矩阵."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
