import torch
import numpy as np
from scipy.spatial.distance import cdist


# ======================
# DICE SCORE (SAFE)
# ======================
def dice_score(pred, target, num_classes=4, eps=1e-5):
    """
    pred: (B, C, D, H, W)
    target: (B, D, H, W)
    """

    pred = torch.argmax(pred, dim=1)  # (B, D, H, W)

    dices = []

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2 * intersection + eps) / (union + eps)
        dices.append(dice.item())

    return dices


# ======================
# HD95 (SAFE + FIXED)
# ======================
def hausdorff_distance_95(pred, target):
    """
    pred, target: (D, H, W)
    """

    # 🔥 IMPORTANT FIX
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)

    # Edge case
    if len(pred_points) == 0 or len(target_points) == 0:
        return 0.0

    # ⚠️ Avoid huge memory
    if len(pred_points) > 10000:
        pred_points = pred_points[np.random.choice(len(pred_points), 10000, replace=False)]
    if len(target_points) > 10000:
        target_points = target_points[np.random.choice(len(target_points), 10000, replace=False)]

    distances = cdist(pred_points, target_points)

    hd95_1 = np.percentile(np.min(distances, axis=1), 95)
    hd95_2 = np.percentile(np.min(distances, axis=0), 95)

    return float(max(hd95_1, hd95_2))


# ======================
# SENSITIVITY (FIXED)
# ======================
def sensitivity(pred, target, num_classes=4, eps=1e-5):
    """
    Macro-averaged sensitivity (recall) across all classes.
    pred:   (B, D, H, W) — argmax integer class labels
    target: (B, D, H, W) — ground truth integer class labels

    For each class c, binary TP/FN are computed via one-vs-rest,
    then averaged across classes. This ensures the result is in [0, 1].
    """
    sens_per_class = []

    for c in range(num_classes):
        pred_c   = (pred == c).float()
        target_c = (target == c).float()

        tp = (pred_c * target_c).sum()
        fn = ((1 - pred_c) * target_c).sum()

        sens_per_class.append(((tp + eps) / (tp + fn + eps)).item())

    return sum(sens_per_class) / len(sens_per_class)


# ======================
# SPECIFICITY (FIXED)
# ======================
def specificity(pred, target, num_classes=4, eps=1e-5):
    """
    Macro-averaged specificity across all classes.
    pred:   (B, D, H, W) — argmax integer class labels
    target: (B, D, H, W) — ground truth integer class labels

    For each class c, binary TN/FP are computed via one-vs-rest,
    then averaged across classes. This ensures the result is in [0, 1].
    """
    spec_per_class = []

    for c in range(num_classes):
        pred_c   = (pred == c).float()
        target_c = (target == c).float()

        tn = ((1 - pred_c) * (1 - target_c)).sum()
        fp = (pred_c   * (1 - target_c)).sum()

        spec_per_class.append(((tn + eps) / (tn + fp + eps)).item())

    return sum(spec_per_class) / len(spec_per_class)