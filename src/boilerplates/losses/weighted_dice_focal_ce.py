"""
Combined Loss: Weighted Dice + Weighted CE + Focal Loss
========================================================
Designed for class-imbalanced 3D medical segmentation (BraTS).

BraTS approximate voxel distribution:
  Class 0 - Background  : ~97%  → weight 0.1  (suppress dominant class)
  Class 1 - TC (Necrotic): ~1%  → weight 3.0  (main bottleneck)
  Class 2 - ED (Edema)   : ~2%  → weight 1.0
  Class 3 - ET (Enhancing): ~0.5% → weight 2.0

Formula:
  Loss = WeightedDice + WeightedCE + focal_weight * FocalLoss

Usage:
  loss = combined_loss(pred, target, class_weights=[0.1, 3.0, 1.0, 2.0])
"""

import torch
import torch.nn.functional as F


# Default BraTS class weights [BG, TC, ED, ET]
DEFAULT_CLASS_WEIGHTS = [0.1, 3.0, 1.0, 2.0]


# ==============================================================
# WEIGHTED DICE LOSS
# ==============================================================
def weighted_dice_loss(pred, target, class_weights, eps=1e-5):
    """
    Per-class Dice loss with class-level weighting.
    Penalises poor performance on minority classes (TC) more heavily.

    Args:
        pred   : [B, C, D, H, W]  raw logits
        target : [B, D, H, W]     integer class labels
        class_weights : list of C floats
    """
    num_classes = pred.shape[1]
    pred_soft = torch.softmax(pred, dim=1)                          # [B, C, D, H, W]

    target_oh = F.one_hot(target.long(), num_classes=num_classes)   # [B, D, H, W, C]
    target_oh = target_oh.permute(0, 4, 1, 2, 3).float()           # [B, C, D, H, W]

    weights = torch.tensor(class_weights, device=pred.device, dtype=torch.float32)

    total = 0.0
    for c in range(num_classes):
        p = pred_soft[:, c]
        t = target_oh[:, c]
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        dice_c = 1.0 - (2.0 * intersection + eps) / (union + eps)
        total += weights[c] * dice_c

    return total / weights.sum()


# ==============================================================
# WEIGHTED CROSS ENTROPY LOSS
# ==============================================================
def weighted_ce_loss(pred, target, class_weights):
    """
    Standard cross entropy with per-class weights.
    Handles voxel-level misclassification on hard classes.

    Args:
        pred   : [B, C, D, H, W]  raw logits
        target : [B, D, H, W]     integer class labels
    """
    weights = torch.tensor(class_weights, device=pred.device, dtype=torch.float32)
    return F.cross_entropy(pred, target.long(), weight=weights)


# ==============================================================
# FOCAL LOSS
# ==============================================================
def focal_loss(pred, target, class_weights, gamma=2.0):
    """
    Focal loss: down-weights easy voxels, focuses gradient on hard voxels.
    Critical for TC which has many uncertain boundary voxels.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        pred   : [B, C, D, H, W]  raw logits
        target : [B, D, H, W]     integer class labels
        gamma  : focusing parameter (2.0 is standard)
    """
    num_classes = pred.shape[1]

    # Alpha weights per class (re-normalised class weights)
    alpha = torch.tensor(class_weights, device=pred.device, dtype=torch.float32)
    alpha = alpha / alpha.sum()

    log_prob = F.log_softmax(pred, dim=1)   # [B, C, D, H, W]

    # Gather log-prob and prob for the true class at each voxel
    target_idx = target.long().unsqueeze(1)                    # [B, 1, D, H, W]
    log_p_t = log_prob.gather(1, target_idx).squeeze(1)        # [B, D, H, W]
    p_t = log_p_t.exp()                                        # [B, D, H, W]

    # Alpha for each voxel based on its true class
    alpha_t = alpha[target.long()]                             # [B, D, H, W]

    focal = -alpha_t * ((1.0 - p_t) ** gamma) * log_p_t
    return focal.mean()


# ==============================================================
# COMBINED LOSS
# ==============================================================
def combined_loss(pred, target, class_weights=None, focal_weight=0.5):
    """
    Full combined loss for training.

    Args:
        pred          : [B, C, D, H, W]  raw logits
        target        : [B, D, H, W]     integer class labels
        class_weights : list of C floats (default: BraTS-tuned)
        focal_weight  : weight on focal component (default 0.5)

    Returns:
        Scalar loss tensor.
        Also returns component dict for logging.
    """
    if class_weights is None:
        class_weights = DEFAULT_CLASS_WEIGHTS

    l_dice  = weighted_dice_loss(pred, target, class_weights)
    l_ce    = weighted_ce_loss(pred, target, class_weights)
    l_focal = focal_loss(pred, target, class_weights)

    total = l_dice + l_ce + focal_weight * l_focal

    components = {
        "loss_dice":  l_dice.item(),
        "loss_ce":    l_ce.item(),
        "loss_focal": l_focal.item(),
        "loss_total": total.item(),
    }

    return total, components
