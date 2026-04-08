"""
Boundary-Aware Loss Components
================================
Two complementary loss functions that target HD95 directly:

1. SURFACE LOSS (Kervadec et al., 2019)
   ----------------------------------------
   Computes the integral of the predicted softmax over the signed distance map
   of the ground truth boundary for each class.

   For each class c:
     phi_c = signed_distance_map(gt_c)
           = dist_edt(outside_gt_c) - dist_edt(inside_gt_c)
     L_surface_c = mean(softmax_c * phi_c)

   Minimizing this pushes the predicted class boundary to coincide with the
   ground truth boundary. This is the only loss that directly minimizes HD:
     - Positive phi → region outside GT (over-prediction penalized)
     - Negative phi → region inside GT (under-prediction encouraged to fill)
   As the predicted boundary converges to the GT boundary, phi at predicted
   boundary voxels approaches 0, driving the loss to zero.

   Reference: Kervadec et al., "Boundary loss for highly unbalanced
   segmentation", MIDL 2019.

2. BOUNDARY-WEIGHTED DICE LOSS
   ----------------------------------------
   Dice loss where each voxel's contribution is upweighted if it is near
   a class boundary. Boundary voxels (within `boundary_width` voxels of
   any class transition) receive weight `boundary_factor`, others weight 1.

   This gives the gradient more signal at ambiguous boundary regions,
   which directly improves boundary delineation and thus HD95.

Combined usage:
   loss = combined_loss_with_boundary(pred, target, ...)
   This adds the boundary losses on top of the existing Dice + CE + Focal.

Config fields:
  loss.boundary_weight:    0.4   (weight of surface loss in total)
  loss.boundary_factor:    5.0   (how much to upweight boundary voxels)
  loss.boundary_width:     2     (number of voxels defining "near boundary")
"""

import torch
import torch.nn.functional as F
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
#  SIGNED DISTANCE MAP
# ══════════════════════════════════════════════════════════════

def compute_signed_distance_map(seg_np, num_classes):
    """
    Compute signed distance maps for each class.

    Args:
        seg_np: np.ndarray [B, D, H, W] int64 — batch of segmentation masks
        num_classes: int

    Returns:
        np.ndarray [B, C, D, H, W] float32
          phi[b,c] = dist_edt(outside gt_c) - dist_edt(inside gt_c)
          Negative inside GT, positive outside GT.
          At the boundary itself: phi ≈ 0.
    """
    B, D, H, W = seg_np.shape
    phi = np.zeros((B, num_classes, D, H, W), dtype=np.float32)

    for b in range(B):
        for c in range(num_classes):
            gt_c = (seg_np[b] == c).astype(np.uint8)

            if gt_c.sum() == 0:
                # Class absent — full positive distance (all "outside")
                phi[b, c] = 1.0
                continue
            if gt_c.sum() == gt_c.size:
                # Class fills entire volume — full negative distance (all "inside")
                phi[b, c] = -1.0
                continue

            dist_out = distance_transform_edt(gt_c == 0).astype(np.float32)  # outside → boundary
            dist_in  = distance_transform_edt(gt_c == 1).astype(np.float32)  # inside  → boundary

            # Normalize by max value so loss scale is consistent across samples
            max_val = max(dist_out.max(), dist_in.max(), 1e-6)
            phi[b, c] = (dist_out - dist_in) / max_val

    return phi


# ══════════════════════════════════════════════════════════════
#  SURFACE LOSS
# ══════════════════════════════════════════════════════════════

def surface_loss(pred, target, class_weights, eps=1e-5):
    """
    Surface loss: integral of softmax over signed distance map.

    Args:
        pred:          [B, C, D, H, W] raw logits
        target:        [B, D, H, W] int64 labels
        class_weights: list[C] — per-class weights

    Returns:
        scalar loss tensor
    """
    if not SCIPY_AVAILABLE:
        return torch.tensor(0.0, device=pred.device, requires_grad=False)

    num_classes = pred.shape[1]
    pred_soft   = torch.softmax(pred, dim=1)       # [B, C, D, H, W]

    # Compute signed distance maps on CPU (scipy)
    seg_np = target.detach().cpu().numpy()
    phi_np = compute_signed_distance_map(seg_np, num_classes)  # [B, C, D, H, W]
    phi    = torch.tensor(phi_np, dtype=torch.float32, device=pred.device)

    weights = torch.tensor(class_weights, dtype=torch.float32, device=pred.device)

    # L = sum_c( w_c * mean(softmax_c * phi_c) )
    # phi is negative inside GT → pushes softmax to fill GT region
    # phi is positive outside  → penalizes over-prediction
    loss = 0.0
    for c in range(num_classes):
        loss = loss + weights[c] * (pred_soft[:, c] * phi[:, c]).mean()

    return loss / (weights.sum() + eps)


# ══════════════════════════════════════════════════════════════
#  BOUNDARY-WEIGHTED DICE LOSS
# ══════════════════════════════════════════════════════════════

def boundary_weight_map(target, num_classes, boundary_width=2, boundary_factor=5.0):
    """
    Compute a per-voxel weight map that upweights voxels near class boundaries.

    Strategy:
      - Dilate each binary class mask by `boundary_width` voxels using max-pool
      - Boundary region = dilated_mask XOR original_mask (ring of width `boundary_width`)
      - Weight map = 1.0 everywhere, `boundary_factor` at boundary voxels

    Args:
        target:          [B, D, H, W] int64
        num_classes:     int
        boundary_width:  int — dilation radius in voxels
        boundary_factor: float — weight multiplier for boundary voxels

    Returns:
        weight_map: [B, D, H, W] float32
    """
    B, D, H, W = target.shape
    weight_map = torch.ones_like(target, dtype=torch.float32)

    # Kernel for 3D max-pool dilation (approximates morphological dilation)
    k = 2 * boundary_width + 1

    for c in range(1, num_classes):   # skip background
        mask = (target == c).float().unsqueeze(1)    # [B, 1, D, H, W]

        # Dilate: max-pool ≡ binary dilation for 0/1 masks
        dilated = F.max_pool3d(
            mask,
            kernel_size=k,
            stride=1,
            padding=boundary_width,
        )   # [B, 1, D, H, W]

        # Boundary ring = dilated XOR original (symmetric difference)
        boundary = (dilated - mask).squeeze(1).clamp(0, 1)   # [B, D, H, W]
        weight_map = weight_map + boundary * (boundary_factor - 1.0)

    return weight_map


def boundary_weighted_dice_loss(pred, target, class_weights,
                                 boundary_width=2, boundary_factor=5.0, eps=1e-5):
    """
    Dice loss with boundary-region upweighting.

    Standard Dice treats all voxels equally. This version assigns higher
    weight to boundary voxels so gradients concentrate on ambiguous edges.

    Args:
        pred:            [B, C, D, H, W] raw logits
        target:          [B, D, H, W] int64
        class_weights:   list[C]
        boundary_width:  voxel radius defining the boundary region
        boundary_factor: upweight multiplier for boundary voxels

    Returns:
        scalar loss tensor
    """
    num_classes = pred.shape[1]
    pred_soft   = torch.softmax(pred, dim=1)

    target_oh = F.one_hot(target.long(), num_classes=num_classes)
    target_oh = target_oh.permute(0, 4, 1, 2, 3).float()       # [B, C, D, H, W]

    w_map = boundary_weight_map(
        target, num_classes, boundary_width, boundary_factor
    ).unsqueeze(1)   # [B, 1, D, H, W]

    weights = torch.tensor(class_weights, dtype=torch.float32, device=pred.device)

    total = 0.0
    for c in range(num_classes):
        p = pred_soft[:, c] * w_map[:, 0]
        t = target_oh[:, c] * w_map[:, 0]
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dice_c = 1.0 - (2.0 * inter + eps) / (union + eps)
        total  = total + weights[c] * dice_c

    return total / (weights.sum() + eps)


# ══════════════════════════════════════════════════════════════
#  COMBINED LOSS WITH BOUNDARY
# ══════════════════════════════════════════════════════════════

def combined_loss_with_boundary(pred, target,
                                 class_weights=None,
                                 focal_weight=0.5,
                                 boundary_weight=0.4,
                                 boundary_factor=5.0,
                                 boundary_width=2):
    """
    Full combined loss: Dice + CE + Focal + BoundaryDice + SurfaceLoss

    Augments the existing combined_loss with two boundary-specific terms:
      L_total = L_dice_w + L_CE_w + focal_w * L_focal
              + boundary_w * (L_boundary_dice + L_surface)

    The boundary components are only active when their weight > 0 and
    scipy is available. If scipy is missing, falls back to combined_loss.

    Args:
        pred:             [B, C, D, H, W] raw logits
        target:           [B, D, H, W]    int64 labels
        class_weights:    list[C]  (default BraTS-tuned)
        focal_weight:     float    weight on focal loss component
        boundary_weight:  float    weight on boundary loss components
        boundary_factor:  float    upweight for boundary voxels in boundary Dice
        boundary_width:   int      voxel radius for boundary region

    Returns:
        (total_loss_tensor, components_dict)
    """
    from src.boilerplates.losses.weighted_dice_focal_ce import combined_loss

    if class_weights is None:
        class_weights = [0.1, 3.0, 1.0, 2.0]

    # ── Base loss (Dice + CE + Focal) ─────────────────────────
    base_loss, components = combined_loss(
        pred, target,
        class_weights=class_weights,
        focal_weight=focal_weight,
    )

    if boundary_weight <= 0.0:
        return base_loss, components

    # ── Boundary-weighted Dice ────────────────────────────────
    l_bdice = boundary_weighted_dice_loss(
        pred, target, class_weights,
        boundary_width=boundary_width,
        boundary_factor=boundary_factor,
    )

    # ── Surface loss (requires scipy) ─────────────────────────
    if SCIPY_AVAILABLE:
        l_surface = surface_loss(pred, target, class_weights)
        l_boundary = (l_bdice + l_surface) * 0.5   # equal weight between the two
    else:
        l_boundary = l_bdice

    total = base_loss + boundary_weight * l_boundary

    components["loss_boundary_dice"]  = l_bdice.item()
    components["loss_surface"]        = l_surface.item() if SCIPY_AVAILABLE else 0.0
    components["loss_total"]          = total.item()

    return total, components
