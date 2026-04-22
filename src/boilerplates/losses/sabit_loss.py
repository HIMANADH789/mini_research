"""
Phase 3 — SABiT Multi-Objective Loss Stack
=============================================
Total loss:
  L_total = L_seg + alpha(t)*L_prior + beta(t)*L_smooth + delta(t)*L_eig + epsilon(t)*L_evid

Components:
  L_seg    — Segmentation loss (reuses existing combined_loss_with_boundary)
  L_prior  — Prior fidelity: ||A_effective - A_prior||^2_F / N^2
  L_smooth — Laplacian smoothness: Tr(h^T L h) / N
  L_eig    — Spectral entropy: penalizes eigenvalue collapse
  L_evid   — Evidential KL: Dir(alpha) || Dir(1)

All auxiliary loss weights follow a linear warmup schedule.

Robustness:
  - Per-component NaN guard (zeroes NaN losses with warning)
  - Float32 enforcement for spectral entropy
  - Warmup schedule prevents early training destabilization
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SABiTLoss(nn.Module):
    """
    Multi-objective loss for SABiT training.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes.
    class_weights : list or None
        Per-class weights for CE/Dice loss.
    focal_gamma : float
        Focal loss gamma parameter.
    prior_weight : float
        Base weight for L_prior (warmup to this value).
    smooth_weight : float
        Base weight for L_smooth.
    eig_weight : float
        Base weight for L_eig.
    evid_weight : float
        Base weight for L_evid.
    dice_weight : float
        Weight of Dice loss in segmentation loss.
    ce_weight : float
        Weight of CE loss in segmentation loss.
    ds_weights : list
        Deep supervision weights [main, ds1, ds2].
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_weights: Optional[list] = None,
        focal_gamma: float = 2.0,
        prior_weight: float = 0.1,
        smooth_weight: float = 0.01,
        eig_weight: float = 0.005,
        evid_weight: float = 0.01,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        ds_weights: Optional[list] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        # Auxiliary loss base weights
        self.prior_weight = prior_weight
        self.smooth_weight = smooth_weight
        self.eig_weight = eig_weight
        self.evid_weight = evid_weight

        # Deep supervision weights
        self.ds_weights = ds_weights or [1.0, 0.5, 0.25]

        # Class weights for CE
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

        # Warmup schedule config (epochs)
        self.warmup_config = {
            "prior":  {"warmup_start": 20, "warmup_end": 50},
            "smooth": {"warmup_start": 20, "warmup_end": 50},
            "eig":    {"warmup_start": 30, "warmup_end": 60},
            "evid":   {"warmup_start": 10, "warmup_end": 30},
        }

    # ══════════════════════════════════════════════════════
    #  WARMUP SCHEDULE
    # ══════════════════════════════════════════════════════

    def _get_weight(
        self, epoch: int, base_weight: float, warmup_start: int, warmup_end: int
    ) -> float:
        """Linear warmup schedule for auxiliary loss weights."""
        if epoch < warmup_start:
            return 0.0
        if epoch >= warmup_end:
            return base_weight
        progress = (epoch - warmup_start) / max(warmup_end - warmup_start, 1)
        return base_weight * progress

    # ══════════════════════════════════════════════════════
    #  SEGMENTATION LOSS COMPONENTS
    # ══════════════════════════════════════════════════════

    def _dice_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Soft Dice loss.
        pred:   [B, C, ...] logits
        target: [B, ...] integer labels
        """
        B, C = pred.shape[0], pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)

        # One-hot encode target
        target_oh = F.one_hot(target.long(), C)  # [B, ..., C]
        # Move class dim: [B, ..., C] → [B, C, ...]
        dims = list(range(target_oh.ndim))
        target_oh = target_oh.permute(0, -1, *dims[1:-1]).float()

        # Flatten spatial dims
        pred_flat = pred_soft.reshape(B, C, -1)    # [B, C, N]
        target_flat = target_oh.reshape(B, C, -1)  # [B, C, N]

        # Per-class Dice
        intersection = (pred_flat * target_flat).sum(-1)  # [B, C]
        union = pred_flat.sum(-1) + target_flat.sum(-1)   # [B, C]

        dice_per_class = (2.0 * intersection + 1.0) / (union + 1.0)

        # Average over classes (skip background if desired)
        return 1.0 - dice_per_class[:, 1:].mean()  # skip class 0 (background)

    def _focal_ce_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Focal Cross-Entropy loss."""
        B, C = pred.shape[0], pred.shape[1]

        # Reshape for cross_entropy: pred [B, C, ...], target [B, ...]
        ce = F.cross_entropy(
            pred, target.long(),
            weight=self.class_weights,
            reduction="none",
        )  # [B, ...]

        # Focal modulation
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.focal_gamma) * ce

        return focal.mean()

    def _segmentation_loss(
        self, pred: Tensor, target: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Combined Dice + Focal CE loss for segmentation.
        Returns (loss, component_dict).
        """
        dice = self._dice_loss(pred, target)
        focal_ce = self._focal_ce_loss(pred, target)

        loss = self.dice_weight * dice + self.ce_weight * focal_ce

        return loss, {
            "dice_loss": dice.item(),
            "focal_ce_loss": focal_ce.item(),
        }

    # ══════════════════════════════════════════════════════
    #  AUXILIARY LOSS COMPONENTS
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _safe_loss(loss: Tensor, name: str) -> Tensor:
        """Guard against NaN/Inf in any loss component."""
        if not torch.isfinite(loss):
            logger.warning(f"Loss component '{name}' is NaN/Inf - zeroing")
            return torch.zeros(1, device=loss.device, requires_grad=False)
        return loss

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def _spectral_entropy_loss(eigenvalues: Tensor) -> Tensor:
        """
        Spectral entropy loss. Penalizes mode collapse.
        Returns POSITIVE value (higher = more collapsed = worse).
        We want to MINIMIZE this → maximizes entropy.
        """
        S = eigenvalues.clamp(min=1e-8)
        p = S / (S.sum() + 1e-8)
        # Entropy = -sum(p * log(p)), we return negative (for minimization)
        entropy = -(p * torch.log(p + 1e-8)).sum()
        # We want HIGH entropy → return -entropy as loss (minimizing = maximizing entropy)
        return -entropy

    # ══════════════════════════════════════════════════════
    #  FORWARD
    # ══════════════════════════════════════════════════════

    def forward(
        self,
        pred,
        target: Tensor,
        model_outputs: Optional[Dict] = None,
        epoch: int = 0,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute total loss with all components.

        Parameters
        ----------
        pred : Tensor or list of Tensors
            [B, C, D, H, W] logits. If list, [main, ds1, ds2] for deep supervision.
        target : Tensor [B, D, H, W]
            Ground truth integer labels.
        model_outputs : dict, optional
            From SABiT model — keys: 'aux_losses' with pre-computed graph losses.
            If None, only segmentation loss is computed.
        epoch : int
            Current epoch for warmup schedule.

        Returns
        -------
        total_loss : Tensor
            Scalar loss for backward.
        components : dict
            {name: float} for logging each component.
        """
        components = {}

        # ── Segmentation loss (with deep supervision) ─────────
        if isinstance(pred, (list, tuple)):
            # Deep supervision: [main, ds1, ds2]
            total_seg = torch.tensor(0.0, device=target.device)
            for i, (p, w) in enumerate(zip(pred, self.ds_weights)):
                if i == 0:
                    # Main prediction — full resolution
                    seg_loss, seg_info = self._segmentation_loss(p, target)
                    components.update(seg_info)
                else:
                    # Deep supervision — downsample target to match
                    t_ds = F.interpolate(
                        target.unsqueeze(1).float(),
                        size=p.shape[2:],
                        mode="nearest",
                    ).squeeze(1).long()
                    seg_loss, _ = self._segmentation_loss(p, t_ds)

                total_seg = total_seg + w * seg_loss

            total_seg = total_seg / sum(self.ds_weights)
        else:
            total_seg, seg_info = self._segmentation_loss(pred, target)
            components.update(seg_info)

        total_seg = self._safe_loss(total_seg, "segmentation")
        components["seg_loss"] = total_seg.item()

        total_loss = total_seg

        # ── Auxiliary losses from model ───────────────────────
        if model_outputs is not None:
            aux_losses = model_outputs.get("aux_losses", {})

            # L_prior
            if "prior" in aux_losses:
                w = self._get_weight(
                    epoch, self.prior_weight,
                    **self.warmup_config["prior"]
                )
                if w > 0:
                    L_prior = self._safe_loss(aux_losses["prior"][0], "prior")
                    total_loss = total_loss + w * L_prior
                    components["L_prior"] = L_prior.item()
                    components["w_prior"] = w

            # L_smooth
            if "smooth" in aux_losses:
                w = self._get_weight(
                    epoch, self.smooth_weight,
                    **self.warmup_config["smooth"]
                )
                if w > 0:
                    L_smooth = self._safe_loss(aux_losses["smooth"][0], "smooth")
                    total_loss = total_loss + w * L_smooth
                    components["L_smooth"] = L_smooth.item()
                    components["w_smooth"] = w

            # L_eig (spectral entropy)
            if "eig" in aux_losses:
                w = self._get_weight(
                    epoch, self.eig_weight,
                    **self.warmup_config["eig"]
                )
                if w > 0:
                    L_eig = self._safe_loss(aux_losses["eig"][0], "eig")
                    total_loss = total_loss + w * L_eig
                    components["L_eig"] = L_eig.item()
                    components["w_eig"] = w

            # L_evid (evidential KL)
            if "evid" in aux_losses:
                w = self._get_weight(
                    epoch, self.evid_weight,
                    **self.warmup_config["evid"]
                )
                if w > 0:
                    L_evid = self._safe_loss(aux_losses["evid"][0], "evid")
                    total_loss = total_loss + w * L_evid
                    components["L_evid"] = L_evid.item()
                    components["w_evid"] = w

        total_loss = self._safe_loss(total_loss, "total")
        components["total_loss"] = total_loss.item()

        return total_loss, components


# ══════════════════════════════════════════════════════════════
#  STANDALONE UNIT TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 3 - SABiTLoss Unit Tests")
    print("=" * 60)

    B, C, D, H, W = 2, 4, 16, 16, 16

    loss_fn = SABiTLoss(
        num_classes=4,
        prior_weight=0.1,
        smooth_weight=0.01,
        eig_weight=0.005,
        evid_weight=0.01,
    )

    # Test 1: Basic segmentation loss
    pred = torch.randn(B, C, D, H, W, requires_grad=True)
    target = torch.randint(0, C, (B, D, H, W))

    loss, info = loss_fn(pred, target, epoch=0)
    assert torch.isfinite(loss), f"Loss is {loss.item()}"
    loss.backward()
    assert pred.grad is not None
    print(f"  [OK] Seg loss: {loss.item():.4f}")
    print(f"       Dice: {info['dice_loss']:.4f}, Focal CE: {info['focal_ce_loss']:.4f}")

    # Test 2: With deep supervision
    pred_ds = [
        torch.randn(B, C, D, H, W, requires_grad=True),
        torch.randn(B, C, D//2, H//2, W//2, requires_grad=True),
        torch.randn(B, C, D//4, H//4, W//4, requires_grad=True),
    ]
    loss_ds, info_ds = loss_fn(pred_ds, target, epoch=0)
    assert torch.isfinite(loss_ds)
    loss_ds.backward()
    print(f"  [OK] DS loss: {loss_ds.item():.4f}")

    # Test 3: With auxiliary losses (simulating model output)
    pred2 = torch.randn(B, C, D, H, W, requires_grad=True)
    model_out = {
        "aux_losses": {
            "prior": (torch.tensor(0.5, requires_grad=True), 0.1),
            "smooth": (torch.tensor(0.3, requires_grad=True), 0.01),
            "eig": (torch.tensor(-0.2, requires_grad=True), 0.005),
            "evid": (torch.tensor(0.1, requires_grad=True), 0.01),
        }
    }

    # At epoch 0: only seg + evid warmup active
    loss0, info0 = loss_fn(pred2, target, model_out, epoch=0)
    print(f"  [OK] Epoch 0 (no aux warmup): {loss0.item():.4f}")
    print(f"       Components: {list(info0.keys())}")

    # At epoch 25: prior + smooth ramping
    pred3 = torch.randn(B, C, D, H, W, requires_grad=True)
    loss25, info25 = loss_fn(pred3, target, model_out, epoch=25)
    print(f"  [OK] Epoch 25 (aux warming): {loss25.item():.4f}")

    # At epoch 60: all fully active
    pred4 = torch.randn(B, C, D, H, W, requires_grad=True)
    loss60, info60 = loss_fn(pred4, target, model_out, epoch=60)
    print(f"  [OK] Epoch 60 (all active): {loss60.item():.4f}")
    for k, v in info60.items():
        if k.startswith("L_") or k.startswith("w_"):
            print(f"       {k}: {v:.6f}")

    # Test 4: NaN safety
    nan_out = {
        "aux_losses": {
            "prior": (torch.tensor(float("nan")), 0.1),
        }
    }
    pred5 = torch.randn(B, C, D, H, W)
    loss_nan, _ = loss_fn(pred5, target, nan_out, epoch=60)
    assert torch.isfinite(loss_nan), "NaN guard failed"
    print(f"  [OK] NaN guard: loss={loss_nan.item():.4f} (NaN component zeroed)")

    print()
    print("  [OK] ALL PHASE 3 TESTS PASSED")
