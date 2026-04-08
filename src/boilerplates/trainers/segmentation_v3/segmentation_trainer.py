"""
Segmentation Trainer v3 — Transformer-Aware Training
======================================================
Improvements over v2:

  1. VALIDATION-BASED BEST CHECKPOINT
     Every `val_every` epochs, the trainer evaluates the model on the
     validation set using patch-based Dice (fast proxy, not sliding window).
     The best checkpoint is saved based on VALIDATION TC Dice — not training
     loss. This prevents saving an overtrained checkpoint.
     Config: training.val_every (default 10)

  2. ADAMW AS DEFAULT FOR TRANSFORMERS
     AdamW with decoupled weight decay is set as default when
     optimizer.type is not specified (or "adamw"). For CNN configs that
     specify "adam", the old behavior is preserved.
     Config: optimizer.type: adamw | adam  (default: adamw)
             optimizer.weight_decay: 0.01

  3. POLY LR DECAY OPTION
     Alternative to cosine annealing: polynomial decay (power=0.9),
     widely used in transformer segmentation papers (UNETR, SwinUNETR).
     Combines with linear warmup.
     Config: training.lr_policy: cosine | poly  (default: cosine)

  4. EXTENDED WARMUP
     Warmup default changed to 20 epochs (appropriate for transformers).
     Config: training.warmup_epochs: 20

  5. GRADIENT CLIPPING WITH NORM LOGGING
     Gradient norm is logged every epoch to detect instability early.
     Config: training.grad_clip: 1.0

  6. DEEP SUPERVISION (INHERITED from v2)
     Full DS support for any model returning a list from forward().
     Boundary loss is only applied to the main (full-resolution) output
     to avoid distorting the low-resolution auxiliary heads.
     Config: training.ds_weights: [1.0, 0.5, 0.25]

  7. EXPONENTIAL MOVING AVERAGE (EMA) — NEW
     Maintains a shadow model whose parameters are an exponential moving
     average of the training model. EMA weights are smoother and generally
     generalize better, especially for transformers.
       ema_param = decay * ema_param + (1 - decay) * model_param
     Validation uses the EMA model. best.pth saves the EMA checkpoint.
     The raw training model checkpoint is saved as epoch_N.pth / best_train.pth.
     Config: training.use_ema: true  (default: true)
             training.ema_decay: 0.9999

  8. BOUNDARY-AWARE LOSS — NEW
     Adds surface loss (Kervadec 2019) and boundary-weighted Dice to the
     training objective. These directly minimize HD95 by concentrating
     gradient signal on class boundary voxels.
       L = L_dice + L_CE + focal_w * L_focal + boundary_w * (L_bdice + L_surface)
     The boundary loss is applied only to the main output (full resolution).
     Disabled if boundary_weight = 0.0 or scipy is not available.
     Config: loss.boundary_weight:  0.4
             loss.boundary_factor:  5.0   (boundary voxel upweight)
             loss.boundary_width:   2     (voxel radius for boundary region)

Config fields consumed:
  training:
    epochs:          300
    batch_size:      1
    lr:              0.0001
    lr_min:          1.0e-6
    lr_policy:       cosine | poly
    mixed_precision: true
    warmup_epochs:   20
    grad_clip:       1.0
    val_every:       10
    ds_weights:      [1.0, 0.5, 0.25]
    use_ema:         true
    ema_decay:       0.9999
    layer_lr_decay:  1.0   (disabled by default)
  optimizer:
    type:         adamw
    weight_decay: 0.01
  loss:
    class_weights:    [0.1, 3.0, 1.0, 2.0]
    focal_weight:     0.5
    boundary_weight:  0.4
    boundary_factor:  5.0
    boundary_width:   2
"""

import os
import math
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from src.boilerplates.model_builder.build import build_model
from src.boilerplates.resolver import build_dataloader
from src.boilerplates.losses.weighted_dice_focal_ce import combined_loss
from src.boilerplates.losses.boundary_aware_loss import combined_loss_with_boundary

from src.utils.experiment_utils.device import get_device
from src.utils.experiment_utils.io import save_model


# ══════════════════════════════════════════════════════════════
#  MODEL EMA
# ══════════════════════════════════════════════════════════════

class ModelEMA:
    """
    Exponential Moving Average of model weights.

    After each optimizer step, EMA weights are updated:
      ema_p = decay * ema_p + (1 - decay) * model_p

    The EMA model is used for validation and final evaluation.
    It has no gradients and never participates in forward/backward
    of the training model — it is a read-only shadow.

    Decay = 0.9999: at 300 epochs × ~100 batches = 30,000 updates,
    the effective averaging window is 1/(1-0.9999) = 10,000 steps.
    """

    def __init__(self, model, decay=0.9999):
        self.decay     = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Update EMA weights after each optimizer step."""
        for ema_p, m_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)
        # Copy non-parameter buffers (e.g. attention mask caches, BN running stats)
        for ema_b, m_b in zip(self.ema_model.buffers(), model.buffers()):
            if ema_b.dtype.is_floating_point:
                ema_b.data.mul_(self.decay).add_(m_b.data, alpha=1.0 - self.decay)
            else:
                ema_b.data.copy_(m_b.data)

    def state_dict(self):
        return self.ema_model.state_dict()

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def batch_dice_per_class(pred_logits, target, num_classes=4, eps=1e-5):
    """Quick per-class Dice on a training/validation batch (argmax)."""
    pred = torch.argmax(pred_logits, dim=1)
    dices = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dices.append(((2 * inter + eps) / (denom + eps)).item())
    return dices


class PolyLR:
    """
    Polynomial LR decay scheduler.
    lr(epoch) = lr_base * (1 - epoch/total_epochs) ^ power
    """
    def __init__(self, optimizer, total_epochs, power=0.9, last_epoch=-1):
        self.optimizer    = optimizer
        self.total_epochs = total_epochs
        self.power        = power
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]
        self.last_epoch   = last_epoch

    def step(self):
        self.last_epoch += 1
        factor = max(0.0, 1.0 - self.last_epoch / self.total_epochs) ** self.power
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ══════════════════════════════════════════════════════════════
#  TRAINER v3
# ══════════════════════════════════════════════════════════════

class Trainer:

    def __init__(self, config, exp_path, logger):
        self.config   = config
        self.logger   = logger
        self.exp_path = exp_path

        self.device = get_device(config)

        # MODEL
        self.model = build_model(config).to(self.device)

        # DATA
        self.train_loader = build_dataloader(config, split="train")
        self.val_loader   = build_dataloader(config, split="val")

        # LOSS CONFIG
        loss_cfg = getattr(config, "loss", None)
        self.class_weights    = getattr(loss_cfg, "class_weights",   [0.1, 3.0, 1.0, 2.0])
        self.focal_weight     = float(getattr(loss_cfg, "focal_weight",    0.5))
        self.boundary_weight  = float(getattr(loss_cfg, "boundary_weight", 0.4))
        self.boundary_factor  = float(getattr(loss_cfg, "boundary_factor", 5.0))
        self.boundary_width   = int(  getattr(loss_cfg, "boundary_width",  2))

        # DEEP SUPERVISION WEIGHTS
        train_cfg = config.training
        raw_ds    = getattr(train_cfg, "ds_weights", [1.0, 0.5, 0.25])
        self.ds_weights = list(raw_ds) if hasattr(raw_ds, '__iter__') else [1.0, 0.5, 0.25]

        # HYPERPARAMS
        self.grad_clip     = float(getattr(train_cfg, "grad_clip",     1.0))
        self.warmup_epochs = int(  getattr(train_cfg, "warmup_epochs", 20))
        self.val_every     = int(  getattr(train_cfg, "val_every",     10))
        self.total_epochs  = int(config.training.epochs)
        self.num_classes   = getattr(config.model, "out_channels", 4)
        lr                 = float(config.training.lr)
        lr_min             = float(getattr(train_cfg, "lr_min",      1e-6))
        lr_policy          = str(  getattr(train_cfg, "lr_policy",   "cosine")).lower()

        # EMA
        use_ema        = bool(getattr(train_cfg, "use_ema",   True))
        ema_decay      = float(getattr(train_cfg, "ema_decay", 0.9999))
        self.ema       = ModelEMA(self.model, decay=ema_decay) if use_ema else None
        self.use_ema   = use_ema

        # OPTIMIZER
        opt_cfg      = getattr(config, "optimizer", None)
        opt_type     = getattr(opt_cfg, "type",         "adamw").lower()
        weight_decay = float(getattr(opt_cfg, "weight_decay", 0.01))

        layer_lr_decay = float(getattr(train_cfg, "layer_lr_decay", 1.0))
        if layer_lr_decay < 1.0 and hasattr(self.model, "get_param_groups"):
            param_groups = self.model.get_param_groups(lr, layer_lr_decay)
        else:
            param_groups = self.model.parameters()

        if opt_type == "adam":
            self.optimizer = torch.optim.Adam(param_groups, lr=lr)
        else:
            self.optimizer = torch.optim.AdamW(
                param_groups, lr=lr, weight_decay=weight_decay
            )

        # LR SCHEDULE: Linear Warmup → Cosine or Poly
        cosine_epochs = max(1, self.total_epochs - self.warmup_epochs)

        if self.warmup_epochs > 0:
            warmup_sched = LinearLR(
                self.optimizer,
                start_factor=1e-6 / lr,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            if lr_policy == "poly":
                self.scheduler      = warmup_sched
                self.main_scheduler = PolyLR(self.optimizer, total_epochs=cosine_epochs, power=0.9)
                self.use_poly       = True
            else:
                cosine_sched = CosineAnnealingLR(
                    self.optimizer, T_max=cosine_epochs, eta_min=lr_min
                )
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[self.warmup_epochs],
                )
                self.use_poly = False
        else:
            if lr_policy == "poly":
                self.scheduler      = PolyLR(self.optimizer, self.total_epochs, power=0.9)
                self.main_scheduler = None
                self.use_poly       = True
            else:
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=self.total_epochs, eta_min=lr_min
                )
                self.use_poly = False

        # AMP
        self.use_amp = getattr(train_cfg, "mixed_precision", True)
        self.scaler  = GradScaler(enabled=self.use_amp)

        # CHECKPOINT TRACKING
        self.best_train_loss = math.inf
        self.best_val_tc     = -1.0

        ema_str = f"EMA(decay={ema_decay})" if use_ema else "no EMA"
        bw_str  = f"boundary_weight={self.boundary_weight}" if self.boundary_weight > 0 else "no boundary loss"
        self.logger.info(
            f"Trainer v3 | {opt_type.upper()} | LR={lr_policy} | "
            f"warmup={self.warmup_epochs} | val_every={self.val_every} | "
            f"{ema_str} | {bw_str}"
        )

    # ──────────────────────────────────────────────────────
    # LOSS COMPUTATION
    # ──────────────────────────────────────────────────────

    def _loss_for_output(self, out, seg, apply_boundary=False):
        """
        Compute loss for a single output tensor.
        apply_boundary=True only for the main (full-resolution) output.
        """
        if apply_boundary and self.boundary_weight > 0:
            return combined_loss_with_boundary(
                out, seg,
                class_weights=self.class_weights,
                focal_weight=self.focal_weight,
                boundary_weight=self.boundary_weight,
                boundary_factor=self.boundary_factor,
                boundary_width=self.boundary_width,
            )
        return combined_loss(
            out, seg,
            class_weights=self.class_weights,
            focal_weight=self.focal_weight,
        )

    def compute_loss(self, outputs, seg):
        """
        Handles single output or list (deep supervision).
        Boundary loss applied only to main output (index 0, full resolution).
        DS auxiliary outputs use standard combined_loss.
        """
        if isinstance(outputs, list):
            total_loss = None
            agg = {"loss_dice": 0.0, "loss_ce": 0.0, "loss_focal": 0.0}

            for i, out in enumerate(outputs):
                w = self.ds_weights[i] if i < len(self.ds_weights) else self.ds_weights[-1]

                # Downsample target to match smaller DS output resolution
                if out.shape[2:] != seg.shape[1:]:
                    seg_i = F.interpolate(
                        seg.float().unsqueeze(1),
                        size=out.shape[2:],
                        mode="nearest",
                    ).squeeze(1).long()
                else:
                    seg_i = seg

                # Boundary loss only on main output (full resolution)
                loss_i, comp_i = self._loss_for_output(out, seg_i, apply_boundary=(i == 0))

                weighted   = w * loss_i
                total_loss = weighted if total_loss is None else total_loss + weighted
                agg["loss_dice"]  += w * comp_i["loss_dice"]
                agg["loss_ce"]    += w * comp_i["loss_ce"]
                agg["loss_focal"] += w * comp_i.get("loss_focal", 0.0)

            w_sum = sum(self.ds_weights[:len(outputs)])
            components = {
                "loss_total": total_loss.item(),
                "loss_dice":  agg["loss_dice"]  / w_sum,
                "loss_ce":    agg["loss_ce"]    / w_sum,
                "loss_focal": agg["loss_focal"] / w_sum,
            }
        else:
            total_loss, components = self._loss_for_output(
                outputs, seg, apply_boundary=True
            )

        return total_loss, components

    # ──────────────────────────────────────────────────────
    # VALIDATION (fast patch-based proxy using EMA model)
    # ──────────────────────────────────────────────────────

    def validate(self):
        """
        Patch-based validation using EMA model (if available) or training model.
        Returns (mean_dice, tc_dice, per_class_dices).
        """
        eval_model = self.ema.ema_model if self.use_ema else self.model
        eval_model.eval()
        total_dices = []
        eps = 1e-5

        with torch.no_grad():
            for img, seg in self.val_loader:
                B, C, D, H, W = img.shape
                pd = min(D, self.config.data.patch_size[0])
                ph = min(H, self.config.data.patch_size[1])
                pw = min(W, self.config.data.patch_size[2])
                d0 = (D - pd) // 2
                h0 = (H - ph) // 2
                w0 = (W - pw) // 2

                patch_img = img[:, :, d0:d0+pd, h0:h0+ph, w0:w0+pw].to(self.device)
                patch_seg = seg[:,    d0:d0+pd, h0:h0+ph, w0:w0+pw].to(self.device)

                outputs  = eval_model(patch_img)
                main_out = outputs[0] if isinstance(outputs, list) else outputs

                pred = torch.argmax(main_out, dim=1)
                dices = []
                for c in range(self.num_classes):
                    p = (pred == c).float()
                    t = (patch_seg == c).float()
                    inter = (p * t).sum()
                    denom = p.sum() + t.sum()
                    dices.append(((2 * inter + eps) / (denom + eps)).item())
                total_dices.append(dices)

        if not self.use_ema:
            self.model.train()

        avg       = [sum(d[c] for d in total_dices) / len(total_dices) for c in range(self.num_classes)]
        mean_dice = sum(avg) / len(avg)
        tc_dice   = avg[1]
        return mean_dice, tc_dice, avg

    # ──────────────────────────────────────────────────────
    # LR STEP
    # ──────────────────────────────────────────────────────

    def _step_scheduler(self, epoch):
        if self.use_poly:
            if epoch < self.warmup_epochs:
                self.scheduler.step()
            elif hasattr(self, "main_scheduler") and self.main_scheduler:
                self.main_scheduler.step()
        else:
            self.scheduler.step()

    # ──────────────────────────────────────────────────────
    # TRAIN
    # ──────────────────────────────────────────────────────

    def train(self):
        self.model.train()

        for epoch in range(self.total_epochs):

            total_loss   = 0.0
            total_dice   = 0.0
            total_ce     = 0.0
            total_focal  = 0.0
            total_bnd    = 0.0
            total_gn     = 0.0
            epoch_dpc    = [0.0] * self.num_classes
            n_batches    = 0

            for img, seg in self.train_loader:
                img = img.to(self.device, non_blocking=True)
                seg = seg.to(self.device, non_blocking=True)

                with autocast(device_type="cuda", enabled=self.use_amp):
                    outputs = self.model(img)
                    loss, components = self.compute_loss(outputs, seg)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                grad_norm = 0.0
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    ).item()

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # EMA update after each optimizer step
                if self.use_ema:
                    self.ema.update(self.model)

                total_loss  += components["loss_total"]
                total_dice  += components["loss_dice"]
                total_ce    += components["loss_ce"]
                total_focal += components.get("loss_focal", 0.0)
                total_bnd   += components.get("loss_boundary_dice", 0.0)
                total_gn    += grad_norm
                n_batches   += 1

                main_out = outputs[0] if isinstance(outputs, list) else outputs
                with torch.no_grad():
                    bd = batch_dice_per_class(main_out, seg, self.num_classes)
                for c in range(self.num_classes):
                    epoch_dpc[c] += bd[c]

            # ── LR step ──────────────────────────────────────
            self._step_scheduler(epoch)

            # ── Logging ──────────────────────────────────────
            lr_now   = self.optimizer.param_groups[0]["lr"]
            avg_loss = total_loss / n_batches
            avg_gn   = total_gn   / n_batches
            avg_bnd  = total_bnd  / n_batches
            avg_dpc  = [d / n_batches for d in epoch_dpc]
            dpc_str  = " | ".join(f"C{c}:{avg_dpc[c]:.3f}" for c in range(self.num_classes))

            self.logger.info(
                f"Epoch {epoch:03d}/{self.total_epochs} | LR: {lr_now:.2e} | "
                f"Loss: {avg_loss:.4f} | Dice: {total_dice/n_batches:.4f} | "
                f"CE: {total_ce/n_batches:.4f} | Focal: {total_focal/n_batches:.4f} | "
                f"Bnd: {avg_bnd:.4f} | GradNorm: {avg_gn:.3f} | PerClass [{dpc_str}]"
            )

            # ── Save raw model checkpoint ─────────────────────
            save_model(self.model, os.path.join(self.exp_path, "checkpoints", f"epoch_{epoch}.pth"))

            if avg_loss < self.best_train_loss:
                self.best_train_loss = avg_loss
                save_model(self.model, os.path.join(self.exp_path, "checkpoints", "best_train.pth"))

            # ── Validation + EMA checkpoint ───────────────────
            if (epoch + 1) % self.val_every == 0 or epoch == self.total_epochs - 1:
                val_mean, val_tc, val_dpc = self.validate()
                val_str  = " | ".join(f"C{c}:{val_dpc[c]:.3f}" for c in range(self.num_classes))
                ema_tag  = " [EMA]" if self.use_ema else ""
                self.logger.info(
                    f"  [VAL{ema_tag}] Epoch {epoch:03d} | MeanDice: {val_mean:.4f} | "
                    f"TC: {val_tc:.4f} | PerClass [{val_str}]"
                )

                if val_tc > self.best_val_tc:
                    self.best_val_tc = val_tc
                    # Save EMA weights as best.pth (used by evaluator)
                    best_path = os.path.join(self.exp_path, "checkpoints", "best.pth")
                    if self.use_ema:
                        torch.save(self.ema.state_dict(), best_path)
                    else:
                        save_model(self.model, best_path)
                    self.logger.info(
                        f"  -> best.pth updated (val TC: {val_tc:.4f})"
                    )

        self.logger.info("Training Finished")
