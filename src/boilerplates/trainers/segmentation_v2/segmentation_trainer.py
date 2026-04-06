"""
Segmentation Trainer v2
========================
Improvements over v1:
  - Deep Supervision support:
      If model.forward() returns a list (e.g. UNetPP_DS), loss is computed
      for each output and combined using configurable weights.
      Loss = sum(ds_weight[i] * loss(output[i])) for i in outputs
  - Gradient clipping (training.grad_clip):
      Prevents exploding gradients in deeper/transformer models.
      Default: 1.0  (set to 0 to disable)
  - LR Warmup (training.warmup_epochs):
      Linear warmup for N epochs before cosine annealing begins.
      Critical for stable training of deeper networks from scratch.
      Default: 5 epochs
  - AdamW support (optimizer.type: adamw):
      Decoupled weight decay for proper L2 regularization.
      Recommended for deeper networks and transformers.
  - Per-class Dice monitoring during training:
      Dice score computed on each training batch prediction and logged
      per epoch. Shows per-class convergence (especially TC).
  - Best checkpoint saving:
      Saves 'best.pth' whenever training loss improves.
      Periodic saves (every epoch) continue for experiment tracking.

Config fields consumed (beyond v1):
  training:
    warmup_epochs: 5          # linear warmup before cosine
    grad_clip: 1.0            # max gradient norm (0 = disabled)
    ds_weights: [1.0, 0.5, 0.25]   # deep supervision weights (per output)
  optimizer:
    type: adam | adamw        # optimizer choice
    weight_decay: 0.01        # used if type=adamw
  loss:
    class_weights: [0.1, 3.0, 1.0, 2.0]
    focal_weight: 0.5
"""

import os
import math
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from src.boilerplates.model_builder.build import build_model
from src.boilerplates.resolver import build_dataloader
from src.boilerplates.losses.weighted_dice_focal_ce import combined_loss

from src.utils.experiment_utils.device import get_device
from src.utils.experiment_utils.io import save_model


# ──────────────────────────────────────────────────────────
# PER-CLASS DICE ON BATCH (for training monitoring)
# ──────────────────────────────────────────────────────────

def batch_dice_per_class(pred_logits, target, num_classes=4, eps=1e-5):
    """
    Quick per-class Dice on a training batch (argmax prediction).
    Returns list of floats length=num_classes.
    """
    pred = torch.argmax(pred_logits, dim=1)  # [B, D, H, W]
    dices = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        denom = p.sum() + t.sum()
        dices.append(((2 * intersection + eps) / (denom + eps)).item())
    return dices


# ──────────────────────────────────────────────────────────
# TRAINER
# ──────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, config, exp_path, logger):
        self.config   = config
        self.logger   = logger
        self.exp_path = exp_path

        # DEVICE
        self.device = get_device(config)

        # MODEL
        self.model = build_model(config).to(self.device)

        # DATA
        self.loader = build_dataloader(config, split="train")

        # LOSS CONFIG
        loss_cfg = getattr(config, "loss", None)
        self.class_weights = getattr(loss_cfg, "class_weights", [0.1, 3.0, 1.0, 2.0])
        self.focal_weight  = float(getattr(loss_cfg, "focal_weight", 0.5))

        # DEEP SUPERVISION WEIGHTS
        # Applied to [main, aux1, aux2, ...] outputs if model returns a list
        train_cfg = config.training
        raw_ds    = getattr(train_cfg, "ds_weights", [1.0, 0.5, 0.25])
        self.ds_weights = list(raw_ds) if hasattr(raw_ds, '__iter__') else [1.0, 0.5, 0.25]

        # TRAINING HYPERPARAMS
        self.grad_clip     = float(getattr(train_cfg, "grad_clip",     1.0))
        self.warmup_epochs = int(  getattr(train_cfg, "warmup_epochs", 5))
        total_epochs       = int(config.training.epochs)
        lr                 = float(config.training.lr)
        lr_min             = float(getattr(train_cfg, "lr_min", 1e-6))

        # OPTIMIZER
        opt_cfg      = getattr(config, "optimizer", None)
        opt_type     = getattr(opt_cfg, "type", "adam").lower()
        weight_decay = float(getattr(opt_cfg, "weight_decay", 1e-4))

        if opt_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr
            )

        # LR SCHEDULE: Linear Warmup → Cosine Annealing
        cosine_epochs = max(1, total_epochs - self.warmup_epochs)

        if self.warmup_epochs > 0:
            warmup_sched = LinearLR(
                self.optimizer,
                start_factor=1e-6 / lr,   # start at near-zero LR
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            cosine_sched = CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs,
                eta_min=lr_min,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[self.warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=total_epochs, eta_min=lr_min
            )

        # AMP
        self.use_amp = getattr(train_cfg, "mixed_precision", True)
        self.scaler  = GradScaler(enabled=self.use_amp)

        # BEST LOSS TRACKING
        self.best_loss = math.inf

    # ──────────────────────────────────────────────────────
    # COMPUTE LOSS (handles single output or DS list)
    # ──────────────────────────────────────────────────────

    def compute_loss(self, outputs, seg):
        """
        Args:
            outputs: Tensor [B,C,D,H,W]  OR  list of Tensors (deep supervision)
            seg:     Tensor [B,D,H,W]

        Returns:
            total_loss: scalar Tensor
            components: dict with loss_total, loss_dice, loss_ce, loss_focal
        """
        if isinstance(outputs, list):
            # Deep supervision: weighted sum across all output levels
            total_loss = None
            agg = {"loss_dice": 0.0, "loss_ce": 0.0, "loss_focal": 0.0}

            for i, out in enumerate(outputs):
                w = self.ds_weights[i] if i < len(self.ds_weights) else self.ds_weights[-1]
                loss_i, comp_i = combined_loss(
                    out, seg,
                    class_weights=self.class_weights,
                    focal_weight=self.focal_weight,
                )
                weighted_loss = w * loss_i
                total_loss    = weighted_loss if total_loss is None else total_loss + weighted_loss
                agg["loss_dice"]  += w * comp_i["loss_dice"]
                agg["loss_ce"]    += w * comp_i["loss_ce"]
                agg["loss_focal"] += w * comp_i["loss_focal"]

            # Normalize component logging by sum of weights
            w_sum = sum(self.ds_weights[:len(outputs)])
            components = {
                "loss_total": total_loss.item(),
                "loss_dice":  agg["loss_dice"]  / w_sum,
                "loss_ce":    agg["loss_ce"]    / w_sum,
                "loss_focal": agg["loss_focal"] / w_sum,
            }
        else:
            total_loss, components = combined_loss(
                outputs, seg,
                class_weights=self.class_weights,
                focal_weight=self.focal_weight,
            )

        return total_loss, components

    # ──────────────────────────────────────────────────────
    # TRAIN
    # ──────────────────────────────────────────────────────

    def train(self):
        self.model.train()

        num_classes = getattr(self.config.model, "out_channels", 4)
        total_epochs = self.config.training.epochs

        for epoch in range(total_epochs):

            total_loss  = 0.0
            total_dice  = 0.0
            total_ce    = 0.0
            total_focal = 0.0
            epoch_dice_per_class = [0.0] * num_classes

            for img, seg in self.loader:
                img = img.to(self.device, non_blocking=True)
                seg = seg.to(self.device, non_blocking=True)

                with autocast(device_type="cuda", enabled=self.use_amp):
                    outputs = self.model(img)
                    loss, components = self.compute_loss(outputs, seg)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping (unscale first for correct norm computation)
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss  += components["loss_total"]
                total_dice  += components["loss_dice"]
                total_ce    += components["loss_ce"]
                total_focal += components["loss_focal"]

                # Per-class Dice monitoring (use main output)
                main_out = outputs[0] if isinstance(outputs, list) else outputs
                with torch.no_grad():
                    batch_dices = batch_dice_per_class(main_out, seg, num_classes)
                for c in range(num_classes):
                    epoch_dice_per_class[c] += batch_dices[c]

            n  = len(self.loader)
            current_lr = self.optimizer.param_groups[0]["lr"]
            avg_loss   = total_loss  / n
            avg_dice_per_class = [d / n for d in epoch_dice_per_class]

            # Format per-class Dice string: BG | TC | ED | ET
            dice_str = " | ".join(
                f"C{c}:{avg_dice_per_class[c]:.3f}" for c in range(num_classes)
            )

            self.logger.info(
                f"Epoch {epoch:03d}/{total_epochs} | "
                f"LR: {current_lr:.2e} | "
                f"Loss: {avg_loss:.4f} | "
                f"Dice: {total_dice/n:.4f} | "
                f"CE: {total_ce/n:.4f} | "
                f"Focal: {total_focal/n:.4f} | "
                f"PerClass [{dice_str}]"
            )

            # STEP SCHEDULER
            self.scheduler.step()

            # SAVE CHECKPOINT (every epoch)
            ckpt_path = os.path.join(
                self.exp_path, "checkpoints", f"epoch_{epoch}.pth"
            )
            save_model(self.model, ckpt_path)

            # SAVE BEST CHECKPOINT
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                best_path = os.path.join(self.exp_path, "checkpoints", "best.pth")
                save_model(self.model, best_path)
                self.logger.info(f"  → Best checkpoint updated (loss: {avg_loss:.4f})")

        self.logger.info("Training Finished")
