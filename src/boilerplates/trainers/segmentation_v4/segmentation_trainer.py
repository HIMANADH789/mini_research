"""
Segmentation Trainer v4 — Checkpoint-Aware Training
=====================================================
Extends Trainer v3 with:

  1. FULL CHECKPOINT SYSTEM (CheckpointManager integration)
     - Tier A: Full resume checkpoint every epoch
       (model, optimizer, scheduler, scaler, ema, RNG states, epoch, best metric)
     - Tier B: Best model checkpoint (EMA weights if available)
     - Tier C: Periodic snapshots with automatic retention (keep last N)
     - Tier D: training_state.json updated every epoch

  2. ROBUST RESUME SUPPORT
     - Call trainer.resume_from(path) before trainer.train()
     - Restores ALL training states: model, optimizer, scheduler,
       scaler, EMA, RNG — ensures fully reproducible continuation

  3. INTERRUPTION SAFETY
     - try/except around training loop
     - Emergency checkpoint saved on KeyboardInterrupt or any exception
     - training_state.json marked with interrupted_flag=True

  4. CONFIG-DRIVEN CHECKPOINT POLICY
     New config section (all optional, backward compatible):

     checkpoint:
       save_best: true               # save best_model.pt
       save_resume_every_epoch: true  # save resume checkpoint after each epoch
       save_periodic_every: 10       # save periodic snapshot every N epochs
       keep_last_n_periodic: 3       # retention: keep only last N periodic ckpts
       monitor: tc_dice              # metric to track for best_model
       mode: max                     # max | min

  5. INHERITS ALL v3 FEATURES
     - AdamW + cosine/poly LR + warmup
     - EMA with configurable decay
     - Boundary-aware loss
     - Deep supervision
     - Validation-based best checkpoint
     - Gradient norm logging
     - Per-class Dice logging

How to use:
    # In your experiment YAML:
    versions:
      trainer: segmentation_v4

    checkpoint:
      save_best: true
      save_resume_every_epoch: true
      save_periodic_every: 10
      keep_last_n_periodic: 3
      monitor: tc_dice
      mode: max
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
from src.utils.checkpoint_utils import CheckpointManager


# ══════════════════════════════════════════════════════════════
#  MODEL EMA  (identical to v3, copied to keep v3 untouched)
# ══════════════════════════════════════════════════════════════

class ModelEMA:
    """
    Exponential Moving Average of model weights.
    EMA model is used for validation and saved as best_model.pt.
    """

    def __init__(self, model, decay=0.9999):
        self.decay     = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, m_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)
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
#  HELPERS  (identical to v3)
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
    """Polynomial LR decay scheduler."""
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

    def state_dict(self):
        return {
            "last_epoch":   self.last_epoch,
            "total_epochs": self.total_epochs,
            "power":        self.power,
            "base_lrs":     self.base_lrs,
        }

    def load_state_dict(self, d):
        self.last_epoch   = d.get("last_epoch", self.last_epoch)
        self.total_epochs = d.get("total_epochs", self.total_epochs)
        self.power        = d.get("power", self.power)
        self.base_lrs     = d.get("base_lrs", self.base_lrs)


# ══════════════════════════════════════════════════════════════
#  TRAINER v4
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

        # CHECKPOINT TRACKING (internal, updated by CheckpointManager)
        self.best_train_loss = math.inf
        self.best_val_metric = -1.0 if self._monitor_mode() == "max" else math.inf
        self.best_val_epoch  = 0
        self.global_step     = 0

        # CHECKPOINT MANAGER
        self.ckpt_mgr = CheckpointManager(exp_path, config, ckpt_log=logger)

        ema_str = f"EMA(decay={ema_decay})" if use_ema else "no EMA"
        bw_str  = f"boundary_weight={self.boundary_weight}" if self.boundary_weight > 0 else "no boundary loss"
        self.logger.info(
            f"Trainer v4 | {opt_type.upper()} | LR={lr_policy} | "
            f"warmup={self.warmup_epochs} | val_every={self.val_every} | "
            f"{ema_str} | {bw_str} | checkpoint monitor={self.ckpt_mgr.monitor}"
        )

    # ──────────────────────────────────────────────────────
    # MONITOR HELPERS
    # ──────────────────────────────────────────────────────

    def _monitor_mode(self) -> str:
        """Return 'max' or 'min' based on checkpoint config."""
        ckpt_cfg = getattr(self.config, "checkpoint", None)
        return str(getattr(ckpt_cfg, "mode", "max")).lower()

    def _is_better(self, new_val: float) -> bool:
        if self._monitor_mode() == "max":
            return new_val > self.best_val_metric
        return new_val < self.best_val_metric

    def _extract_monitor_metric(self, val_dpc, mean_dice, tc_dice) -> float:
        """Extract the monitored metric value from validation results."""
        monitor = self.ckpt_mgr.monitor

        metric_map = {
            "dice_mean":   mean_dice,
            "mean_dice":   mean_dice,
            "tc_dice":     tc_dice,
            "wt_dice":     val_dpc[1] if len(val_dpc) > 1 else mean_dice,
            "et_dice":     val_dpc[3] if len(val_dpc) > 3 else mean_dice,
        }
        return metric_map.get(monitor, mean_dice)

    # ──────────────────────────────────────────────────────
    # RESUME SUPPORT
    # ──────────────────────────────────────────────────────

    def resume_from(self, ckpt_path: str) -> None:
        """
        Load a full resume checkpoint and restore all training states.
        Call this BEFORE train().
        """
        info = self.ckpt_mgr.load_resume(
            path=ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema=self.ema,
            device=self.device,
        )
        self._resume_start_epoch = info["start_epoch"]
        self.global_step         = info["global_step"]
        self.best_val_metric     = info["best_metric"]
        self.logger.info(
            f"[Trainer v4] Resuming from epoch {self._resume_start_epoch} | "
            f"global_step={self.global_step} | best_metric={self.best_val_metric:.4f}"
        )

    # ──────────────────────────────────────────────────────
    # LOSS COMPUTATION  (identical to v3)
    # ──────────────────────────────────────────────────────

    def _loss_for_output(self, out, seg, apply_boundary=False):
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
        if isinstance(outputs, list):
            total_loss = None
            agg = {"loss_dice": 0.0, "loss_ce": 0.0, "loss_focal": 0.0}

            for i, out in enumerate(outputs):
                w = self.ds_weights[i] if i < len(self.ds_weights) else self.ds_weights[-1]

                if out.shape[2:] != seg.shape[1:]:
                    seg_i = F.interpolate(
                        seg.float().unsqueeze(1),
                        size=out.shape[2:],
                        mode="nearest",
                    ).squeeze(1).long()
                else:
                    seg_i = seg

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
    # VALIDATION  (identical to v3)
    # ──────────────────────────────────────────────────────

    def validate(self):
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
    # TRAIN  (v4: wrapped in try/except, CheckpointManager calls)
    # ──────────────────────────────────────────────────────

    def train(self):
        self.model.train()

        # Resume offset: if resume_from() was called, start from correct epoch
        start_epoch = getattr(self, "_resume_start_epoch", 0)

        try:
            self._train_loop(start_epoch)
        except KeyboardInterrupt:
            self.logger.warning("[Trainer v4] Training interrupted by user (KeyboardInterrupt).")
            self._emergency_save(getattr(self, "_current_epoch", start_epoch))
            raise
        except Exception as e:
            self.logger.error(f"[Trainer v4] Training failed with exception: {e}")
            self._emergency_save(getattr(self, "_current_epoch", start_epoch))
            raise

    def _emergency_save(self, epoch: int) -> None:
        self.ckpt_mgr.save_emergency(
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            ema=self.ema,
            best_metric=self.best_val_metric,
        )
        self.ckpt_mgr.update_state(
            epoch=epoch,
            best_epoch=self.best_val_epoch,
            best_metric=self.best_val_metric,
            interrupted=True,
        )

    def _train_loop(self, start_epoch: int) -> None:
        import time as _time
        for epoch in range(start_epoch, self.total_epochs):
            self._current_epoch = epoch   # tracked for emergency save
            _epoch_start = _time.monotonic()

            # ── Batch loop ────────────────────────────────────────────
            total_loss   = 0.0
            total_dice   = 0.0
            total_ce     = 0.0
            total_focal  = 0.0
            total_bnd    = 0.0
            total_gn     = 0.0
            epoch_dpc    = [0.0] * self.num_classes
            n_batches    = 0
            grad_clip_count = 0
            nan_detected    = False

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
                    # Track if gradient was actually clipped
                    if grad_norm > self.grad_clip:
                        grad_clip_count += 1

                # Detect NaN in loss
                if not torch.isfinite(loss):
                    nan_detected = True

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.use_ema:
                    self.ema.update(self.model)

                self.global_step += 1

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

            # ── LR step ───────────────────────────────────────────────
            self._step_scheduler(epoch)

            # ── Logging ───────────────────────────────────────────────
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

            # ── Legacy best_train (kept for backward compat) ──────────
            if avg_loss < self.best_train_loss:
                self.best_train_loss = avg_loss
                save_model(self.model, os.path.join(self.exp_path, "checkpoints", "best_train.pth"))

            # ── Validation ───────────────────────────────────────────
            val_mean, val_tc, val_dpc = None, None, None
            if (epoch + 1) % self.val_every == 0 or epoch == self.total_epochs - 1:
                _val_start = _time.monotonic()
                val_mean, val_tc, val_dpc = self.validate()
                _val_elapsed = _time.monotonic() - _val_start

                val_str  = " | ".join(f"C{c}:{val_dpc[c]:.3f}" for c in range(self.num_classes))
                ema_tag  = " [EMA]" if self.use_ema else ""
                self.logger.info(
                    f"  [VAL{ema_tag}] Epoch {epoch:03d} | MeanDice: {val_mean:.4f} | "
                    f"TC: {val_tc:.4f} | PerClass [{val_str}]"
                )

                monitor_val = self._extract_monitor_metric(val_dpc, val_mean, val_tc)

                if self._is_better(monitor_val):
                    self.best_val_metric = monitor_val
                    self.best_val_epoch  = epoch

                    # ── Tier B: Best Model checkpoint ─────────────────
                    self.ckpt_mgr.save_best(
                        epoch=epoch,
                        model=self.model,
                        metric_value=monitor_val,
                        ema=self.ema,
                    )

                    # Legacy best.pth compatibility (evaluators that expect it)
                    best_pth = os.path.join(self.exp_path, "checkpoints", "best.pth")
                    if self.use_ema:
                        torch.save(self.ema.state_dict(), best_pth)
                    else:
                        save_model(self.model, best_pth)
                    self.logger.info(
                        f"  -> best.pth updated ({self.ckpt_mgr.monitor}: {monitor_val:.4f})"
                    )

                # Fire validation timing hook
                if hasattr(self, "_on_validation_end"):
                    try:
                        self._on_validation_end(epoch, _val_elapsed)
                    except Exception:
                        pass

            # ── Tier A: Full Resume checkpoint ────────────────────────
            self.ckpt_mgr.save_resume(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                ema=self.ema,
                best_metric=self.best_val_metric,
                global_step=self.global_step,
            )

            # ── Tier C: Periodic snapshot ─────────────────────────────
            self.ckpt_mgr.save_periodic(epoch, self.model)

            # ── Tier D: Training state JSON ───────────────────────────
            self.ckpt_mgr.update_state(
                epoch=epoch,
                best_epoch=self.best_val_epoch,
                best_metric=self.best_val_metric,
                interrupted=False,
                extra={
                    "current_lr":    self.optimizer.param_groups[0]["lr"],
                    "global_step":   self.global_step,
                    "train_loss":    round(total_loss / n_batches, 6),
                    "val_mean_dice": round(val_mean, 6) if val_mean is not None else None,
                    "val_tc_dice":   round(val_tc, 6)   if val_tc   is not None else None,
                }
            )

            # ── Fire _on_epoch_end hook (external tracking) ───────────
            # run_experiment.py attaches this hook for CurveTracker + TimingTracker.
            # Older trainers or trainers without the hook are unaffected.
            _epoch_dur = _time.monotonic() - _epoch_start
            if hasattr(self, "_on_epoch_end"):
                try:
                    # Collect GPU memory for diagnostics
                    _gpu_mem_gb = None
                    try:
                        if torch.cuda.is_available():
                            _gpu_mem_gb = round(
                                torch.cuda.max_memory_allocated() / (1024 ** 3), 3
                            )
                    except Exception:
                        pass

                    hook_metrics = {
                        "train_loss":       round(total_loss / n_batches, 6),
                        "train_dice":       round(total_dice  / n_batches, 6),
                        "lr":               round(lr_now, 8),
                        "grad_norm":        round(avg_gn, 4),
                        "epoch_duration_s": round(_epoch_dur, 2),
                        "grad_clip_count":  grad_clip_count,
                        "nan_detected":     nan_detected,
                    }
                    if _gpu_mem_gb is not None:
                        hook_metrics["gpu_memory_gb"] = _gpu_mem_gb
                    if val_mean is not None:
                        hook_metrics["val_dice"]    = round(val_mean, 6)
                        hook_metrics["val_tc_dice"] = round(val_tc,   6)
                        hook_metrics["val_loss"]    = round(total_loss / n_batches, 6)
                        if val_dpc and len(val_dpc) > 1:
                            hook_metrics["val_wt_dice"] = round(val_dpc[1], 6)
                        if val_dpc and len(val_dpc) > 3:
                            hook_metrics["val_et_dice"] = round(val_dpc[3], 6)
                    self._on_epoch_end(epoch, hook_metrics, _epoch_dur)
                except Exception:
                    pass  # hooks must never crash training

        self.logger.info("Training Finished")
