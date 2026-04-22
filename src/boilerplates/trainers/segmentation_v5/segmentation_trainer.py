"""
Segmentation Trainer v5 — Advanced Research Trainer
=====================================================
Extends Trainer v4 with features required for novel architecture research:

  1. CUSTOM OPTIMIZER FACTORY
     - Models can define get_optimizer_groups() returning separate param groups
       with different learning rates and optimizer types
     - Enables bi-level optimization (e.g., SpectralOptimizer + Adam)
     - Falls back to standard AdamW if model doesn't define groups

  2. GRADIENT ACCUMULATION
     - Config: training.accumulate_steps (default 1)
     - Effective batch size = physical batch × accumulate_steps
     - Critical for 3D medical at batch_size=1 on 16GB GPUs

  3. GRADIENT CHECKPOINTING
     - Config: training.gradient_checkpointing (default false)
     - Trades ~30% speed for ~40% VRAM savings
     - Model must implement enable_gradient_checkpointing()

  4. BUILT-IN HOOKABLE TRACKER INTEGRATION
     - Automatically uses trainer.hookable_tracker if attached by run_experiment.py
     - Calls model.get_spectral_metrics() each epoch if available
     - Pushes eigenvalues, condition numbers, graph stats to artifact system

  5. MODEL PROFILING AT INIT
     - Logs total/trainable/frozen params, memory estimate
     - Logs per-module param breakdown for paper appendix

  6. INHERITS ALL v4 FEATURES
     - AdamW + cosine/poly LR + warmup
     - EMA with configurable decay
     - Boundary-aware loss + deep supervision
     - 4-tier checkpoint system (CheckpointManager)
     - Emergency save on crash/interrupt
     - Per-class Dice logging + gradient norm
     - _on_epoch_end / _on_validation_end hooks

Config additions (all optional, backward compatible):
    training:
      accumulate_steps: 4
      gradient_checkpointing: false
      aux_lr: 1e-4              # LR for auxiliary optimizer (StructureNet)
      spectral_log_every: 1     # epochs between spectral metric logging

    versions:
      trainer: segmentation_v5
"""

import os
import math
import time as _time
from copy import deepcopy
from typing import Optional, Dict, Any, List, Tuple

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
#  MODEL EMA  (identical to v4)
# ══════════════════════════════════════════════════════════════

class ModelEMA:
    """Exponential Moving Average of model weights."""

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


def _count_parameters(model) -> Dict[str, Any]:
    """Count parameters and estimate memory for paper reporting."""
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen   = total - trainable
    mem_mb   = total * 4 / (1024 ** 2)  # float32 estimate

    # Per-module breakdown (top-level children only)
    breakdown = {}
    for name, child in model.named_children():
        child_params = sum(p.numel() for p in child.parameters())
        breakdown[name] = {
            "params": child_params,
            "percent": round(100 * child_params / max(total, 1), 1),
        }

    return {
        "total_params":     total,
        "trainable_params": trainable,
        "frozen_params":    frozen,
        "memory_mb":        round(mem_mb, 1),
        "breakdown":        breakdown,
    }


# ══════════════════════════════════════════════════════════════
#  TRAINER v5
# ══════════════════════════════════════════════════════════════

class Trainer:

    def __init__(self, config, exp_path, logger):
        self.config   = config
        self.logger   = logger
        self.exp_path = exp_path
        self.device   = get_device(config)

        # ── MODEL ──────────────────────────────────────────────
        self.model = build_model(config).to(self.device)

        # ── MODEL PROFILING (for paper appendix) ───────────────
        self._model_profile = _count_parameters(self.model)
        logger.info(
            f"[Trainer v5] Model: {self._model_profile['total_params']:,} params "
            f"({self._model_profile['trainable_params']:,} trainable, "
            f"~{self._model_profile['memory_mb']:.0f} MB)"
        )
        # Save breakdown to experiment folder
        self._save_model_profile()

        # ── GRADIENT CHECKPOINTING ──────────────────────────────
        train_cfg = config.training
        self.use_grad_ckpt = bool(getattr(train_cfg, "gradient_checkpointing", False))
        if self.use_grad_ckpt:
            if hasattr(self.model, "enable_gradient_checkpointing"):
                self.model.enable_gradient_checkpointing()
                logger.info("[Trainer v5] Gradient checkpointing enabled")
            else:
                logger.info("[Trainer v5] Model does not support gradient checkpointing — skipping")
                self.use_grad_ckpt = False

        # ── DATA ───────────────────────────────────────────────
        self.train_loader = build_dataloader(config, split="train")
        self.val_loader   = build_dataloader(config, split="val")

        # ── LOSS CONFIG ────────────────────────────────────────
        loss_cfg = getattr(config, "loss", None)
        self.class_weights    = getattr(loss_cfg, "class_weights",   [0.1, 3.0, 1.0, 2.0])
        self.focal_weight     = float(getattr(loss_cfg, "focal_weight",    0.5))
        self.boundary_weight  = float(getattr(loss_cfg, "boundary_weight", 0.4))
        self.boundary_factor  = float(getattr(loss_cfg, "boundary_factor", 5.0))
        self.boundary_width   = int(  getattr(loss_cfg, "boundary_width",  2))

        # ── DEEP SUPERVISION WEIGHTS ───────────────────────────
        raw_ds    = getattr(train_cfg, "ds_weights", [1.0, 0.5, 0.25])
        self.ds_weights = list(raw_ds) if hasattr(raw_ds, '__iter__') else [1.0, 0.5, 0.25]

        # ── HYPERPARAMS ────────────────────────────────────────
        self.grad_clip     = float(getattr(train_cfg, "grad_clip",     1.0))
        self.warmup_epochs = int(  getattr(train_cfg, "warmup_epochs", 20))
        self.val_every     = int(  getattr(train_cfg, "val_every",     10))
        self.total_epochs  = int(config.training.epochs)
        self.num_classes   = getattr(config.model, "out_channels", 4)
        lr                 = float(config.training.lr)
        lr_min             = float(getattr(train_cfg, "lr_min",      1e-6))
        lr_policy          = str(  getattr(train_cfg, "lr_policy",   "cosine")).lower()

        # ── GRADIENT ACCUMULATION ──────────────────────────────
        self.accumulate_steps = int(getattr(train_cfg, "accumulate_steps", 1))

        # ── SPECTRAL LOGGING FREQUENCY ─────────────────────────
        self.spectral_log_every = int(getattr(train_cfg, "spectral_log_every", 1))

        # ── EMA ────────────────────────────────────────────────
        use_ema        = bool(getattr(train_cfg, "use_ema",   True))
        ema_decay      = float(getattr(train_cfg, "ema_decay", 0.9999))
        self.ema       = ModelEMA(self.model, decay=ema_decay) if use_ema else None
        self.use_ema   = use_ema

        # ── OPTIMIZERS (v5: custom optimizer factory) ──────────
        self.optimizer, self.optimizer_aux = self._build_optimizers(config, lr)

        # ── LR SCHEDULE (applied to primary optimizer) ─────────
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

        # ── AMP ────────────────────────────────────────────────
        self.use_amp = getattr(train_cfg, "mixed_precision", True)
        self.scaler  = GradScaler(enabled=self.use_amp)

        # ── CHECKPOINT TRACKING ────────────────────────────────
        self.best_train_loss = math.inf
        self.best_val_metric = -1.0 if self._monitor_mode() == "max" else math.inf
        self.best_val_epoch  = 0
        self.global_step     = 0

        # ── CHECKPOINT MANAGER ─────────────────────────────────
        self.ckpt_mgr = CheckpointManager(exp_path, config, ckpt_log=logger)

        # ── CUSTOM LOSS (model can augment the loss) ───────────
        # If model defines get_auxiliary_losses(), those are added to total
        self._has_aux_losses = hasattr(self.model, "get_auxiliary_losses")

        ema_str = f"EMA(decay={ema_decay})" if use_ema else "no EMA"
        bw_str  = f"boundary_weight={self.boundary_weight}" if self.boundary_weight > 0 else "no boundary loss"
        accum_str = f"accum={self.accumulate_steps}" if self.accumulate_steps > 1 else "no accum"
        aux_str = "bi-level" if self.optimizer_aux else "single"
        self.logger.info(
            f"Trainer v5 | {aux_str} optimizer | LR={lr_policy} | "
            f"warmup={self.warmup_epochs} | val_every={self.val_every} | "
            f"{ema_str} | {bw_str} | {accum_str} | "
            f"checkpoint monitor={self.ckpt_mgr.monitor}"
        )

    # ──────────────────────────────────────────────────────
    # OPTIMIZER FACTORY (v5 NEW)
    # ──────────────────────────────────────────────────────

    def _build_optimizers(self, config, lr: float):
        """
        Build primary and auxiliary optimizers.

        If the model defines get_optimizer_groups(lr), use those groups.
        This enables bi-level optimization:
          - Group 'primary': transformer weights → primary optimizer
          - Group 'auxiliary': StructureNet weights → auxiliary optimizer

        If the model defines get_custom_optimizer(lr), use that directly.
        Otherwise, fall back to standard AdamW.

        Returns:
            (primary_optimizer, auxiliary_optimizer_or_None)
        """
        train_cfg = config.training
        opt_cfg   = getattr(config, "optimizer", None)
        weight_decay = float(getattr(opt_cfg, "weight_decay", 0.01))

        # ── Strategy 1: Model provides a full custom optimizer ──────
        if hasattr(self.model, "get_custom_optimizer"):
            primary = self.model.get_custom_optimizer(lr, weight_decay)
            self.logger.info("[Trainer v5] Using model-defined custom optimizer")
            return primary, None

        # ── Strategy 2: Model defines separate param groups ─────────
        if hasattr(self.model, "get_optimizer_groups"):
            groups = self.model.get_optimizer_groups(lr)
            # groups is a dict: {'primary': [param_dicts], 'auxiliary': [param_dicts]}
            primary_groups = groups.get("primary", [{"params": self.model.parameters()}])
            aux_groups     = groups.get("auxiliary", None)

            primary = torch.optim.AdamW(primary_groups, lr=lr, weight_decay=weight_decay)

            aux_optimizer = None
            if aux_groups:
                aux_lr = float(getattr(train_cfg, "aux_lr", lr * 0.1))
                aux_optimizer = torch.optim.Adam(aux_groups, lr=aux_lr)
                self.logger.info(
                    f"[Trainer v5] Bi-level optimizer: "
                    f"primary LR={lr:.2e}, auxiliary LR={aux_lr:.2e}"
                )

            return primary, aux_optimizer

        # ── Strategy 3: Standard AdamW (backward compatible) ────────
        opt_type = getattr(opt_cfg, "type", "adamw").lower() if opt_cfg else "adamw"

        # Layer-wise LR decay
        layer_lr_decay = float(getattr(train_cfg, "layer_lr_decay", 1.0))
        if layer_lr_decay < 1.0 and hasattr(self.model, "get_param_groups"):
            param_groups = self.model.get_param_groups(lr, layer_lr_decay)
        else:
            param_groups = self.model.parameters()

        if opt_type == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=lr)
        else:
            optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        return optimizer, None

    # ──────────────────────────────────────────────────────
    # MODEL PROFILING (v5 NEW)
    # ──────────────────────────────────────────────────────

    def _save_model_profile(self) -> None:
        """Save model parameter breakdown to experiment folder for paper appendix."""
        import json
        profile_path = os.path.join(self.exp_path, "model_profile.json")
        try:
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(self._model_profile, f, indent=2, default=str)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────
    # MONITOR HELPERS (from v4)
    # ──────────────────────────────────────────────────────

    def _monitor_mode(self) -> str:
        ckpt_cfg = getattr(self.config, "checkpoint", None)
        return str(getattr(ckpt_cfg, "mode", "max")).lower()

    def _is_better(self, new_val: float) -> bool:
        if self._monitor_mode() == "max":
            return new_val > self.best_val_metric
        return new_val < self.best_val_metric

    def _extract_monitor_metric(self, val_dpc, mean_dice, tc_dice) -> float:
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
    # RESUME SUPPORT (from v4)
    # ──────────────────────────────────────────────────────

    def resume_from(self, ckpt_path: str) -> None:
        """Load a full resume checkpoint and restore all training states."""
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
            f"[Trainer v5] Resuming from epoch {self._resume_start_epoch} | "
            f"global_step={self.global_step} | best_metric={self.best_val_metric:.4f}"
        )

    # ──────────────────────────────────────────────────────
    # LOSS COMPUTATION (from v4, extended with aux losses)
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

        # ── v5: Model auxiliary losses (prior fidelity, smoothness, spectral) ──
        if self._has_aux_losses:
            try:
                aux_losses = self.model.get_auxiliary_losses()
                for name, (loss_val, weight) in aux_losses.items():
                    total_loss = total_loss + weight * loss_val
                    components[f"loss_{name}"] = loss_val.item()
            except Exception:
                pass

        return total_loss, components

    # ──────────────────────────────────────────────────────
    # VALIDATION (from v4)
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
    # LR STEP (from v4)
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
    # SPECTRAL METRIC COLLECTION (v5 NEW)
    # ──────────────────────────────────────────────────────

    def _collect_spectral_metrics(self, epoch: int) -> Dict[str, float]:
        """
        Collect spectral/structural metrics from model if available.

        The model can implement get_spectral_metrics() returning a dict:
            {
                "condition_number": float,
                "eigenvalue_gap": float,
                "gradient_alignment": float,
                "spectrum_entropy": float,
                "graph_sparsity": float,
                "mean_uncertainty": float,
            }

        These are logged to curves and pushed to the artifact system.
        """
        if not hasattr(self.model, "get_spectral_metrics"):
            return {}

        if epoch % self.spectral_log_every != 0:
            return {}

        try:
            metrics = self.model.get_spectral_metrics()
            if not isinstance(metrics, dict):
                return {}

            # Push to hookable tracker if available
            hookable = getattr(self, "hookable_tracker", None)
            if hookable is not None:
                for key, value in metrics.items():
                    try:
                        hookable.log_scalar(key, value)
                    except Exception:
                        pass

            return metrics
        except Exception as e:
            self.logger.debug(f"[Trainer v5] Spectral metric collection failed: {e}")
            return {}

    def _collect_model_tensors(self, epoch: int) -> None:
        """
        Collect tensor artifacts from model (graph matrices, eigenvalues, etc.).

        The model can implement get_tensor_artifacts() returning:
            {
                "A_prior": Tensor,
                "A_learned": Tensor,
                "eigenvalues": Tensor,
                "attention_map": Tensor,
            }
        """
        if not hasattr(self.model, "get_tensor_artifacts"):
            return

        hookable = getattr(self, "hookable_tracker", None)
        if hookable is None:
            return

        try:
            artifacts = self.model.get_tensor_artifacts()
            if isinstance(artifacts, dict):
                for name, tensor in artifacts.items():
                    try:
                        hookable.log_tensor(name, tensor)
                    except Exception:
                        pass
        except Exception:
            pass

    # ──────────────────────────────────────────────────────
    # TRAIN (v5: with gradient accumulation + bi-level + spectral)
    # ──────────────────────────────────────────────────────

    def train(self):
        self.model.train()
        start_epoch = getattr(self, "_resume_start_epoch", 0)

        try:
            self._train_loop(start_epoch)
        except KeyboardInterrupt:
            self.logger.warning("[Trainer v5] Training interrupted by user (KeyboardInterrupt).")
            self._emergency_save(getattr(self, "_current_epoch", start_epoch))
            raise
        except Exception as e:
            self.logger.error(f"[Trainer v5] Training failed with exception: {e}")
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
        accum = self.accumulate_steps

        for epoch in range(start_epoch, self.total_epochs):
            self._current_epoch = epoch
            _epoch_start = _time.monotonic()

            # v5: Notify model of current epoch (for warmup-aware aux losses)
            if hasattr(self.model, "set_epoch"):
                self.model.set_epoch(epoch)

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

            # Zero grads at start (for accumulation)
            self.optimizer.zero_grad()
            if self.optimizer_aux:
                self.optimizer_aux.zero_grad()

            for batch_idx, (img, seg) in enumerate(self.train_loader):
                img = img.to(self.device, non_blocking=True)
                seg = seg.to(self.device, non_blocking=True)

                with autocast(device_type="cuda", enabled=self.use_amp):
                    outputs = self.model(img)
                    loss, components = self.compute_loss(outputs, seg)

                    # Scale loss for gradient accumulation
                    if accum > 1:
                        loss = loss / accum

                self.scaler.scale(loss).backward()

                # ── Optimizer step (every accum steps or last batch) ────
                is_accumulation_step = (
                    (batch_idx + 1) % accum == 0 or
                    (batch_idx + 1) == len(self.train_loader)
                )

                if is_accumulation_step:
                    # Primary optimizer
                    grad_norm = 0.0
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        ).item()
                        if grad_norm > self.grad_clip:
                            grad_clip_count += 1

                    # Detect NaN in loss
                    if not torch.isfinite(loss * accum):
                        nan_detected = True

                    self.scaler.step(self.optimizer)

                    # Auxiliary optimizer (bi-level)
                    if self.optimizer_aux:
                        try:
                            self.scaler.unscale_(self.optimizer_aux)
                            self.scaler.step(self.optimizer_aux)
                        except Exception:
                            pass  # aux optimizer failure is non-fatal

                    self.scaler.update()

                    # Zero grads for next accumulation
                    self.optimizer.zero_grad()
                    if self.optimizer_aux:
                        self.optimizer_aux.zero_grad()

                    if self.use_ema:
                        self.ema.update(self.model)

                    self.global_step += 1
                    total_gn += grad_norm

                # Accumulate metrics (always, not just on step)
                total_loss  += components["loss_total"]
                total_dice  += components["loss_dice"]
                total_ce    += components["loss_ce"]
                total_focal += components.get("loss_focal", 0.0)
                total_bnd   += components.get("loss_boundary_dice", 0.0)
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
            avg_gn   = total_gn / max(self.global_step, 1)  # per-step average
            avg_bnd  = total_bnd / n_batches
            avg_dpc  = [d / n_batches for d in epoch_dpc]
            dpc_str  = " | ".join(f"C{c}:{avg_dpc[c]:.3f}" for c in range(self.num_classes))

            self.logger.info(
                f"Epoch {epoch:03d}/{self.total_epochs} | LR: {lr_now:.2e} | "
                f"Loss: {avg_loss:.4f} | Dice: {total_dice/n_batches:.4f} | "
                f"CE: {total_ce/n_batches:.4f} | Focal: {total_focal/n_batches:.4f} | "
                f"Bnd: {avg_bnd:.4f} | GradNorm: {avg_gn:.3f} | PerClass [{dpc_str}]"
            )

            # Log auxiliary loss components if present
            if self._has_aux_losses:
                aux_strs = []
                for k in ["loss_prior", "loss_smooth", "loss_eig"]:
                    v = components.get(k)
                    if v is not None:
                        aux_strs.append(f"{k}={v:.4f}")
                if aux_strs:
                    self.logger.info(f"  [AUX] {' | '.join(aux_strs)}")

            # ── Legacy best_train (backward compat) ───────────────────
            if avg_loss < self.best_train_loss:
                self.best_train_loss = avg_loss
                save_model(self.model, os.path.join(self.exp_path, "checkpoints", "best_train.pth"))

            # ── v5: Collect spectral metrics from model ───────────────
            spectral_metrics = self._collect_spectral_metrics(epoch)
            if spectral_metrics:
                spec_str = " | ".join(f"{k}={v:.4f}" for k, v in spectral_metrics.items())
                self.logger.info(f"  [SPECTRAL] {spec_str}")

            # ── v5: Collect tensor artifacts (graphs, eigenvalues) ─────
            self._collect_model_tensors(epoch)

            # ── Validation ────────────────────────────────────────────
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

                    self.ckpt_mgr.save_best(
                        epoch=epoch,
                        model=self.model,
                        metric_value=monitor_val,
                        ema=self.ema,
                    )

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
            state_extra = {
                "current_lr":    self.optimizer.param_groups[0]["lr"],
                "global_step":   self.global_step,
                "train_loss":    round(total_loss / n_batches, 6),
                "val_mean_dice": round(val_mean, 6) if val_mean is not None else None,
                "val_tc_dice":   round(val_tc, 6)   if val_tc   is not None else None,
                "trainer_version": "v5",
            }
            # Include spectral metrics in training state
            for k, v in spectral_metrics.items():
                state_extra[f"spectral_{k}"] = round(v, 6) if isinstance(v, float) else v

            self.ckpt_mgr.update_state(
                epoch=epoch,
                best_epoch=self.best_val_epoch,
                best_metric=self.best_val_metric,
                interrupted=False,
                extra=state_extra,
            )

            # ── Fire _on_epoch_end hook (external tracking) ───────────
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

                    # Include spectral metrics in hook
                    for k, v in spectral_metrics.items():
                        hook_metrics[k] = round(v, 6) if isinstance(v, float) else v

                    # Include auxiliary loss components
                    for k in ["loss_prior", "loss_smooth", "loss_eig"]:
                        v = components.get(k)
                        if v is not None:
                            hook_metrics[k] = round(v, 6)

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

            # ── v5: Flush hookable tracker artifacts for this epoch ────
            hookable = getattr(self, "hookable_tracker", None)
            if hookable is not None:
                try:
                    hookable.flush(epoch)
                except Exception:
                    pass

        self.logger.info("Training Finished")
