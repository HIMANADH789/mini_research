"""
checkpoint_utils.py
===================
4-tier checkpoint system for the mini_research training pipeline.

Tier A — Resume Checkpoint  (checkpoint_resume_latest.pt)
    Full training state: model, optimizer, scheduler, scaler, ema,
    epoch, global_step, best metric, config hash, RNG states.
    Needed to fully reconstruct interrupted training.

Tier B — Best Model  (best_model.pt)
    Inference-only: model state_dict + metadata.
    Always kept. Overwritten when a new best is found.

Tier C — Periodic Snapshot  (periodic_epoch_050.pt)
    Model weights only, written every N epochs.
    Retention policy: keep last K, auto-delete older ones.

Tier D — Training State JSON  (training_state.json)
    Human-readable progress log, updated each epoch.

Design choices:
- Atomic writes via tempfile + os.replace (Windows NTFS safe)
- All tensors moved to CPU before saving (avoids GPU memory issues)
- Config hash stored to warn on resume mismatches
- Emergency checkpoint on exception / KeyboardInterrupt
"""

import os
import json
import time
import hashlib
import random
import tempfile
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

import torch
import numpy as np


logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _config_hash(config) -> str:
    """Compute a short hash of the config to detect mismatches on resume."""
    try:
        import yaml
        cfg_str = yaml.dump(vars(config) if hasattr(config, "__dict__") else str(config),
                            default_flow_style=True, sort_keys=True)
    except Exception:
        cfg_str = str(config)
    return hashlib.md5(cfg_str.encode()).hexdigest()[:8]


def _collect_rng_states() -> Dict[str, Any]:
    """Collect all RNG states for full reproducibility on resume."""
    states: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        try:
            states["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return states


def _restore_rng_states(states: Dict[str, Any]) -> None:
    """Restore RNG states exactly as they were at the saved checkpoint."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch.set_rng_state(torch.tensor(states["torch"], dtype=torch.uint8))
    if "cuda" in states and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(states["cuda"])
        except Exception:
            pass


def _atomic_save(obj: Any, path: str) -> None:
    """
    Save a PyTorch object atomically using a temp file + os.replace.

    On Windows, os.replace is atomic within the same filesystem.
    This prevents corrupt checkpoints from partially-written files.
    """
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)

    # Write to a temp file in the same directory
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        os.close(fd)
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)   # atomic on Windows NTFS
    except Exception:
        # Clean up temp file if something went wrong
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _atomic_json_save(obj: Any, path: str) -> None:
    """JSON equivalent of _atomic_save."""
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


# ══════════════════════════════════════════════════════════════
#  CHECKPOINT MANAGER
# ══════════════════════════════════════════════════════════════

class CheckpointManager:
    """
    Manages all checkpoint tiers for a single training run.

    Usage in Trainer.__init__:
        self.ckpt_mgr = CheckpointManager(exp_path, config)

    Usage in Trainer.train() per epoch:
        self.ckpt_mgr.save_resume(epoch, model, optimizer, scheduler,
                                   scaler, ema, best_metric, global_step)
        self.ckpt_mgr.save_periodic(epoch, model)
        self.ckpt_mgr.update_state(epoch, best_epoch, best_metric, interrupted=False)

    When a new best metric is found:
        self.ckpt_mgr.save_best(epoch, model, metric_value, ema=self.ema)

    On exception:
        self.ckpt_mgr.save_emergency(epoch, model, optimizer)
    """

    def __init__(self, exp_path: str, config, ckpt_log: Optional[logging.Logger] = None):
        """
        Parameters
        ----------
        exp_path : str
            Root experiment directory (e.g., experiments/exp_012_swinunetr)
        config : Config
            Parsed experiment config object
        ckpt_log : Logger, optional
            If None, uses module-level logger.
        """
        self.exp_path   = exp_path
        self.ckpt_dir   = os.path.join(exp_path, "checkpoints")
        self.config     = config
        self.log        = ckpt_log or logger
        self.config_hash = _config_hash(config)

        os.makedirs(self.ckpt_dir, exist_ok=True)

        # ── Read checkpoint config section ────────────────────────────────
        ckpt_cfg = getattr(config, "checkpoint", None)

        self.save_best_flag          = bool(getattr(ckpt_cfg, "save_best",               True))
        self.save_resume_every_epoch = bool(getattr(ckpt_cfg, "save_resume_every_epoch", True))
        self.save_periodic_every     = int(getattr(ckpt_cfg,  "save_periodic_every",     10))
        self.keep_last_n_periodic    = int(getattr(ckpt_cfg,  "keep_last_n_periodic",    3))
        self.monitor                 = str(getattr(ckpt_cfg,  "monitor",                 "dice_mean"))
        self.mode                    = str(getattr(ckpt_cfg,  "mode",                    "max"))

        # Track which periodic checkpoints exist for retention policy
        self._periodic_saved: List[str] = self._scan_existing_periodic()

        self.log.info(
            f"[CheckpointManager] dir={self.ckpt_dir} | "
            f"monitor={self.monitor} | mode={self.mode} | "
            f"periodic_every={self.save_periodic_every} | "
            f"keep_last_n={self.keep_last_n_periodic}"
        )

    # ─────────────────────────────────────────────────────────
    # PATHS
    # ─────────────────────────────────────────────────────────

    def _resume_latest_path(self) -> str:
        return os.path.join(self.ckpt_dir, "checkpoint_resume_latest.pt")

    def _resume_epoch_path(self, epoch: int) -> str:
        return os.path.join(self.ckpt_dir, f"checkpoint_resume_epoch_{epoch:04d}.pt")

    def _best_model_path(self) -> str:
        return os.path.join(self.ckpt_dir, "best_model.pt")

    def _periodic_path(self, epoch: int) -> str:
        return os.path.join(self.ckpt_dir, f"periodic_epoch_{epoch:04d}.pt")

    def _state_json_path(self) -> str:
        return os.path.join(self.exp_path, "training_state.json")

    def _emergency_path(self, epoch: int) -> str:
        return os.path.join(self.ckpt_dir, f"emergency_epoch_{epoch:04d}.pt")

    # ─────────────────────────────────────────────────────────
    # A — FULL RESUME CHECKPOINT
    # ─────────────────────────────────────────────────────────

    def save_resume(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,  # any scheduler
        scaler: Optional[torch.cuda.amp.GradScaler],
        ema,        # ModelEMA instance or None
        best_metric: float,
        global_step: int = 0,
    ) -> None:
        """Save a full resume checkpoint after each epoch."""
        if not self.save_resume_every_epoch:
            return

        payload = {
            "epoch":        epoch,
            "global_step":  global_step,
            "best_metric":  best_metric,
            "config_hash":  self.config_hash,
            "timestamp":    datetime.now().isoformat(),
            # Model (move to CPU to avoid GPU memory pressure)
            "model_state":  {k: v.cpu() for k, v in model.state_dict().items()},
            # Optimizer
            "optimizer_state": optimizer.state_dict(),
            # Scheduler — handle SequentialLR and custom PolyLR
            "scheduler_state": _safe_scheduler_state(scheduler),
            # AMP scaler
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            # EMA
            "ema_state": (
                {k: v.cpu() for k, v in ema.state_dict().items()}
                if ema is not None else None
            ),
            # RNG states for full reproducibility
            "rng_states": _collect_rng_states(),
        }

        # Write to latest (always overwrite)
        _atomic_save(payload, self._resume_latest_path())
        self.log.debug(f"[Checkpoint] Resume saved → {self._resume_latest_path()}")

    # ─────────────────────────────────────────────────────────
    # B — BEST MODEL CHECKPOINT
    # ─────────────────────────────────────────────────────────

    def save_best(
        self,
        epoch: int,
        model: torch.nn.Module,
        metric_value: float,
        ema=None,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Save the best model checkpoint (inference-only weights).
        If EMA is present, saves EMA weights as the best model.
        """
        if not self.save_best_flag:
            return

        weights = ema.state_dict() if ema is not None else model.state_dict()
        payload = {
            "epoch":        epoch,
            "best_metric":  metric_value,
            "monitor":      self.monitor,
            "config_hash":  self.config_hash,
            "timestamp":    datetime.now().isoformat(),
            "model_state":  {k: v.cpu() for k, v in weights.items()},
        }
        if extra:
            payload.update(extra)

        _atomic_save(payload, self._best_model_path())
        self.log.info(
            f"[Checkpoint] best_model.pt updated | "
            f"epoch={epoch} | {self.monitor}={metric_value:.4f}"
        )

    # ─────────────────────────────────────────────────────────
    # C — PERIODIC SNAPSHOT + RETENTION POLICY
    # ─────────────────────────────────────────────────────────

    def save_periodic(self, epoch: int, model: torch.nn.Module) -> None:
        """
        Save a periodic snapshot (model weights only).
        Applies retention policy: keep only last `keep_last_n_periodic` files.
        """
        if self.save_periodic_every <= 0:
            return
        if (epoch + 1) % self.save_periodic_every != 0:
            return

        path = self._periodic_path(epoch)
        payload = {
            "epoch":       epoch,
            "timestamp":   datetime.now().isoformat(),
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        }
        _atomic_save(payload, path)
        self._periodic_saved.append(path)
        self.log.info(f"[Checkpoint] Periodic saved → periodic_epoch_{epoch:04d}.pt")

        # Retention policy
        self._apply_periodic_retention()

    def _scan_existing_periodic(self) -> List[str]:
        """Find any existing periodic checkpoints (for pipeline restarts)."""
        import glob
        pattern = os.path.join(self.ckpt_dir, "periodic_epoch_*.pt")
        existing = sorted(glob.glob(pattern))
        return existing

    def _apply_periodic_retention(self) -> None:
        """Delete oldest periodic checkpoints, keeping only last N."""
        keep = self.keep_last_n_periodic
        while len(self._periodic_saved) > keep:
            oldest = self._periodic_saved.pop(0)
            if os.path.exists(oldest):
                try:
                    os.remove(oldest)
                    self.log.debug(f"[Checkpoint] Deleted old periodic: {os.path.basename(oldest)}")
                except OSError as e:
                    self.log.warning(f"[Checkpoint] Could not delete {oldest}: {e}")

    # ─────────────────────────────────────────────────────────
    # D — TRAINING STATE JSON
    # ─────────────────────────────────────────────────────────

    def update_state(
        self,
        epoch: int,
        best_epoch: int,
        best_metric: float,
        interrupted: bool = False,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Write training_state.json with current progress.
        Human-readable; used by search system to assess run status.
        """
        state = {
            "last_epoch":       epoch,
            "best_epoch":       best_epoch,
            "best_metric":      best_metric,
            "monitor":          self.monitor,
            "interrupted_flag": interrupted,
            "last_update_time": datetime.now().isoformat(),
        }
        if extra:
            state.update(extra)

        _atomic_json_save(state, self._state_json_path())

    # ─────────────────────────────────────────────────────────
    # EMERGENCY CHECKPOINT (on crash / OOM / Ctrl+C)
    # ─────────────────────────────────────────────────────────

    def save_emergency(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler=None,
        ema=None,
        best_metric: float = 0.0,
    ) -> None:
        """
        Fast, best-effort checkpoint on unexpected termination.
        Saves enough to resume training.
        """
        path = self._emergency_path(epoch)
        try:
            payload = {
                "epoch":       epoch,
                "best_metric": best_metric,
                "config_hash": self.config_hash,
                "timestamp":   datetime.now().isoformat(),
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "ema_state": (
                    {k: v.cpu() for k, v in ema.state_dict().items()}
                    if ema is not None else None
                ),
            }
            torch.save(payload, path)   # non-atomic — speed matters here
            self.log.info(f"[Checkpoint] Emergency checkpoint saved → {os.path.basename(path)}")
        except Exception as e:
            self.log.error(f"[Checkpoint] Emergency save FAILED: {e}")

        # Mark run as interrupted in state JSON
        try:
            state_path = self._state_json_path()
            if os.path.exists(state_path):
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            else:
                state = {}
            state["interrupted_flag"] = True
            state["last_update_time"] = datetime.now().isoformat()
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────
    # RESUME LOADING
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def find_resume_checkpoint(exp_path: str) -> Optional[str]:
        """
        Auto-find the best checkpoint to resume from in an experiment folder.

        Priority:
          1. checkpoint_resume_latest.pt  (most complete)
          2. Most recent emergency_epoch_*.pt
          3. None (start fresh)
        """
        ckpt_dir = os.path.join(exp_path, "checkpoints")

        latest = os.path.join(ckpt_dir, "checkpoint_resume_latest.pt")
        if os.path.exists(latest):
            return latest

        # Fall back to emergency checkpoints
        import glob
        emergencies = sorted(glob.glob(os.path.join(ckpt_dir, "emergency_epoch_*.pt")))
        if emergencies:
            return emergencies[-1]

        return None

    def load_resume(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: Optional[torch.cuda.amp.GradScaler],
        ema=None,
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a resume checkpoint and restore all training states.

        Returns a dict with:
            start_epoch, global_step, best_metric
        """
        self.log.info(f"[Checkpoint] Loading resume checkpoint: {path}")
        map_location = device if device is not None else "cpu"
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

        # Config hash check
        saved_hash = ckpt.get("config_hash", None)
        if saved_hash and saved_hash != self.config_hash:
            self.log.warning(
                f"[Checkpoint] Config hash mismatch! "
                f"saved={saved_hash} current={self.config_hash}. "
                f"Proceeding anyway — verify config manually."
            )

        # Restore model
        if "model_state" in ckpt:
            missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=strict)
            if missing:
                self.log.warning(f"[Checkpoint] Missing keys: {missing}")
            if unexpected:
                self.log.warning(f"[Checkpoint] Unexpected keys: {unexpected}")

        # Restore optimizer
        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])

        # Restore scheduler
        if "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
            _safe_scheduler_load(scheduler, ckpt["scheduler_state"])

        # Restore AMP scaler
        if scaler is not None and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
            scaler.load_state_dict(ckpt["scaler_state"])

        # Restore EMA
        if ema is not None and "ema_state" in ckpt and ckpt["ema_state"] is not None:
            ema.ema_model.load_state_dict(ckpt["ema_state"])
            self.log.info("[Checkpoint] EMA state restored.")

        # Restore RNG states (full reproducibility)
        if "rng_states" in ckpt:
            try:
                _restore_rng_states(ckpt["rng_states"])
            except Exception as e:
                self.log.warning(f"[Checkpoint] RNG state restore failed: {e}")

        start_epoch  = int(ckpt.get("epoch", 0)) + 1  # resume from next epoch
        global_step  = int(ckpt.get("global_step", 0))
        best_metric  = float(ckpt.get("best_metric", 0.0))

        self.log.info(
            f"[Checkpoint] Resumed from epoch {start_epoch - 1} | "
            f"global_step={global_step} | best_metric={best_metric:.4f}"
        )

        return {
            "start_epoch":  start_epoch,
            "global_step":  global_step,
            "best_metric":  best_metric,
        }


# ══════════════════════════════════════════════════════════════
#  SCHEDULER STATE HELPERS
# ══════════════════════════════════════════════════════════════

def _safe_scheduler_state(scheduler) -> Optional[Dict]:
    """
    Extract state_dict from any scheduler type.
    Custom schedulers (like PolyLR) may not have state_dict().
    """
    if scheduler is None:
        return None
    if hasattr(scheduler, "state_dict"):
        try:
            return scheduler.state_dict()
        except Exception:
            pass
    # Fallback: store the whole object's __dict__ minus non-serializable
    try:
        d = {k: v for k, v in scheduler.__dict__.items()
             if isinstance(v, (int, float, bool, str, list, dict))}
        d["__class__"] = scheduler.__class__.__name__
        return d
    except Exception:
        return None


def _safe_scheduler_load(scheduler, state: Dict) -> None:
    """
    Load scheduler state. Handles torch schedulers and custom ones.
    """
    if scheduler is None or state is None:
        return
    cls_name = state.get("__class__", "")
    if hasattr(scheduler, "load_state_dict") and "__class__" not in state:
        try:
            scheduler.load_state_dict(state)
            return
        except Exception:
            pass
    # Custom PolyLR: restore __dict__ fields
    for k, v in state.items():
        if k != "__class__" and hasattr(scheduler, k):
            try:
                setattr(scheduler, k, v)
            except Exception:
                pass
