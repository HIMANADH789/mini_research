"""
tracking_utils.py
=================
Extended experiment tracking — epoch-level curves, GPU memory timeline,
fine-grained timing (dataloader/forward/backward/optimizer), and
failure/resume logging.

This module extends the existing ExperimentTracker with rich per-epoch
time series data that is saved as both CSV (for Excel/spreadsheets) and
JSON (for programmatic use).

Saved files (in experiment_dir/runs/):
    training_curves.csv        — per-epoch: loss, dice, lr, grad_norm, gpu_mem, ...
    training_curves.json       — same data in JSON (used by PlotManager)
    timing.json                — wall-clock breakdown: epoch/val/inference + fine-grained
    failure_resume.json        — interruption reason, resume count, traceback, last ckpt

Key design principle — raw data storage:
    All curve data is saved as JSON/CSV so plots can be regenerated
    at any time in any format (PNG, SVG, PDF) without re-running training.
    The PlotManager reads from training_curves.json to render plots on demand.

Usage in trainer (or via run_experiment.py hooks):
    from src.utils.tracking_utils import CurveTracker, FailureLogger, TimingTracker

    curve_tracker   = CurveTracker(exp_path)
    failure_logger  = FailureLogger(exp_path)
    timing_tracker  = TimingTracker(exp_path)
    timing_tracker.training_start()

    # Per epoch (called via trainer._on_epoch_end hook):
    curve_tracker.record(epoch, {
        "train_loss": avg_loss, "val_dice": val_mean,
        "lr": lr_now, "grad_norm": avg_gn,
    })
    timing_tracker.record_epoch(epoch, duration_s, n_samples=batch_size * steps)
    curve_tracker.save()          # incremental save after each epoch

    # Fine-grained timing (optional, from trainer internals):
    timing_tracker.record_dataloader_time(epoch, dl_secs)
    timing_tracker.record_forward_time(epoch, fwd_secs)
    timing_tracker.record_backward_time(epoch, bwd_secs)
    timing_tracker.record_optimizer_step_time(epoch, opt_secs)

    # On resume:
    failure_logger.record_resume(from_checkpoint=ckpt_path, at_epoch=start_epoch)

    # On completion:
    timing_tracker.training_end()
    timing_tracker.save()
    failure_logger.mark_completed(final_epoch, last_checkpoint_path)
"""

import os
import csv
import json
import time
import logging
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _atomic_json(obj: Any, path: str) -> None:
    """Write JSON atomically (tempfile + os.replace). Windows NTFS safe."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        os.close(fd)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def _write_csv(rows: List[Dict], path: str) -> None:
    """Write a list of dicts to CSV. Skips if empty."""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Collect all field names in insertion order
    all_keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
#  CURVE TRACKER — per-epoch training curves
# ══════════════════════════════════════════════════════════════

class CurveTracker:
    """
    Records per-epoch metrics as time-series data.

    Supports ANY metric key — just pass a dict per epoch.
    All data is saved as JSON + CSV so plots can be regenerated
    at any time in any format (PNG, SVG, PDF) using PlotManager.

    Common keys recorded:
        train_loss, val_loss
        train_dice, val_dice, val_tc_dice, val_wt_dice, val_et_dice
        lr, grad_norm, param_norm
        gpu_mem_gb, epoch_duration_s, samples_per_sec
        condition_number, rank_k        (SABiT spectral)
        val_hd95

    Saved to:
        runs/training_curves.csv
        runs/training_curves.json
        curves/training_curves.json   (copy in curves/ subdir)
    """

    def __init__(self, exp_path: str):
        self.exp_path  = exp_path
        self.runs_dir  = os.path.join(exp_path, "runs")
        self.curves_dir = os.path.join(exp_path, "curves")
        os.makedirs(self.runs_dir,   exist_ok=True)
        os.makedirs(self.curves_dir, exist_ok=True)

        self._rows: List[Dict] = []
        self._csv_path  = os.path.join(self.runs_dir, "training_curves.csv")
        self._json_path = os.path.join(self.runs_dir, "training_curves.json")
        self._curves_json = os.path.join(self.curves_dir, "training_curves.json")

        # Load existing rows on resume
        self._rows = self._load_existing()

    def _load_existing(self) -> List[Dict]:
        """On resume, read existing JSON to continue appending."""
        if not os.path.exists(self._json_path):
            return []
        try:
            with open(self._json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("epochs", [])
        except Exception:
            return []

    def record(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        gpu_mem_gb: Optional[float] = None,
    ) -> None:
        """
        Record metrics for one epoch.

        Parameters
        ----------
        epoch      : zero-based epoch index
        metrics    : dict of metric_name → value (any keys accepted)
        gpu_mem_gb : optional GPU memory snapshot (auto-read if None)
        """
        row: Dict[str, Any] = {"epoch": epoch}

        # Serialize metrics — round floats for cleanliness
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                row[k] = round(float(v), 7) if isinstance(v, float) else v
            except (TypeError, ValueError):
                row[k] = v

        # Auto-read GPU memory
        if gpu_mem_gb is None:
            gpu_mem_gb = _safe(lambda: _read_gpu_mem_gb())
        if gpu_mem_gb is not None:
            row["gpu_mem_gb"] = gpu_mem_gb

        # Overwrite if epoch already recorded (resumable runs)
        existing_epochs = {r["epoch"] for r in self._rows}
        if epoch in existing_epochs:
            self._rows = [row if r["epoch"] == epoch else r for r in self._rows]
        else:
            self._rows.append(row)

    def save(self) -> None:
        """
        Write CSV and JSON. Safe to call incrementally after each epoch.
        Also writes a copy to curves/ directory for PlotManager.
        Raw data in JSON allows PlotManager to regenerate any plot format
        (PNG, SVG, PDF) at any time without re-running training.
        """
        if not self._rows:
            return
        try:
            payload = {
                "generated_at":  datetime.now(timezone.utc).isoformat(),
                "n_epochs":      len(self._rows),
                "metric_keys":   list(self._rows[0].keys()) if self._rows else [],
                "epochs":        self._rows,
            }
            _write_csv(self._rows, self._csv_path)
            _atomic_json(payload, self._json_path)
            # Mirror to curves/ directory
            _atomic_json(payload, self._curves_json)
        except Exception as e:
            logger.warning(f"[CurveTracker] Save failed: {e}")

    def get_rows(self) -> List[Dict]:
        return list(self._rows)

    @property
    def csv_path(self) -> str:
        return self._csv_path

    @property
    def json_path(self) -> str:
        return self._json_path

    @property
    def curves_json_path(self) -> str:
        return self._curves_json


# ══════════════════════════════════════════════════════════════
#  FAILURE LOGGER — interruption and resume logging
# ══════════════════════════════════════════════════════════════

class FailureLogger:
    """
    Tracks interruptions, resumes, crash tracebacks, and failure causes.

    Saves to: runs/failure_resume.json  (canonical path for the evidence package)

    Design:
      - Records every resume event with checkpoint path + epoch
      - Records failures with full traceback
      - On completion, marks run as clean with final checkpoint path
      - Partial metrics are saved on failure if available
    """

    def __init__(self, exp_path: str):
        self.runs_dir = os.path.join(exp_path, "runs")
        os.makedirs(self.runs_dir, exist_ok=True)
        # New canonical path: failure_resume.json
        self._path     = os.path.join(self.runs_dir, "failure_resume.json")
        self._data     = self._load()
        self._last_checkpoint: Optional[str] = None
        self._partial_metrics: Dict = {}

    def _load(self) -> Dict:
        # Support both old name (failure_log.json) and new name
        for candidate in [self._path,
                          os.path.join(self.runs_dir, "failure_log.json")]:
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
        return {
            "interrupted":            False,
            "interruption_reason":    None,
            "traceback":              None,
            "last_successful_epoch":  None,
            "last_saved_checkpoint":  None,
            "partial_metrics_before_failure": {},
            "resume_count":           0,
            "resumes":                [],
            "failures":               [],
            "completed":              False,
            "completed_at":           None,
        }

    def _save(self) -> None:
        try:
            _atomic_json(self._data, self._path)
        except Exception as e:
            logger.warning(f"[FailureLogger] Save failed: {e}")

    def set_last_checkpoint(self, path: str) -> None:
        """Call whenever a checkpoint is saved — tracks latest safe checkpoint."""
        self._last_checkpoint = path
        self._data["last_saved_checkpoint"] = path

    def set_partial_metrics(self, metrics: Dict) -> None:
        """Call periodically (e.g. after each epoch) to capture last-known metrics."""
        self._partial_metrics = {
            k: round(float(v), 6) if isinstance(v, float) else v
            for k, v in metrics.items()
        }
        self._data["partial_metrics_before_failure"] = self._partial_metrics

    def record_resume(self, from_checkpoint: str, at_epoch: int) -> None:
        """Call when training resumes from a checkpoint."""
        self._data["resume_count"] += 1
        self._data["interrupted"]  = False
        self._data["resumes"].append({
            "resume_number":   self._data["resume_count"],
            "from_checkpoint": from_checkpoint,
            "at_epoch":        at_epoch,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        })
        logger.info(
            f"[FailureLogger] Resume #{self._data['resume_count']} "
            f"from checkpoint at epoch {at_epoch}"
        )
        self._save()

    def record_epoch_completed(self, epoch: int) -> None:
        """Call at the end of each epoch to update last_successful_epoch."""
        self._data["last_successful_epoch"] = epoch
        self._data["interrupted"] = False
        # Don't save every epoch — do it periodically or on failure

    def record_failure(
        self,
        reason: str,
        last_epoch: int,
        traceback_str: Optional[str] = None,
        partial_metrics: Optional[Dict] = None,
    ) -> None:
        """Call when training fails or is interrupted."""
        self._data["interrupted"]           = True
        self._data["interruption_reason"]   = reason
        self._data["last_successful_epoch"] = last_epoch
        self._data["traceback"]             = traceback_str
        self._data["last_saved_checkpoint"] = self._last_checkpoint

        if partial_metrics:
            self._data["partial_metrics_before_failure"] = {
                k: round(float(v), 6) if isinstance(v, float) else v
                for k, v in partial_metrics.items()
            }

        self._data["failures"].append({
            "reason":     reason,
            "last_epoch": last_epoch,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "traceback":  traceback_str,
            "last_checkpoint": self._last_checkpoint,
        })
        self._save()

    def mark_completed(
        self,
        final_epoch: int,
        last_checkpoint_path: Optional[str] = None,
    ) -> None:
        """Call when training finishes successfully."""
        self._data["interrupted"]           = False
        self._data["interruption_reason"]   = None
        self._data["traceback"]             = None
        self._data["last_successful_epoch"] = final_epoch
        self._data["completed"]             = True
        self._data["completed_at"]          = datetime.now(timezone.utc).isoformat()
        if last_checkpoint_path:
            self._data["last_saved_checkpoint"] = last_checkpoint_path
        self._save()

    def save(self) -> None:
        self._save()


# ══════════════════════════════════════════════════════════════
#  TIMING TRACKER — fine-grained wall-clock timing
# ══════════════════════════════════════════════════════════════

class TimingTracker:
    """
    Records detailed wall-clock timing for training stages.

    Tracks:
      - Total training wall time
      - Per-epoch duration
      - Per-epoch: dataloader time, forward pass, backward pass, optimizer step
      - Validation pass duration per epoch
      - Inference batch timing
      - Samples/sec and iters/sec throughput

    Saved to: runs/timing.json  (also mirrored to runs/runtime.json for paper package)
    """

    def __init__(self, exp_path: str):
        self.runs_dir = os.path.join(exp_path, "runs")
        os.makedirs(self.runs_dir, exist_ok=True)
        self._path = os.path.join(self.runs_dir, "timing.json")

        # Per-epoch records (list of dicts, keyed by epoch)
        self._epoch_records: Dict[int, Dict[str, Any]] = {}
        self._val_times:     List[Dict] = []
        self._inference_times: List[float] = []

        # Wall clocks
        self._total_start: Optional[float] = None
        self._total_end:   Optional[float] = None

    # ─────────────────────────────────────────────────────────
    # WALL CLOCK
    # ─────────────────────────────────────────────────────────

    def training_start(self) -> None:
        self._total_start = time.monotonic()

    def training_end(self) -> None:
        self._total_end = time.monotonic()

    # ─────────────────────────────────────────────────────────
    # PER-EPOCH RECORDING
    # ─────────────────────────────────────────────────────────

    def _epoch_record(self, epoch: int) -> Dict[str, Any]:
        if epoch not in self._epoch_records:
            self._epoch_records[epoch] = {"epoch": epoch}
        return self._epoch_records[epoch]

    def record_epoch(
        self,
        epoch: int,
        duration_s: float,
        n_samples: int = 0,
        n_iters: int = 0,
    ) -> None:
        """Record total wall time for one training epoch."""
        rec = self._epoch_record(epoch)
        rec["duration_s"] = round(duration_s, 3)
        if n_samples > 0 and duration_s > 0:
            rec["samples_per_sec"] = round(n_samples / duration_s, 2)
        if n_iters > 0 and duration_s > 0:
            rec["iters_per_sec"] = round(n_iters / duration_s, 2)

    def record_dataloader_time(self, epoch: int, duration_s: float) -> None:
        """Record time spent in DataLoader (data loading + augmentation)."""
        self._epoch_record(epoch)["dataloader_s"] = round(duration_s, 4)

    def record_forward_time(self, epoch: int, duration_s: float) -> None:
        """Record time for the forward pass (model inference during training)."""
        self._epoch_record(epoch)["forward_pass_s"] = round(duration_s, 4)

    def record_backward_time(self, epoch: int, duration_s: float) -> None:
        """Record time for loss.backward()."""
        self._epoch_record(epoch)["backward_pass_s"] = round(duration_s, 4)

    def record_optimizer_step_time(self, epoch: int, duration_s: float) -> None:
        """Record time for optimizer.step()."""
        self._epoch_record(epoch)["optimizer_step_s"] = round(duration_s, 4)

    def record_gpu_utilization(self, epoch: int, utilization_pct: float) -> None:
        """Record mean GPU utilization percentage for this epoch."""
        self._epoch_record(epoch)["gpu_utilization_pct"] = round(float(utilization_pct), 1)

    def record_validation(self, epoch: int, duration_s: float) -> None:
        """Record wall time for one validation pass."""
        self._val_times.append({
            "epoch":      epoch,
            "duration_s": round(duration_s, 3),
        })

    def record_inference(self, batch_times_s: List[float]) -> None:
        """Record per-batch inference times (from evaluation pass)."""
        self._inference_times.extend(batch_times_s)

    # ─────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write timing.json with full breakdown."""
        total_s = (
            round(self._total_end - self._total_start, 2)
            if self._total_start and self._total_end else None
        )

        per_epoch = sorted(self._epoch_records.values(), key=lambda r: r["epoch"])
        durations  = [r["duration_s"] for r in per_epoch if "duration_s" in r]

        avg_epoch  = round(sum(durations) / len(durations), 3) if durations else None
        min_epoch  = round(min(durations), 3) if durations else None
        max_epoch  = round(max(durations), 3) if durations else None

        val_durs   = [v["duration_s"] for v in self._val_times]
        avg_val    = round(sum(val_durs) / len(val_durs), 3) if val_durs else None

        inf_times  = self._inference_times
        avg_inf    = round(sum(inf_times) / len(inf_times), 4) if inf_times else None
        total_inf  = round(sum(inf_times), 3) if inf_times else None

        # Aggregate fine-grained timing
        dl_times  = [r["dataloader_s"]    for r in per_epoch if "dataloader_s" in r]
        fwd_times = [r["forward_pass_s"]  for r in per_epoch if "forward_pass_s" in r]
        bwd_times = [r["backward_pass_s"] for r in per_epoch if "backward_pass_s" in r]
        opt_times = [r["optimizer_step_s"] for r in per_epoch if "optimizer_step_s" in r]

        def avg(lst): return round(sum(lst) / len(lst), 4) if lst else None

        throughput = None
        samples_per_sec_list = [r["samples_per_sec"] for r in per_epoch
                                 if "samples_per_sec" in r]
        if samples_per_sec_list:
            throughput = round(sum(samples_per_sec_list) / len(samples_per_sec_list), 2)

        data = {
            # Summary
            "total_training_s":         total_s,
            "total_training_human":     _fmt_duration(total_s),
            "avg_epoch_duration_s":     avg_epoch,
            "min_epoch_duration_s":     min_epoch,
            "max_epoch_duration_s":     max_epoch,
            "avg_validation_time_s":    avg_val,
            "avg_inference_time_s":     avg_inf,
            "total_inference_time_s":   total_inf,
            "avg_samples_per_sec":      throughput,

            # Fine-grained averages
            "avg_dataloader_time_s":    avg(dl_times),
            "avg_forward_pass_time_s":  avg(fwd_times),
            "avg_backward_pass_time_s": avg(bwd_times),
            "avg_optimizer_step_s":     avg(opt_times),

            # Per-epoch timeline
            "per_epoch":         per_epoch,
            "per_validation":    self._val_times,
            "inference_batch_times_s": [round(t, 4) for t in inf_times[:200]],

            # Metadata
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            _atomic_json(data, self._path)
        except Exception as e:
            logger.warning(f"[TimingTracker] Save failed: {e}")


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _fmt_duration(s: Optional[float]) -> str:
    """Format seconds into human-readable string."""
    if s is None:
        return "N/A"
    s = max(0, int(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else (f"{m}m {sec:02d}s" if m else f"{sec}s")


def _read_gpu_mem_gb() -> Optional[float]:
    """Read current GPU memory allocation in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1e9, 3)
    except Exception:
        pass
    return None
