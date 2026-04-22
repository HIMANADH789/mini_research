"""
experiment_tracker.py
=====================
Central metadata tracker for the mini_research pipeline.

Coordinates the collection of system, model, and runtime info and
writes everything required for a self-contained evidence package:

    experiment_dir/
        system_info.json       — hardware / software environment
        model_info.json        — architecture, parameters, class path
        runtime.json           — timing totals (human-readable)
        summary.json           — high-level run summary
        runs/
            timing.json        — detailed per-epoch timing (from TimingTracker)
            failure_resume.json

All functions imported from system_utils for single source of truth.
Old function signatures are preserved as module-level re-exports
for backward compatibility.
"""

import os
import json
import logging
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ── System utilities (single source of truth) ─────────────────────────────────
from src.utils.system_utils import (
    collect_system_info       as _collect_system_info,
    collect_gpu_memory_used   as _collect_gpu_memory_used,
    collect_git_info          as _collect_git_info,
    collect_environment_info  as _collect_environment_info,
    count_parameters          as _count_parameters,
    config_hash               as _config_hash,
    format_duration,
    atomic_json_write,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  BACKWARD-COMPATIBLE MODULE-LEVEL EXPORTS
#  (existing code that calls these directly still works)
# ══════════════════════════════════════════════════════════════

def collect_system_info() -> Dict[str, Any]:
    """Backward-compat wrapper → system_utils.collect_system_info()."""
    return _collect_system_info()


def collect_git_info() -> Dict[str, Any]:
    """Backward-compat wrapper → system_utils.collect_git_info()."""
    return _collect_git_info()


def count_parameters(model) -> Dict[str, Any]:
    """Backward-compat wrapper → system_utils.count_parameters()."""
    return _count_parameters(model)


def config_hash(config) -> str:
    """Backward-compat wrapper → system_utils.config_hash()."""
    return _config_hash(config)


def _safe(fn, default="unavailable"):
    try:
        return fn()
    except Exception:
        return default


def _atomic_json_save(obj: Any, path: str) -> None:
    """Backward-compat wrapper for atomic JSON write."""
    atomic_json_write(obj, path)


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT TRACKER
# ══════════════════════════════════════════════════════════════

class ExperimentTracker:
    """
    Writes all metadata JSON files for one experiment:
        system_info.json    — hardware + software snapshot
        model_info.json     — architecture, params, git, config hash
        runtime.json        — timing summary (human readable)
        summary.json        — overview of the whole run

    Usage:
        tracker = ExperimentTracker(exp_path, config, run_id="exp_001")
        tracker.start()                       # at training start
        tracker.set_model_info(model)         # after model init
        tracker.set_best(epoch, name, value)  # after validation
        tracker.finish(final_metrics={...})   # on completion
        tracker.save()                        # write all JSONs
        tracker.print_summary()               # pretty-print results
    """

    def __init__(self, exp_path: str, config, run_id: Optional[str] = None):
        self.exp_path  = exp_path
        self.config    = config
        self.run_id    = run_id or os.path.basename(exp_path)

        # Paths
        self._sys_path      = os.path.join(exp_path, "system_info.json")
        self._model_path    = os.path.join(exp_path, "model_info.json")
        self._runtime_path  = os.path.join(exp_path, "runtime.json")
        self._summary_path  = os.path.join(exp_path, "summary.json")

        # Collected data
        self._system_info:  Dict[str, Any] = {}
        self._model_info:   Dict[str, Any] = {}
        self._runtime_info: Dict[str, Any] = {}
        self._summary:      Dict[str, Any] = {}
        self._gpu_mem:      Dict[str, Any] = {}
        self._checkpoint_paths: Dict[str, str] = {}

        # Internal state
        self._start_iso:    Optional[str] = None
        self._start_epoch_count: int = 0

    # ─────────────────────────────────────────────────────────
    # START (call at experiment initialization)
    # ─────────────────────────────────────────────────────────

    def start(self, model=None) -> None:
        """
        Collect system info, model info, and write initial JSON files.
        Call this at the very beginning of the experiment, after model init.
        """
        self._start_iso = datetime.now(timezone.utc).isoformat()

        # ── System info ─────────────────────────────────────
        self._system_info = _collect_system_info()
        self._system_info.update(_collect_git_info())

        # ── Model info ──────────────────────────────────────
        self._model_info = self._build_model_info(model)

        # ── Runtime (initial, updated on completion) ────────
        self._runtime_info = {
            "start_time":            self._start_iso,
            "end_time":              None,
            "total_training_s":      None,
            "total_training_human":  None,
            "avg_epoch_duration_s":  None,
            "avg_validation_time_s": None,
            "avg_inference_time_s":  None,
            "avg_samples_per_sec":   None,
            "status":                "running",
        }

        # ── Summary (initial) ───────────────────────────────
        self._summary = {
            "experiment":          os.path.basename(self.exp_path),
            "run_id":              self.run_id,
            "model_name":          _safe(lambda: self.config.name),
            "seed":                _safe(lambda: self.config.seed),
            "status":              "running",
            "start_time":          self._start_iso,
            "end_time":            None,
            "best_epoch":          None,
            "best_metric":         None,
            "best_metric_name":    None,
            "monitor":             _safe(lambda: self.config.checkpoint.monitor, "N/A"),
            "machine":             self._system_info.get("hostname", "N/A"),
            "gpu_name":            self._system_info.get("gpu_name", "N/A"),
            "total_params":        self._model_info.get("total_params", "N/A"),
            "config_hash":         self._model_info.get("config_hash", "N/A"),
            "git_commit":          self._system_info.get("git_commit", "N/A"),
            "checkpoint_paths":    {},
        }

        # Write immediately so evidence exists even if crash
        self.save()
        logger.info(f"[ExperimentTracker] Initial metadata saved → {self.exp_path}/")

    # ─────────────────────────────────────────────────────────
    # UPDATE RUNTIME (call on completion or periodically)
    # ─────────────────────────────────────────────────────────

    def update_runtime(
        self,
        total_training_s:      Optional[float] = None,
        avg_epoch_duration_s:  Optional[float] = None,
        avg_validation_s:      Optional[float] = None,
        avg_inference_s:       Optional[float] = None,
        avg_samples_per_sec:   Optional[float] = None,
        avg_dataloader_s:      Optional[float] = None,
        avg_forward_s:         Optional[float] = None,
        avg_backward_s:        Optional[float] = None,
        avg_optimizer_s:       Optional[float] = None,
        status:                str             = "running",
        extra:                 Optional[Dict]  = None,
    ) -> None:
        """
        Update runtime info with timing summary from TimingTracker.
        Reads gpu_memory_used automatically.
        """
        end_iso = datetime.now(timezone.utc).isoformat()

        self._runtime_info.update({
            "end_time":              end_iso,
            "total_training_s":      total_training_s,
            "total_training_human":  format_duration(total_training_s),
            "avg_epoch_duration_s":  avg_epoch_duration_s,
            "avg_validation_time_s": avg_validation_s,
            "avg_inference_time_s":  avg_inference_s,
            "avg_samples_per_sec":   avg_samples_per_sec,
            "avg_dataloader_time_s": avg_dataloader_s,
            "avg_forward_pass_time_s":  avg_forward_s,
            "avg_backward_pass_time_s": avg_backward_s,
            "avg_optimizer_step_s":     avg_optimizer_s,
            "status":                status,
        })

        # GPU peak memory
        gpu_mem = _safe(lambda: _collect_gpu_memory_used(), {})
        self._runtime_info.update(gpu_mem)

        # System identifiers (for reproducibility)
        self._runtime_info["hostname"]  = self._system_info.get("hostname", "N/A")
        self._runtime_info["gpu_name"]  = self._system_info.get("gpu_name", "N/A")
        self._runtime_info["cuda_version"] = self._system_info.get("cuda_version", "N/A")

        if extra:
            self._runtime_info.update(extra)

    def update_runtime_from_timing(self, timing_data: Dict) -> None:
        """
        Convenience: update runtime directly from a timing.json dict
        (as produced by TimingTracker.save()).
        """
        self.update_runtime(
            total_training_s      = timing_data.get("total_training_s"),
            avg_epoch_duration_s  = timing_data.get("avg_epoch_duration_s"),
            avg_validation_s      = timing_data.get("avg_validation_time_s"),
            avg_inference_s       = timing_data.get("avg_inference_time_s"),
            avg_samples_per_sec   = timing_data.get("avg_samples_per_sec"),
            avg_dataloader_s      = timing_data.get("avg_dataloader_time_s"),
            avg_forward_s         = timing_data.get("avg_forward_pass_time_s"),
            avg_backward_s        = timing_data.get("avg_backward_pass_time_s"),
            avg_optimizer_s       = timing_data.get("avg_optimizer_step_s"),
        )

    # ─────────────────────────────────────────────────────────
    # SET MODEL INFO (call after trainer builds the model)
    # ─────────────────────────────────────────────────────────

    def set_model_info(self, model) -> None:
        """
        Update model_info.json with architecture + parameter details.
        Call after the trainer has built the model.
        """
        if model is None:
            return
        param_info = _safe(lambda: _count_parameters(model), {})
        self._model_info.update(param_info)
        self._summary["total_params"] = self._model_info.get("total_params", "N/A")
        self.save()

    # ─────────────────────────────────────────────────────────
    # SET CHECKPOINT PATHS
    # ─────────────────────────────────────────────────────────

    def set_checkpoint_paths(
        self,
        best:   Optional[str] = None,
        resume: Optional[str] = None,
    ) -> None:
        """Record which checkpoint files exist for this experiment."""
        if best:
            self._checkpoint_paths["best"] = best
        if resume:
            self._checkpoint_paths["resume"] = resume
        self._summary["checkpoint_paths"] = dict(self._checkpoint_paths)

    # ─────────────────────────────────────────────────────────
    # SET BEST (record best metric from trainer)
    # ─────────────────────────────────────────────────────────

    def set_best(
        self,
        epoch:       int,
        metric_name: str,
        value:       float,
    ) -> None:
        """Record the best validation metric found during training."""
        self._summary["best_epoch"]       = epoch
        self._summary["best_metric"]      = round(float(value), 6)
        self._summary["best_metric_name"] = metric_name

    # ─────────────────────────────────────────────────────────
    # FINALIZE / FINISH (call on completion)
    # ─────────────────────────────────────────────────────────

    def finalize(
        self,
        best_epoch:     Optional[int]   = None,
        best_metric:    Optional[float] = None,
        final_metrics:  Optional[Dict]  = None,
        status:         str             = "completed",
    ) -> None:
        """
        Update summary with final results and mark run as completed.
        """
        end_iso = datetime.now(timezone.utc).isoformat()
        self._summary["status"]   = status
        self._summary["end_time"] = end_iso

        if best_epoch is not None:
            self._summary["best_epoch"] = best_epoch
        if best_metric is not None:
            self._summary["best_metric"] = round(best_metric, 6) if isinstance(best_metric, float) else best_metric

        if final_metrics:
            normalized = {
                k: round(float(v), 6) if isinstance(v, float) else v
                for k, v in final_metrics.items()
            }
            self._summary["final_metrics"] = normalized

        # Collect GPU peak memory at end of training
        self._gpu_mem = _safe(lambda: _collect_gpu_memory_used(), {})
        self._runtime_info.update(self._gpu_mem)

        # Update runtime status
        self._runtime_info["status"] = status
        self._runtime_info["end_time"] = end_iso
        self.save()
        logger.info(f"[ExperimentTracker] Finalized: {status} — {self.exp_path}")

    def finish(
        self,
        final_metrics: Optional[Dict] = None,
        status:        str            = "completed",
    ) -> None:
        """
        Convenience alias for finalize(). Used by run_experiment.py.
        Reads best_epoch / best_metric from summary (set via set_best()).
        """
        self.finalize(
            best_epoch    = self._summary.get("best_epoch"),
            best_metric   = self._summary.get("best_metric"),
            final_metrics = final_metrics,
            status        = status,
        )

    # ─────────────────────────────────────────────────────────
    # PRINT SUMMARY
    # ─────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """
        Print a formatted summary box to stdout for quick visual feedback.
        """
        s = self._summary
        gpu = self._system_info.get("gpu_name", "N/A")
        total_p = self._model_info.get("total_params", "N/A")
        if isinstance(total_p, int):
            total_p_str = f"{total_p:,}"
        else:
            total_p_str = str(total_p)

        best = s.get("best_metric", "N/A")
        best_name = s.get("best_metric_name", "metric")
        best_epoch = s.get("best_epoch", "N/A")
        status = s.get("status", "unknown")
        config_hash = s.get("config_hash", "N/A")

        lines = [
            "",
            "╔" + "═" * 58 + "╗",
            "║" + "  EXPERIMENT SUMMARY".ljust(58) + "║",
            "╠" + "═" * 58 + "╣",
            f"║  Experiment:  {s.get('experiment', 'N/A')[:40]}".ljust(59) + "║",
            f"║  Status:      {status}".ljust(59) + "║",
            f"║  GPU:         {gpu[:40]}".ljust(59) + "║",
            f"║  Parameters:  {total_p_str}".ljust(59) + "║",
            f"║  Config Hash: {config_hash}".ljust(59) + "║",
            "╠" + "─" * 58 + "╣",
            f"║  Best {best_name}: {best}  (epoch {best_epoch})".ljust(59) + "║",
        ]

        # Add final metrics if available
        final = s.get("final_metrics", {})
        if final:
            lines.append("╠" + "─" * 58 + "╣")
            for k, v in list(final.items())[:8]:
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                lines.append(f"║  {k}: {val_str}".ljust(59) + "║")

        lines.append("╚" + "═" * 58 + "╝")
        lines.append("")

        summary_text = "\n".join(lines)
        print(summary_text)
        logger.info(summary_text)

    # ─────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────

    def save(self) -> Dict[str, str]:
        """Write all metadata JSON files atomically. Returns dict of saved paths."""
        writes = [
            (self._system_info,  self._sys_path),
            (self._model_info,   self._model_path),
            (self._runtime_info, self._runtime_path),
            (self._summary,      self._summary_path),
        ]
        saved = {}
        for data, path in writes:
            try:
                atomic_json_write(data, path)
                saved[os.path.basename(path)] = path
            except Exception as e:
                logger.warning(f"[ExperimentTracker] Could not write {os.path.basename(path)}: {e}")
        return saved

    # ─────────────────────────────────────────────────────────
    # PROPERTIES (read-only access for other modules)
    # ─────────────────────────────────────────────────────────

    @property
    def system_info(self) -> Dict[str, Any]:
        return dict(self._system_info)

    @property
    def model_info(self) -> Dict[str, Any]:
        return dict(self._model_info)

    @property
    def runtime_info(self) -> Dict[str, Any]:
        return dict(self._runtime_info)

    # ─────────────────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────────────────

    def _build_model_info(self, model) -> Dict[str, Any]:
        """Build the model_info.json payload from config + model."""
        info: Dict[str, Any] = {}

        # ── Config fields ────────────────────────────────────
        config = self.config
        info["model_name"]       = _safe(lambda: config.name, "unknown")
        info["experiment_name"]  = os.path.basename(self.exp_path)
        info["seed"]             = _safe(lambda: config.seed, "N/A")
        info["config_hash"]      = _safe(lambda: _config_hash(config), "unavailable")

        # ── Model architecture ───────────────────────────────
        info["model_type"]       = _safe(lambda: config.model.type, "N/A")
        info["in_channels"]      = _safe(lambda: config.model.in_channels, "N/A")
        info["out_channels"]     = _safe(lambda: config.model.out_channels, "N/A")

        # ── Parameter counts ────────────────────────────────
        if model is not None:
            param_info = _safe(lambda: _count_parameters(model), {})
            info.update(param_info)
        else:
            info.update({
                "total_params":               "unavailable",
                "trainable_params":           "unavailable",
                "frozen_params":              "unavailable",
                "parameter_memory_estimate_mb": "unavailable",
                "model_class":                "unavailable",
                "model_class_path":           "unavailable",
            })

        # ── Git provenance ───────────────────────────────────
        git = _safe(lambda: _collect_git_info(), {})
        info["git_commit"]     = git.get("git_commit", "unavailable")
        info["git_branch"]     = git.get("git_branch", "unavailable")
        info["git_dirty"]      = git.get("git_dirty", "unavailable")
        info["git_commit_full"] = git.get("git_commit_full", "unavailable")

        # ── Training config snapshot ─────────────────────────
        info["training_config"] = {
            "epochs":          _safe(lambda: config.training.epochs, "N/A"),
            "batch_size":      _safe(lambda: config.training.batch_size, "N/A"),
            "lr":              _safe(lambda: config.training.lr, "N/A"),
            "lr_policy":       _safe(lambda: config.training.lr_policy, "N/A"),
            "optimizer":       _safe(lambda: config.optimizer.type, "N/A"),
            "mixed_precision": _safe(lambda: config.training.mixed_precision, "N/A"),
            "use_ema":         _safe(lambda: config.training.use_ema, "N/A"),
            "grad_clip":       _safe(lambda: config.training.grad_clip, "N/A"),
        }

        info["generated_at"] = datetime.now(timezone.utc).isoformat()
        return info

    # ─────────────────────────────────────────────────────────
    # STATIC: READ BACK
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def load_summary(exp_path: str) -> Optional[Dict]:
        """Read summary.json from an experiment directory."""
        path = os.path.join(exp_path, "summary.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def load_model_info(exp_path: str) -> Optional[Dict]:
        """Read model_info.json from an experiment directory."""
        path = os.path.join(exp_path, "model_info.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def load_runtime(exp_path: str) -> Optional[Dict]:
        """Read runtime.json from an experiment directory."""
        path = os.path.join(exp_path, "runtime.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
