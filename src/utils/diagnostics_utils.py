"""
diagnostics_utils.py
====================
Scientific diagnostics system for the mini_research pipeline.

Tracks deeper research value metrics that standard training loops miss:

A) Optimization Diagnostics
   - Gradient variance across batches
   - Update norm (optimizer step magnitude)
   - Grad/update ratio (are we taking reasonable steps?)
   - Cosine similarity between consecutive gradients
   - Loss landscape proxy stats

B) Calibration / Reliability
   - Confidence histogram (at evaluation time)
   - Expected Calibration Error (ECE) if softmax outputs available
   - Uncertainty summary

C) Generalization
   - Train vs val gap (per epoch)
   - Best epoch gap
   - Overfitting progress indicator

D) Stability
   - NaN incidents (epoch, batch)
   - Skipped batches
   - Gradient clipping events
   - Resume count

Saves to: experiment_dir/diagnostics/
    optimization_diagnostics.json
    calibration_diagnostics.json
    generalization_diagnostics.json
    stability_diagnostics.json
    diagnostics_summary.json

Usage in trainer:
    from src.utils.diagnostics_utils import DiagnosticsTracker

    diag = DiagnosticsTracker(exp_path)

    # Per batch (optional):
    diag.record_gradient_stats(epoch, batch, grad_variance=..., update_norm=...)
    diag.record_stability_event("nan", epoch=e, batch=b)
    diag.record_stability_event("grad_clip", epoch=e)

    # Per epoch (from curve data):
    diag.record_generalization(epoch, train_loss, val_loss, train_dice, val_dice)

    # At end of training:
    diag.finalize()
    diag.save()
"""

import os
import json
import math
import logging
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _atomic_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        os.close(fd)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# ══════════════════════════════════════════════════════════════
#  DIAGNOSTICS TRACKER
# ══════════════════════════════════════════════════════════════

class DiagnosticsTracker:
    """
    Tracks scientific diagnostics throughout training.

    All record_* methods are try/except guarded — never crash training.
    Call save() to persist all diagnostics to disk.

    Parameters
    ----------
    exp_path : str
        Root experiment directory.
    """

    def __init__(self, exp_path: str):
        self.exp_path    = exp_path
        self.diag_dir    = os.path.join(exp_path, "diagnostics")
        os.makedirs(self.diag_dir, exist_ok=True)

        # A) Optimization
        self._opt_per_epoch: List[Dict]  = []
        self._opt_per_batch: List[Dict]  = []

        # B) Calibration
        self._calibration: List[Dict]    = []

        # C) Generalization
        self._gen_per_epoch: List[Dict]  = []
        self._best_epoch_gap: Optional[Dict] = None

        # D) Stability
        self._stability_events: List[Dict] = []
        self._nan_count        = 0
        self._skipped_count    = 0
        self._clip_count       = 0

        # Summary
        self._summary: Dict[str, Any] = {}

    # ─────────────────────────────────────────────────────────
    # A) OPTIMIZATION DIAGNOSTICS
    # ─────────────────────────────────────────────────────────

    def record_gradient_stats(
        self,
        epoch: int,
        grad_variance: Optional[float] = None,
        update_norm:   Optional[float] = None,
        grad_norm:     Optional[float] = None,
        param_norm:    Optional[float] = None,
        batch:         Optional[int]   = None,
    ) -> None:
        """
        Record gradient / update statistics.

        Parameters
        ----------
        epoch          : current epoch
        grad_variance  : variance of gradient magnitudes across params
        update_norm    : L2 norm of the optimizer parameter update
        grad_norm      : L2 norm of gradients (before clipping)
        param_norm     : L2 norm of all parameters
        batch          : optional batch index (for per-batch recording)
        """
        try:
            entry: Dict[str, Any] = {"epoch": epoch}
            if batch is not None:
                entry["batch"] = batch
            if grad_variance is not None:
                entry["grad_variance"] = round(float(grad_variance), 8)
            if update_norm is not None:
                entry["update_norm"] = round(float(update_norm), 6)
            if grad_norm is not None:
                entry["grad_norm"] = round(float(grad_norm), 6)
            if param_norm is not None:
                entry["param_norm"] = round(float(param_norm), 6)

            # Grad/update ratio (how large are updates relative to gradients)
            if update_norm is not None and grad_norm is not None and grad_norm > 1e-12:
                entry["grad_update_ratio"] = round(float(update_norm / grad_norm), 6)

            if batch is not None:
                self._opt_per_batch.append(entry)
            else:
                self._opt_per_epoch.append(entry)
        except Exception as e:
            logger.debug(f"[Diagnostics] record_gradient_stats failed: {e}")

    def record_gradient_cosine(
        self,
        epoch: int,
        cosine_sim: float,
    ) -> None:
        """
        Record cosine similarity between consecutive epoch gradients.
        Measures gradient consistency (high = stable direction, low = chaotic).
        """
        try:
            for row in self._opt_per_epoch:
                if row.get("epoch") == epoch:
                    row["gradient_cosine_sim"] = round(float(cosine_sim), 6)
                    return
            self._opt_per_epoch.append({
                "epoch": epoch,
                "gradient_cosine_sim": round(float(cosine_sim), 6),
            })
        except Exception as e:
            logger.debug(f"[Diagnostics] record_gradient_cosine failed: {e}")

    def record_epoch_from_curves(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Extract optimization diagnostics automatically from curve metrics dict.
        Call with the same metrics dict passed to CurveTracker.record().

        Extracts: grad_norm, param_norm, lr, train_loss, val_loss
        """
        try:
            entry: Dict[str, Any] = {"epoch": epoch}
            for key in ["grad_norm", "param_norm", "lr", "train_loss", "val_loss",
                        "condition_number", "rank_k"]:
                if key in metrics and metrics[key] is not None:
                    try:
                        entry[key] = round(float(metrics[key]), 8)
                    except (TypeError, ValueError):
                        entry[key] = metrics[key]

            # Add to per-epoch (merge if epoch already exists)
            for row in self._opt_per_epoch:
                if row.get("epoch") == epoch:
                    row.update(entry)
                    return
            self._opt_per_epoch.append(entry)
        except Exception as e:
            logger.debug(f"[Diagnostics] record_epoch_from_curves failed: {e}")

    # ─────────────────────────────────────────────────────────
    # B) CALIBRATION / RELIABILITY
    # ─────────────────────────────────────────────────────────

    def record_calibration(
        self,
        epoch: int,
        ece: Optional[float] = None,
        confidence_hist: Optional[List[float]] = None,
        uncertainty_mean: Optional[float] = None,
        uncertainty_std: Optional[float] = None,
    ) -> None:
        """
        Record calibration metrics for one epoch or at final evaluation.

        Parameters
        ----------
        epoch            : epoch index (-1 for final evaluation)
        ece              : Expected Calibration Error (lower is better)
        confidence_hist  : list of confidence bin frequencies (20 bins)
        uncertainty_mean : mean epistemic/aleatoric uncertainty
        uncertainty_std  : std of uncertainty across samples
        """
        try:
            entry: Dict[str, Any] = {"epoch": epoch}
            if ece is not None:
                entry["ece"] = round(float(ece), 6)
            if confidence_hist is not None:
                entry["confidence_histogram"] = [round(float(v), 6) for v in confidence_hist]
            if uncertainty_mean is not None:
                entry["uncertainty_mean"] = round(float(uncertainty_mean), 6)
            if uncertainty_std is not None:
                entry["uncertainty_std"]  = round(float(uncertainty_std), 6)
            self._calibration.append(entry)
        except Exception as e:
            logger.debug(f"[Diagnostics] record_calibration failed: {e}")

    # ─────────────────────────────────────────────────────────
    # C) GENERALIZATION
    # ─────────────────────────────────────────────────────────

    def record_generalization(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_dice: Optional[float] = None,
        val_dice: Optional[float] = None,
    ) -> None:
        """
        Record generalization gap metrics for one epoch.
        Computes train-val gap and overfitting indicator automatically.
        """
        try:
            entry: Dict[str, Any] = {"epoch": epoch}
            if train_loss is not None and val_loss is not None:
                entry["train_loss"]  = round(float(train_loss), 6)
                entry["val_loss"]    = round(float(val_loss), 6)
                entry["loss_gap"]    = round(float(val_loss - train_loss), 6)
                entry["overfitting"] = entry["loss_gap"] > 0.0  # val > train = overfitting
            if train_dice is not None and val_dice is not None:
                entry["train_dice"] = round(float(train_dice), 6)
                entry["val_dice"]   = round(float(val_dice), 6)
                entry["dice_gap"]   = round(float(train_dice - val_dice), 6)  # positive = generalization gap

            # Update or append
            for row in self._gen_per_epoch:
                if row.get("epoch") == epoch:
                    row.update(entry)
                    return
            self._gen_per_epoch.append(entry)
        except Exception as e:
            logger.debug(f"[Diagnostics] record_generalization failed: {e}")

    def record_generalization_from_curves(self, curve_rows: List[Dict]) -> None:
        """
        Auto-compute generalization diagnostics from all curve data at once.
        Call at end of training with CurveTracker.get_rows().
        """
        try:
            for row in curve_rows:
                epoch = row.get("epoch", -1)
                self.record_generalization(
                    epoch=epoch,
                    train_loss=row.get("train_loss"),
                    val_loss=row.get("val_loss"),
                    train_dice=row.get("train_dice"),
                    val_dice=row.get("val_dice") or row.get("val_mean_dice"),
                )
        except Exception as e:
            logger.debug(f"[Diagnostics] record_generalization_from_curves failed: {e}")

    def compute_best_epoch_gap(
        self,
        best_epoch: int,
        best_val_metric: float,
        final_train_metric: Optional[float] = None,
    ) -> None:
        """Record the best epoch information with train-val gap at that point."""
        try:
            self._best_epoch_gap = {
                "best_epoch":         best_epoch,
                "best_val_metric":    round(float(best_val_metric), 6),
                "final_train_metric": round(float(final_train_metric), 6)
                                      if final_train_metric is not None else None,
            }
            # Find loss gap at best epoch
            for row in self._gen_per_epoch:
                if row.get("epoch") == best_epoch:
                    for key in ["loss_gap", "dice_gap", "overfitting"]:
                        if key in row:
                            self._best_epoch_gap[f"at_best_{key}"] = row[key]
        except Exception as e:
            logger.debug(f"[Diagnostics] compute_best_epoch_gap failed: {e}")

    # ─────────────────────────────────────────────────────────
    # D) STABILITY
    # ─────────────────────────────────────────────────────────

    def record_stability_event(
        self,
        event_type: str,
        epoch: int = -1,
        batch: Optional[int] = None,
        details: Optional[str] = None,
    ) -> None:
        """
        Record a stability event.

        event_type : "nan" | "skipped_batch" | "grad_clip" | "oom" | "resume" | custom
        """
        try:
            entry: Dict[str, Any] = {
                "event_type": event_type,
                "epoch":      epoch,
            }
            if batch is not None:
                entry["batch"] = batch
            if details:
                entry["details"] = str(details)[:200]  # cap length

            self._stability_events.append(entry)

            # Counters
            if event_type == "nan":
                self._nan_count += 1
            elif event_type == "skipped_batch":
                self._skipped_count += 1
            elif event_type == "grad_clip":
                self._clip_count += 1
        except Exception as e:
            logger.debug(f"[Diagnostics] record_stability_event failed: {e}")

    # ─────────────────────────────────────────────────────────
    # FINALIZE + SAVE
    # ─────────────────────────────────────────────────────────

    def finalize(self) -> None:
        """Compute final summary statistics from all collected data."""
        try:
            # Compute overfitting trend
            gen_entries = self._gen_per_epoch
            if gen_entries:
                loss_gaps = [r["loss_gap"] for r in gen_entries if "loss_gap" in r]
                dice_gaps = [r["dice_gap"] for r in gen_entries if "dice_gap" in r]
                overfitting_epochs = [r["epoch"] for r in gen_entries
                                      if r.get("overfitting") is True]

                gen_summary = {
                    "overfitting_epoch_count": len(overfitting_epochs),
                    "total_epochs_tracked":    len(gen_entries),
                    "overfitting_fraction":    round(
                        len(overfitting_epochs) / max(len(gen_entries), 1), 3
                    ),
                }
                if loss_gaps:
                    gen_summary["mean_loss_gap"]  = round(sum(loss_gaps) / len(loss_gaps), 6)
                    gen_summary["max_loss_gap"]   = round(max(loss_gaps), 6)
                    gen_summary["final_loss_gap"] = round(loss_gaps[-1], 6)
                if dice_gaps:
                    gen_summary["mean_dice_gap"]  = round(sum(dice_gaps) / len(dice_gaps), 6)
                    gen_summary["final_dice_gap"] = round(dice_gaps[-1], 6)
            else:
                gen_summary = {}

            # Optimization summary
            opt_entries = self._opt_per_epoch
            opt_summary: Dict[str, Any] = {}
            if opt_entries:
                grad_norms = [r["grad_norm"] for r in opt_entries if "grad_norm" in r]
                if grad_norms:
                    opt_summary["grad_norm_mean"]  = round(sum(grad_norms) / len(grad_norms), 6)
                    opt_summary["grad_norm_max"]   = round(max(grad_norms), 6)
                    opt_summary["grad_norm_final"] = round(grad_norms[-1], 6)

                cosines = [r["gradient_cosine_sim"] for r in opt_entries
                           if "gradient_cosine_sim" in r]
                if cosines:
                    opt_summary["grad_cosine_mean"] = round(sum(cosines) / len(cosines), 4)
                    opt_summary["grad_cosine_min"]  = round(min(cosines), 4)

            self._summary = {
                "optimization":  opt_summary,
                "generalization": gen_summary,
                "best_epoch":    self._best_epoch_gap or {},
                "stability": {
                    "nan_incidents":       self._nan_count,
                    "skipped_batches":     self._skipped_count,
                    "gradient_clip_events": self._clip_count,
                    "total_events":        len(self._stability_events),
                    "is_unstable":         self._nan_count > 0 or self._skipped_count > 5,
                },
                "calibration_epochs_recorded": len(self._calibration),
            }
        except Exception as e:
            logger.warning(f"[Diagnostics] finalize failed: {e}")

    def save(self) -> None:
        """Write all diagnostic JSON files to diagnostics/ directory."""
        saves = [
            (
                {"epochs": self._opt_per_epoch, "batches_sample": self._opt_per_batch[:200]},
                os.path.join(self.diag_dir, "optimization_diagnostics.json"),
            ),
            (
                {"epochs": self._calibration},
                os.path.join(self.diag_dir, "calibration_diagnostics.json"),
            ),
            (
                {
                    "epochs":           self._gen_per_epoch,
                    "best_epoch_gap":   self._best_epoch_gap,
                    "summary":          self._summary.get("generalization", {}),
                },
                os.path.join(self.diag_dir, "generalization_diagnostics.json"),
            ),
            (
                {
                    "events":  self._stability_events,
                    "summary": self._summary.get("stability", {}),
                },
                os.path.join(self.diag_dir, "stability_diagnostics.json"),
            ),
            (
                self._summary,
                os.path.join(self.diag_dir, "diagnostics_summary.json"),
            ),
        ]
        for data, path in saves:
            try:
                _atomic_json(data, path)
            except Exception as e:
                logger.warning(f"[Diagnostics] Could not save {os.path.basename(path)}: {e}")

        logger.info(f"[Diagnostics] Saved → {self.diag_dir}/")

    # ─────────────────────────────────────────────────────────
    # CONVENIENCE: COMPUTE GRAD STATS FROM MODEL
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_model_grad_stats(model) -> Dict[str, float]:
        """
        Compute gradient and parameter norm statistics from a model.
        Call after loss.backward() and before optimizer.step().
        Returns dict safe for passing to record_gradient_stats().

        Example:
            stats = DiagnosticsTracker.compute_model_grad_stats(model)
            diag.record_gradient_stats(epoch, **stats)
        """
        result: Dict[str, float] = {}
        try:
            import torch
            grad_norms:  List[float] = []
            param_norms: List[float] = []

            for p in model.parameters():
                if p.grad is not None:
                    grad_norms.append(p.grad.data.norm(2).item())
                if p.data is not None:
                    param_norms.append(p.data.norm(2).item())

            if grad_norms:
                total_grad_norm = math.sqrt(sum(g**2 for g in grad_norms))
                result["grad_norm"]     = total_grad_norm
                result["grad_variance"] = _variance(grad_norms)
            if param_norms:
                result["param_norm"] = math.sqrt(sum(p**2 for p in param_norms))
        except Exception as e:
            logger.debug(f"[Diagnostics] compute_model_grad_stats failed: {e}")
        return result


# ══════════════════════════════════════════════════════════════
#  PURE ECE COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_ece(
    confidences: Any,
    labels: Any,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Used for classification heads (softmax output).

    Parameters
    ----------
    confidences : numpy array or torch tensor [N] — predicted max probability
    labels      : numpy array or torch tensor [N] — true binary correctness (0/1)
    n_bins      : number of calibration bins

    Returns
    -------
    ECE (float, lower is better, 0 = perfectly calibrated)
    """
    try:
        import numpy as np
        conf = _to_numpy(confidences)
        lbls = _to_numpy(labels)
        if conf is None or lbls is None:
            return float("nan")

        bins = np.linspace(0, 1, n_bins + 1)
        ece  = 0.0
        n    = len(conf)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask   = (conf >= lo) & (conf < hi)
            if mask.sum() == 0:
                continue
            acc  = float(lbls[mask].mean())
            conf_ = float(conf[mask].mean())
            ece  += mask.sum() / n * abs(acc - conf_)
        return round(ece, 6)
    except Exception as e:
        logger.debug(f"[Diagnostics] compute_ece failed: {e}")
        return float("nan")


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _variance(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _to_numpy(x):
    try:
        import numpy as np
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.array(x)
    except Exception:
        return None
