"""
artifact_utils.py
=================
Research artifact collector + Trainer Hook API for the mini_research pipeline.

TWO MAIN CLASSES:

1. ArtifactCollector  — saves model-internal tensors to disk (unchanged API)
    collector.record_graph(epoch, A_prior, A_learned)
    collector.record_spectral(epoch, eigenvalues, condition_number, ...)
    collector.record_evidential(epoch, uncertainty_map, confidence_map)
    collector.record_attention(epoch, attention_weights)

2. HookableTracker  — trainer-facing hook API that routes everything automatically
    tracker = HookableTracker(exp_path, curve_tracker, diagnostics_tracker)
    tracker.log_tensor("A_prior", tensor)        → ArtifactCollector
    tracker.log_tensor("eigenvalues", vals)      → ArtifactCollector spectral
    tracker.log_tensor("attention_map", attn)    → ArtifactCollector attention
    tracker.log_scalar("condition_number", val)  → CurveTracker + DiagnosticsTracker
    tracker.log_scalar("rank_k", k)              → CurveTracker
    tracker.log_gradient_stats(epoch, model)     → DiagnosticsTracker
    tracker.flush(epoch)                         → saves all stats

Artifact directory layout:
    experiment_dir/artifacts/
        graph/
            graph_epoch_0000.npz
            graph_stats.json
        spectral/
            eigenvalues_epoch_0000.npy
            spectral_stats.json
        evidential/
            uncertainty_hist_epoch_0000.npz
            evidential_stats.json
        attention/
            attention_mean_epoch_0000.npy
            attention_stats.json
"""

import os
import json
import logging
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _safe(fn, default=None):
    try:
        return fn()
    except Exception as e:
        logger.debug(f"[ArtifactCollector] Skipped: {e}")
        return default


def _to_numpy(tensor) -> Optional[Any]:
    """Safely convert a torch Tensor or numpy array to numpy (CPU, float32)."""
    if tensor is None:
        return None
    try:
        if np is None:
            return None
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().float().numpy()
        if isinstance(tensor, np.ndarray):
            return tensor.astype(np.float32)
        return np.array(tensor, dtype=np.float32)
    except Exception:
        return None


def _atomic_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
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


def _save_npz(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np_arrays = {k: v for k, v in arrays.items() if v is not None}
    if np_arrays and np is not None:
        np.savez_compressed(path, **np_arrays)


# ══════════════════════════════════════════════════════════════
#  ARTIFACT COLLECTOR
# ══════════════════════════════════════════════════════════════

class ArtifactCollector:
    """
    Collects and saves research-grade tensor artifacts from training.

    All record_* methods are optional hooks — call when tensors are available.
    The collector handles numpy conversion, stat aggregation, and file rotation.
    All methods are fully try/except guarded — never crash training.

    Parameters
    ----------
    exp_path            : root experiment directory
    save_every_n_epochs : how often to save full tensor files to disk
    max_files_per_type  : retention: keep at most this many tensor files per type
    """

    def __init__(
        self,
        exp_path: str,
        save_every_n_epochs: int = 10,
        max_files_per_type:  int = 5,
    ):
        self.exp_path   = exp_path
        self.save_every = max(1, save_every_n_epochs)
        self.max_files  = max_files_per_type

        self.artifacts_dir  = os.path.join(exp_path, "artifacts")
        self.graph_dir      = os.path.join(self.artifacts_dir, "graph")
        self.spectral_dir   = os.path.join(self.artifacts_dir, "spectral")
        self.evidential_dir = os.path.join(self.artifacts_dir, "evidential")
        self.attention_dir  = os.path.join(self.artifacts_dir, "attention")

        for d in [self.graph_dir, self.spectral_dir,
                  self.evidential_dir, self.attention_dir]:
            os.makedirs(d, exist_ok=True)

        self._graph_stats:      List[Dict] = []
        self._spectral_stats:   List[Dict] = []
        self._evidential_stats: List[Dict] = []
        self._attention_stats:  List[Dict] = []

    # ─────────────────────────────────────────────────────────
    # A) GRAPH ARTIFACTS
    # ─────────────────────────────────────────────────────────

    def record_graph(
        self,
        epoch: int,
        A_prior:   Optional[Any] = None,
        A_learned: Optional[Any] = None,
    ) -> None:
        """Record graph matrix snapshots + sparsity / degree stats."""
        stats: Dict[str, Any] = {"epoch": epoch}
        for tag, A_tensor in [("prior", A_prior), ("learned", A_learned)]:
            A_np = _safe(lambda: _to_numpy(A_tensor))
            if A_np is not None and A_np.ndim == 2:
                N   = A_np.shape[0]
                nnz = int(np.count_nonzero(A_np))
                degs = A_np.sum(axis=1)
                stats[f"{tag}_sparsity_ratio"] = round(1.0 - nnz / max(N * N, 1), 6)
                stats[f"{tag}_mean_degree"]    = round(float(degs.mean()), 4)
                stats[f"{tag}_max_degree"]     = round(float(degs.max()), 4)
                stats[f"{tag}_std_degree"]     = round(float(degs.std()), 4)
                stats[f"{tag}_nnz"]            = nnz
                stats[f"{tag}_shape"]          = list(A_np.shape)
                # Graph entropy (degree distribution entropy)
                try:
                    p = degs / (degs.sum() + 1e-12)
                    p = p[p > 0]
                    stats[f"{tag}_entropy"] = round(
                        float(-(p * np.log(p + 1e-12)).sum()), 6
                    )
                except Exception:
                    pass
        self._graph_stats.append(stats)
        if epoch % self.save_every == 0:
            _safe(lambda: self._save_graph_tensors(epoch, A_prior, A_learned))

    def _save_graph_tensors(self, epoch: int, A_prior, A_learned) -> None:
        prior_np   = _to_numpy(A_prior)
        learned_np = _to_numpy(A_learned)
        if prior_np is not None or learned_np is not None:
            path = os.path.join(self.graph_dir, f"graph_epoch_{epoch:04d}.npz")
            _save_npz(path, A_prior=prior_np, A_learned=learned_np)
            self._enforce_file_limit(self.graph_dir, "graph_epoch_*.npz")
            logger.debug(f"[Artifacts] Graph saved epoch {epoch}")

    # ─────────────────────────────────────────────────────────
    # B) SPECTRAL ARTIFACTS
    # ─────────────────────────────────────────────────────────

    def record_spectral(
        self,
        epoch: int,
        eigenvalues:      Optional[Any]   = None,
        condition_number: Optional[float] = None,
        rank_used:        Optional[int]   = None,
        solver_time_ms:   Optional[float] = None,
    ) -> None:
        """Record spectral decomposition artifacts."""
        stats: Dict[str, Any] = {"epoch": epoch}
        eig_np = _safe(lambda: _to_numpy(eigenvalues))
        if eig_np is not None:
            eig_list = eig_np.tolist()
            stats["eigenvalues"]       = [round(float(v), 6) for v in eig_list]
            stats["top_eigenvalue"]    = round(float(eig_np.max()), 6)
            stats["bottom_eigenvalue"] = round(float(eig_np.min()), 6)
            stats["eigenvalue_range"]  = round(float(eig_np.max() - eig_np.min()), 6)
            stats["eigenvalue_gap"]    = round(
                float(eig_np[-1] - eig_np[-2]), 6
            ) if len(eig_np) > 1 else None
            # Spectral entropy
            try:
                p = np.abs(eig_np) / (np.abs(eig_np).sum() + 1e-12)
                stats["spectral_entropy"] = round(float(-(p * np.log(p + 1e-12)).sum()), 6)
            except Exception:
                pass

        if condition_number is not None:
            stats["condition_number"] = round(float(condition_number), 4)
        if rank_used is not None:
            stats["rank_used"] = int(rank_used)
        if solver_time_ms is not None:
            stats["solver_time_ms"] = round(float(solver_time_ms), 3)

        self._spectral_stats.append(stats)
        if epoch % self.save_every == 0 and eig_np is not None:
            _safe(lambda: self._save_spectral_tensors(epoch, eig_np))

    def _save_spectral_tensors(self, epoch: int, eig_np) -> None:
        path = os.path.join(self.spectral_dir, f"eigenvalues_epoch_{epoch:04d}.npy")
        np.save(path, eig_np)
        self._enforce_file_limit(self.spectral_dir, "eigenvalues_epoch_*.npy")
        logger.debug(f"[Artifacts] Eigenvalues saved epoch {epoch}")

    # ─────────────────────────────────────────────────────────
    # C) EVIDENTIAL ARTIFACTS
    # ─────────────────────────────────────────────────────────

    def record_evidential(
        self,
        epoch: int,
        uncertainty_map: Optional[Any] = None,
        confidence_map:  Optional[Any] = None,
    ) -> None:
        """Record evidential uncertainty statistics."""
        stats: Dict[str, Any] = {"epoch": epoch}
        unc_np = _safe(lambda: _to_numpy(uncertainty_map))
        if unc_np is not None:
            unc_flat = unc_np.flatten()
            stats["uncertainty_mean"]     = round(float(unc_flat.mean()), 6)
            stats["uncertainty_std"]      = round(float(unc_flat.std()),  6)
            stats["uncertainty_max"]      = round(float(unc_flat.max()),  6)
            stats["high_uncertainty_pct"] = round(
                float((unc_flat > unc_flat.mean() + unc_flat.std()).mean() * 100), 3
            )
        conf_np = _safe(lambda: _to_numpy(confidence_map))
        if conf_np is not None:
            conf_flat = conf_np.flatten()
            stats["confidence_mean"] = round(float(conf_flat.mean()), 6)
            stats["confidence_std"]  = round(float(conf_flat.std()),  6)
        self._evidential_stats.append(stats)
        if epoch % self.save_every == 0 and unc_np is not None:
            _safe(lambda: self._save_evidential_tensors(epoch, unc_np, conf_np))

    def _save_evidential_tensors(self, epoch: int, unc_np, conf_np) -> None:
        hist_vals, hist_edges = np.histogram(unc_np.flatten(), bins=50, density=True)
        path = os.path.join(self.evidential_dir, f"uncertainty_hist_epoch_{epoch:04d}.npz")
        _save_npz(path, hist_values=hist_vals, hist_edges=hist_edges)
        if conf_np is not None:
            hc, ec = np.histogram(conf_np.flatten(), bins=50, density=True)
            pc = os.path.join(self.evidential_dir, f"confidence_hist_epoch_{epoch:04d}.npz")
            _save_npz(pc, hist_values=hc, hist_edges=ec)
        self._enforce_file_limit(self.evidential_dir, "uncertainty_hist_*.npz")
        logger.debug(f"[Artifacts] Evidential histograms saved epoch {epoch}")

    # ─────────────────────────────────────────────────────────
    # D) ATTENTION ARTIFACTS
    # ─────────────────────────────────────────────────────────

    def record_attention(
        self,
        epoch: int,
        attention_weights: Optional[Any] = None,
    ) -> None:
        """Record attention weight statistics."""
        stats: Dict[str, Any] = {"epoch": epoch}
        attn_np = _safe(lambda: _to_numpy(attention_weights))
        if attn_np is not None and attn_np.ndim >= 3:
            eps   = 1e-8
            p     = np.clip(attn_np, eps, 1.0)
            entropy = -(p * np.log(p)).sum(axis=-1).mean()
            stats["mean_entropy"]   = round(float(entropy), 6)
            stats["mean_magnitude"] = round(float(attn_np.mean()), 6)
            stats["max_magnitude"]  = round(float(attn_np.max()),  6)
            if attn_np.ndim == 4:
                head_means     = attn_np.mean(axis=(0, 2, 3))
                stats["head_std"]  = round(float(head_means.std()), 6)
                stats["num_heads"] = attn_np.shape[1]
        self._attention_stats.append(stats)
        if epoch % self.save_every == 0 and attn_np is not None:
            _safe(lambda: self._save_attention_tensors(epoch, attn_np))

    def _save_attention_tensors(self, epoch: int, attn_np) -> None:
        mean_attn = attn_np.mean(axis=0) if attn_np.ndim == 4 else attn_np
        path = os.path.join(self.attention_dir, f"attention_mean_epoch_{epoch:04d}.npy")
        np.save(path, mean_attn.astype(np.float16))
        self._enforce_file_limit(self.attention_dir, "attention_mean_*.npy")
        logger.debug(f"[Artifacts] Attention saved epoch {epoch}")

    # ─────────────────────────────────────────────────────────
    # STAT ACCESS (for plotting)
    # ─────────────────────────────────────────────────────────

    def get_graph_stats(self)      -> List[Dict]: return list(self._graph_stats)
    def get_spectral_stats(self)   -> List[Dict]: return list(self._spectral_stats)
    def get_evidential_stats(self) -> List[Dict]: return list(self._evidential_stats)
    def get_attention_stats(self)  -> List[Dict]: return list(self._attention_stats)

    def get_eigenvalue_timeline(self) -> List[Dict]:
        return [
            {"epoch": s["epoch"], "eigenvalues": s.get("eigenvalues", [])}
            for s in self._spectral_stats if "eigenvalues" in s
        ]

    def get_condition_numbers(self) -> List[float]:
        return [s["condition_number"] for s in self._spectral_stats
                if "condition_number" in s]

    # ─────────────────────────────────────────────────────────
    # SAVE STATS
    # ─────────────────────────────────────────────────────────

    def save_stats(self) -> None:
        """Write all collected stats to JSON."""
        saves = [
            (self._graph_stats,      os.path.join(self.graph_dir,      "graph_stats.json")),
            (self._spectral_stats,   os.path.join(self.spectral_dir,   "spectral_stats.json")),
            (self._evidential_stats, os.path.join(self.evidential_dir, "evidential_stats.json")),
            (self._attention_stats,  os.path.join(self.attention_dir,  "attention_stats.json")),
        ]
        for data, path in saves:
            if data:
                try:
                    _atomic_json({"epochs": data}, path)
                except Exception as e:
                    logger.warning(f"[Artifacts] Could not save {os.path.basename(path)}: {e}")

    # ─────────────────────────────────────────────────────────
    # STORAGE RETENTION
    # ─────────────────────────────────────────────────────────

    def _enforce_file_limit(self, directory: str, glob_pattern: str) -> None:
        import glob as _glob
        files = sorted(_glob.glob(os.path.join(directory, glob_pattern)))
        while len(files) > self.max_files:
            oldest = files.pop(0)
            try:
                os.remove(oldest)
            except OSError:
                pass


# ══════════════════════════════════════════════════════════════
#  HOOKABLE TRACKER  — trainer-facing hook API
# ══════════════════════════════════════════════════════════════

# Tensor routing table — maps logical name → (artifact_type, kwarg_name)
_TENSOR_ROUTES: Dict[str, tuple] = {
    # Graph artifacts
    "A_prior":        ("graph",     "A_prior"),
    "a_prior":        ("graph",     "A_prior"),
    "A_learned":      ("graph",     "A_learned"),
    "a_learned":      ("graph",     "A_learned"),
    "adjacency":      ("graph",     "A_learned"),
    # Spectral artifacts
    "eigenvalues":    ("spectral",  "eigenvalues"),
    "eigvals":        ("spectral",  "eigenvalues"),
    # Evidential artifacts
    "uncertainty":    ("evidential","uncertainty_map"),
    "uncertainty_map":("evidential","uncertainty_map"),
    "confidence":     ("evidential","confidence_map"),
    "confidence_map": ("evidential","confidence_map"),
    # Attention artifacts
    "attention":      ("attention", "attention_weights"),
    "attention_map":  ("attention", "attention_weights"),
    "attention_weights":("attention","attention_weights"),
}

# Scalar routing — maps logical name to CurveTracker metric key
_SCALAR_ROUTES: Dict[str, str] = {
    "condition_number": "condition_number",
    "rank_k":           "rank_k",
    "solver_time_ms":   "solver_time_ms",
    "lr":               "lr",
    "grad_norm":        "grad_norm",
    "param_norm":       "param_norm",
    "train_loss":       "train_loss",
    "val_loss":         "val_loss",
    "val_dice":         "val_dice",
    "train_dice":       "train_dice",
}


class HookableTracker:
    """
    Trainer Hook API — the single interface between the trainer and
    all tracking systems (curves, artifacts, diagnostics).

    The trainer calls these methods; HookableTracker routes everything:
        - Tensors    → ArtifactCollector (saves .npy / .npz)
        - Scalars    → CurveTracker      (saves training_curves.csv/json)
        - Scalars    → DiagnosticsTracker (saves diagnostics/)
        - Grad stats → DiagnosticsTracker

    Usage in trainer:
        # Initialize (done in run_experiment.py):
        tracker = HookableTracker(exp_path, curve_tracker, diag_tracker)

        # In trainer per epoch / per batch:
        tracker.log_tensor("A_prior", A_prior_tensor)
        tracker.log_tensor("A_learned", A_learned_tensor)
        tracker.log_tensor("eigenvalues", eigenvalues_tensor)
        tracker.log_tensor("attention_map", attn_tensor)
        tracker.log_scalar("condition_number", cond_num)
        tracker.log_scalar("rank_k", rank_k)
        tracker.log_gradient_stats(epoch, model)   # optional

        # At end of each epoch:
        tracker.flush(epoch)   # saves all stats to disk

    Parameters
    ----------
    exp_path            : experiment root directory
    curve_tracker       : CurveTracker instance (or None)
    diagnostics_tracker : DiagnosticsTracker instance (or None)
    save_every_n_epochs : artifact save frequency
    """

    def __init__(
        self,
        exp_path: str,
        curve_tracker=None,
        diagnostics_tracker=None,
        save_every_n_epochs: int = 10,
    ):
        self.exp_path    = exp_path
        self._curves     = curve_tracker
        self._diag       = diagnostics_tracker
        self._collector  = ArtifactCollector(
            exp_path, save_every_n_epochs=save_every_n_epochs
        )

        # Pending scalars / tensors for current epoch
        self._pending_scalars:  Dict[str, Any] = {}
        self._pending_tensors:  Dict[str, Any] = {}
        self._current_epoch:    int = -1

        # Accumulated spectral scalars (condition_number, rank_used, solver_time_ms)
        self._spectral_scalars: Dict[str, Any] = {}

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch to update epoch context."""
        # Auto-flush previous epoch if not done
        if self._current_epoch >= 0 and self._current_epoch != epoch:
            self.flush(self._current_epoch)
        self._current_epoch = epoch
        self._pending_scalars = {}
        self._pending_tensors = {}
        self._spectral_scalars = {}

    def log_tensor(
        self,
        name: str,
        tensor: Any,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Log a tensor artifact.

        Supported names (case-insensitive):
            A_prior, A_learned, adjacency
            eigenvalues, eigvals
            uncertainty, uncertainty_map
            confidence, confidence_map
            attention, attention_map, attention_weights

        Unknown names are stored in a generic custom bucket for future use.

        Parameters
        ----------
        name   : logical tensor name (see supported names above)
        tensor : torch.Tensor, numpy array, or list
        epoch  : override epoch (defaults to current epoch)
        """
        if epoch is None:
            epoch = self._current_epoch
        ep = max(epoch, 0)

        _safe(lambda: self._route_tensor(name.lower(), tensor, ep))

    def log_scalar(
        self,
        name: str,
        value: Any,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Log a scalar metric.

        Routes to CurveTracker for time-series storage and
        DiagnosticsTracker for diagnostic analysis.

        Parameters
        ----------
        name  : metric name (e.g. 'condition_number', 'rank_k', 'lr')
        value : numeric value
        epoch : override epoch
        """
        if epoch is None:
            epoch = self._current_epoch
        try:
            val = float(value)
        except (TypeError, ValueError):
            return

        # Store pending (will be committed in flush())
        metric_key = _SCALAR_ROUTES.get(name.lower(), name.lower())
        self._pending_scalars[metric_key] = val

        # Route spectral scalars for ArtifactCollector
        if name.lower() == "condition_number":
            self._spectral_scalars["condition_number"] = val
        elif name.lower() == "rank_k":
            self._spectral_scalars["rank_used"] = int(val)
        elif name.lower() == "solver_time_ms":
            self._spectral_scalars["solver_time_ms"] = val

    def log_metrics(self, metrics: Dict[str, Any], epoch: Optional[int] = None) -> None:
        """Bulk log multiple scalars at once."""
        for name, value in metrics.items():
            self.log_scalar(name, value, epoch=epoch)

    def log_gradient_stats(self, epoch: int, model) -> None:
        """
        Compute gradient and parameter norm statistics from a model
        and route to DiagnosticsTracker.

        Call AFTER loss.backward() and BEFORE optimizer.step().
        """
        if self._diag is None:
            return
        try:
            from src.utils.diagnostics_utils import DiagnosticsTracker
            stats = DiagnosticsTracker.compute_model_grad_stats(model)
            if stats:
                _safe(lambda: self._diag.record_gradient_stats(epoch, **stats))
                # Also add to pending scalars for CurveTracker
                if "grad_norm" in stats:
                    self._pending_scalars["grad_norm"] = stats["grad_norm"]
                if "param_norm" in stats:
                    self._pending_scalars["param_norm"] = stats["param_norm"]
        except Exception as e:
            logger.debug(f"[HookableTracker] log_gradient_stats failed: {e}")

    def flush(self, epoch: int) -> None:
        """
        Commit all pending scalars + spectral stats for the epoch.
        Call at the END of each epoch.

        Writes:
         - pending scalars to CurveTracker (but doesn't save CSV yet)
         - spectral scalars to ArtifactCollector.record_spectral()
         - epoch metrics to DiagnosticsTracker
        """
        # ── Route pending tensors that weren't flushed via log_tensor ──
        for name, tensor in self._pending_tensors.items():
            _safe(lambda n=name, t=tensor: self._route_tensor(n, t, epoch))

        # ── Commit spectral scalars to ArtifactCollector ──
        if self._spectral_scalars or any(k == "spectral" for k in self._pending_tensors):
            eigenvalues = self._pending_tensors.pop("eigenvalues", None)
            _safe(lambda: self._collector.record_spectral(
                epoch=epoch,
                eigenvalues=eigenvalues,
                **self._spectral_scalars,
            ))

        # ── Route pending scalars to DiagnosticsTracker ──
        if self._diag is not None and self._pending_scalars:
            _safe(lambda: self._diag.record_epoch_from_curves(epoch, self._pending_scalars))

        # ── Save artifact stats ──
        _safe(lambda: self._collector.save_stats())

        # Reset
        self._pending_scalars = {}
        self._pending_tensors = {}
        self._spectral_scalars = {}

    def finalize(self) -> None:
        """Call at end of training. Saves all artifact stats."""
        if self._current_epoch >= 0:
            _safe(lambda: self.flush(self._current_epoch))
        _safe(lambda: self._collector.save_stats())

    def get_collector(self) -> ArtifactCollector:
        """Access the underlying ArtifactCollector."""
        return self._collector

    def _route_tensor(self, name: str, tensor: Any, epoch: int) -> None:
        """Internal: route a named tensor to the correct artifact bucket."""
        route = _TENSOR_ROUTES.get(name)
        if route is None:
            # Unknown tensor: store in pending for potential custom handling
            self._pending_tensors[name] = tensor
            logger.debug(f"[HookableTracker] Unknown tensor '{name}' — buffered")
            return

        artifact_type, kwarg = route

        if artifact_type == "graph":
            if kwarg == "A_prior":
                self._collector.record_graph(epoch, A_prior=tensor)
            else:
                self._collector.record_graph(epoch, A_learned=tensor)

        elif artifact_type == "spectral":
            # Eigenvalues accumulate in pending; spectral_scalars merged in flush()
            self._pending_tensors["eigenvalues"] = tensor

        elif artifact_type == "evidential":
            if kwarg == "uncertainty_map":
                self._collector.record_evidential(epoch, uncertainty_map=tensor)
            else:
                self._collector.record_evidential(epoch, confidence_map=tensor)

        elif artifact_type == "attention":
            self._collector.record_attention(epoch, attention_weights=tensor)


# ══════════════════════════════════════════════════════════════
#  STANDALONE GRAPH / SPECTRAL STAT HELPERS (backward compat)
# ══════════════════════════════════════════════════════════════

def compute_graph_stats(A: Any) -> Dict[str, Any]:
    """Compute graph statistics from an adjacency matrix [N, N]."""
    A_np = _to_numpy(A)
    if A_np is None or A_np.ndim != 2:
        return {}
    N   = A_np.shape[0]
    nnz = int(np.count_nonzero(A_np))
    degrees = A_np.sum(axis=1)
    return {
        "n_nodes":        N,
        "nnz":            nnz,
        "sparsity_ratio": round(1.0 - nnz / max(N * N, 1), 6),
        "mean_degree":    round(float(degrees.mean()), 4),
        "max_degree":     round(float(degrees.max()),  4),
        "min_degree":     round(float(degrees.min()),  4),
        "std_degree":     round(float(degrees.std()),  4),
        "is_symmetric":   bool(np.allclose(A_np, A_np.T, atol=1e-4)),
    }


def compute_spectral_stats(A: Any, k: int = 10) -> Dict[str, Any]:
    """Compute spectral statistics from an adjacency matrix."""
    try:
        A_np = _to_numpy(A)
        if A_np is None:
            return {}
        N = A_np.shape[0]
        k = min(k, N - 1)
        if N > 1000:
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csr_matrix
            A_sparse = csr_matrix(A_np)
            eigs = eigsh(A_sparse, k=k, which="LM", return_eigenvectors=False)
            eigs = np.sort(np.abs(eigs))[::-1]
        else:
            eigs = np.linalg.eigvalsh(A_np)
            eigs = np.sort(np.abs(eigs))[::-1][:k]
        cond_num = float(eigs[0] / (eigs[-1] + 1e-12)) if eigs[-1] > 1e-12 else float("inf")
        return {
            "eigenvalues":      [round(float(v), 6) for v in eigs.tolist()],
            "condition_number": round(cond_num, 4),
            "spectral_gap":     round(float(eigs[0] - eigs[1]), 6) if len(eigs) > 1 else None,
            "top_eigenvalue":   round(float(eigs[0]), 6),
        }
    except Exception as e:
        logger.debug(f"[ArtifactUtils] Spectral stats failed: {e}")
        return {}
