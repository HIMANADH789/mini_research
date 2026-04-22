"""
run_experiment.py
=================
Single-experiment runner with robust --resume support.

This script is the canonical way to run ONE experiment, and it is also
called programmatically by run_search.py for each search trial.

Usage:
    # Normal run
    python scripts/run_experiment.py --config configs/exp_swinunetr.yaml

    # Auto-resume (finds latest resume checkpoint automatically)
    python scripts/run_experiment.py --config configs/exp_swinunetr.yaml --resume auto

    # Resume from specific checkpoint
    python scripts/run_experiment.py --config configs/exp_swinunetr.yaml \\
        --resume experiments/exp_012_swinunetr/checkpoints/checkpoint_resume_latest.pt

    # Resume from specific experiment folder (will search for checkpoint inside)
    python scripts/run_experiment.py --config configs/exp_swinunetr.yaml \\
        --resume-from-dir experiments/exp_012_swinunetr

What this does:
    1. Load config
    2. If --resume: find and validate checkpoint
    3. Create experiment folder (or reuse existing on resume)
    4. Run trainer (with resume if applicable)
    5. Evaluate best checkpoint
    6. Return results dict (used by run_search.py)

Returns (when called as function):
    dict with keys: exp_dir, best_dice, wt_dice, tc_dice, et_dice, hd95, ...
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import shutil
import logging
import glob
import json
import time
from typing import Optional, Dict, Any

from src.utils.experiment_utils.config import load_config
from src.utils.experiment_utils.seed import set_seed
from src.utils.experiment_utils.experiment import create_experiment
from src.utils.experiment_utils.logger import get_logger
from src.utils.experiment_utils.io import save_environment
from src.utils.experiment_utils.output import save_evaluation_results, save_evaluation_summary
from src.utils.checkpoint_utils import CheckpointManager
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.tracking_utils import CurveTracker, FailureLogger, TimingTracker
from src.utils.plotting_utils import PlotManager
from src.utils.benchmark_utils import BenchmarkExporter
from src.utils.repro_utils import ReproducibilityBuilder
from src.utils.diagnostics_utils import DiagnosticsTracker
from src.utils.artifact_utils import HookableTracker

from src.boilerplates.resolver import get_trainer_class, get_evaluator_class


# ══════════════════════════════════════════════════════
#  RESUME HELPERS
# ══════════════════════════════════════════════════════

def _resolve_resume_path(resume_arg: str, exp_dir: Optional[str]) -> Optional[str]:
    """
    Resolve the --resume argument to an actual checkpoint path.

    resume_arg:
      "auto"            → search for checkpoint in exp_dir
      "path/to/ckpt.pt" → use directly
      "none" / ""       → no resume

    exp_dir: if provided, used for "auto" resolution
    """
    if not resume_arg or resume_arg.lower() in {"none", "false", ""}:
        return None

    if resume_arg.lower() == "auto":
        if exp_dir is None:
            return None  # no existing experiment to resume from
        found = CheckpointManager.find_resume_checkpoint(exp_dir)
        if found:
            logging.getLogger(__name__).info(
                f"[Resume] Auto-detected checkpoint: {found}"
            )
        else:
            logging.getLogger(__name__).info(
                "[Resume] No resume checkpoint found — starting fresh."
            )
        return found

    # Explicit path
    if os.path.isfile(resume_arg):
        return resume_arg

    raise FileNotFoundError(f"[Resume] Checkpoint not found: {resume_arg}")


def _find_existing_exp_dir(config_path: str) -> Optional[str]:
    """
    Look for an existing experiment folder that matches the config name.
    Used for auto-resume: finds the most recent matching folder.

    Pattern: experiments/exp_NNN_<name>/config.yaml must exist.
    """
    if not os.path.isdir("experiments"):
        return None

    config = load_config(config_path)
    model_name = getattr(config, "name", None)
    if not model_name:
        return None

    pattern = os.path.join("experiments", f"exp_*_{model_name}")
    matches = sorted(glob.glob(pattern))

    # Filter to folders that have a checkpoints dir
    valid = [
        m for m in matches
        if os.path.isdir(os.path.join(m, "checkpoints"))
    ]

    return valid[-1] if valid else None


# ══════════════════════════════════════════════════════
#  MAIN RUNNER FUNCTION
# ══════════════════════════════════════════════════════

def run_experiment(
    config_path: str,
    resume: str = "none",
    resume_from_dir: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    exp_name_suffix: str = "",
) -> Dict[str, Any]:
    """
    Run a single experiment: train + evaluate.

    Parameters
    ----------
    config_path      : path to experiment YAML config
    resume           : "none" | "auto" | path/to/checkpoint.pt
    resume_from_dir  : explicit experiment dir to resume from (overrides auto-detection)
    config_overrides : dict of {dot_path: value} overrides (used by run_search.py)
    exp_name_suffix  : optional suffix appended to exp folder name (e.g. trial_003)

    Returns
    -------
    dict with keys: exp_dir, status, metrics (sub-dict with all evaluated metrics)
    """
    setup_logging()
    log = logging.getLogger("run_experiment")

    start_time = time.time()

    # ── Load config ────────────────────────────────────────────────
    config = load_config(config_path)
    if config_overrides:
        _apply_config_overrides(config, config_overrides)

    # ── Seed ───────────────────────────────────────────────────────
    set_seed(config.seed)

    # ── Resolve resume ─────────────────────────────────────────────
    existing_exp_dir = resume_from_dir or _find_existing_exp_dir(config_path)
    resume_path = _resolve_resume_path(resume, existing_exp_dir)

    # ── Create or reuse experiment folder ──────────────────────────
    if resume_path and existing_exp_dir and os.path.isdir(existing_exp_dir):
        exp_path = existing_exp_dir
        log.info(f"[Experiment] Resuming in existing folder: {exp_path}")
    else:
        exp_path = create_experiment(config, suffix=exp_name_suffix)
        log.info(f"[Experiment] Created new folder: {exp_path}")

    # ── Ensure all subdirs exist ───────────────────────────────────
    for subdir in ["plots", "tables", "artifacts", "paper_assets", "runs", "logs"]:
        os.makedirs(os.path.join(exp_path, subdir), exist_ok=True)

    # ── Save config + script ───────────────────────────────────────
    shutil.copy(config_path, os.path.join(exp_path, "config.yaml"))
    shutil.copy(__file__, os.path.join(exp_path, "run_experiment_script.py"))
    save_environment(exp_path)

    if config_overrides:
        override_path = os.path.join(exp_path, "config_overrides.json")
        with open(override_path, "w") as f:
            json.dump(config_overrides, f, indent=2, default=str)

    # ── Logger ─────────────────────────────────────────────────────
    logger = get_logger(os.path.join(exp_path, "logs", "train.log"))
    logger.info("=" * 60)
    logger.info(" Starting Experiment")
    logger.info(f" Config:    {config_path}")
    logger.info(f" Exp Path:  {exp_path}")
    if resume_path:
        logger.info(f" Resume:    {resume_path}")
    if config_overrides:
        logger.info(f" Overrides: {config_overrides}")
    logger.info("=" * 60)

    # ── Tracking stack ─────────────────────────────────────────────
    # 1. ExperimentTracker  — system/model/runtime metadata
    tracker = ExperimentTracker(
        exp_path=exp_path,
        config=config,
        run_id=os.path.basename(exp_path),
    )
    tracker.start()

    # 2. CurveTracker — per-epoch training curves (CSV + JSON)
    curve_tracker = CurveTracker(exp_path)

    # 3. FailureLogger — interruption + resume logging
    failure_logger = FailureLogger(exp_path)

    # 4. TimingTracker — epoch/val/inference timing
    timing_tracker = TimingTracker(exp_path)
    timing_tracker.training_start()

    # 5. PlotManager — auto-generate plots after training
    plotter = PlotManager(exp_path)

    # 6. BenchmarkExporter — tables + paper assets
    benchmark = BenchmarkExporter(exp_path, config)

    # 7. ReproducibilityBuilder — reproducibility package + paper templates
    repro = ReproducibilityBuilder(exp_path, config)
    try:
        repro.build()
    except Exception as e:
        logger.warning(f"[Experiment] Reproducibility package build failed (non-fatal): {e}")

    # 8. DiagnosticsTracker — scientific diagnostics (optimization, calibration, stability)
    diag_tracker = DiagnosticsTracker(exp_path)

    # 9. HookableTracker — trainer hook API for custom tensors (artifacts)
    hookable = HookableTracker(
        exp_path=exp_path,
        curve_tracker=curve_tracker,
        diagnostics_tracker=diag_tracker,
    )

    # ── Build trainer ──────────────────────────────────────────────
    Trainer = get_trainer_class(config)
    trainer = Trainer(config, exp_path, logger)

    if hasattr(trainer, "model"):
        tracker.set_model_info(trainer.model)

    # Expose hookable tracker to trainer so model hooks can push tensors
    try:
        trainer.hookable_tracker = hookable
    except Exception:
        pass  # read-only trainer — hook silently unavailable

    # ── Resume ─────────────────────────────────────────────────────
    if resume_path:
        trainer.resume_from(resume_path)
        start_epoch = getattr(trainer, "_resume_start_epoch", 0)
        failure_logger.record_resume(
            from_checkpoint=resume_path,
            at_epoch=start_epoch,
        )
        diag_tracker.record_stability_event("resume", epoch=start_epoch)

    # ── Attach tracking hooks to trainer ───────────────────────────
    # These hooks are called by the trainer IF it supports them.
    # v4 trainer checks hasattr(self, '_on_epoch_end') and calls it.
    # For other trainers, hooks are silently skipped.
    _attach_tracking_hooks(trainer, curve_tracker, timing_tracker, failure_logger, diag_tracker)

    # ── TRAIN ──────────────────────────────────────────────────────
    train_ok = False
    try:
        trainer.train()
        train_ok = True
    except KeyboardInterrupt:
        last_epoch = getattr(trainer, "_current_epoch", -1)
        logger.warning(f"[Experiment] Training interrupted at epoch {last_epoch}.")
        failure_logger.record_failure(
            reason="KeyboardInterrupt",
            last_epoch=last_epoch,
        )
        # Save partial curves even on interrupt
        curve_tracker.save()
        timing_tracker.training_end()
        timing_tracker.save()
        raise  # re-raise so the caller knows it was interrupted
    except Exception as exc:
        import traceback
        last_epoch = getattr(trainer, "_current_epoch", -1)
        tb_str = traceback.format_exc()
        logger.error(f"[Experiment] Training failed at epoch {last_epoch}: {exc}")
        failure_logger.record_failure(
            reason=str(exc),
            last_epoch=last_epoch,
            traceback_str=tb_str,
        )
        curve_tracker.save()
        timing_tracker.training_end()
        timing_tracker.save()
        raise

    logger.info("Training Completed")
    timing_tracker.training_end()

    # Mark last epoch as successful
    total_epochs = getattr(config.training, "epochs", 0)
    failure_logger.mark_completed(total_epochs - 1)

    # Save curve data + timing
    curve_tracker.save()
    timing_tracker.save()

    # Sync best metric from trainer
    _sync_tracker_from_trainer(tracker, trainer)

    # ── Finalize diagnostics ───────────────────────────────────────
    logger.info("[Experiment] Finalizing diagnostics...")
    try:
        diag_tracker.finalize()
        diag_tracker.save()
    except Exception as e:
        logger.warning(f"[Experiment] Diagnostics finalize failed (non-fatal): {e}")

    # ── Flush hookable tracker (artifact summaries) ────────────────
    try:
        hookable.flush(epoch=total_epochs - 1)
    except Exception as e:
        logger.warning(f"[Experiment] Artifact flush failed (non-fatal): {e}")

    # ── Generate plots ─────────────────────────────────────────────
    logger.info("[Experiment] Generating plots...")
    try:
        plotter.plot_all(curve_tracker.json_path)
        plotter.plot_paper_figures(curve_tracker.json_path)

        # If trainer exposed SABiT artifacts, generate those plots too
        _generate_artifact_plots(trainer, hookable, plotter, logger)
    except Exception as e:
        logger.warning(f"[Experiment] Plot generation failed (non-fatal): {e}")

    # ── Find best checkpoint for evaluation ────────────────────────
    ckpt = _get_eval_checkpoint(exp_path)
    if ckpt is None:
        logger.error("[Experiment] No checkpoint found for evaluation!")
        tracker.set_checkpoint_paths(best="not_found")
        tracker.finish(final_metrics={})
        tracker.save()
        tracker.print_summary()
        return {
            "exp_dir": exp_path,
            "status":  "no_checkpoint",
            "metrics": {},
            "elapsed_hours": round((time.time() - start_time) / 3600, 2),
        }

    logger.info(f"[Experiment] Evaluating checkpoint: {ckpt}")
    tracker.set_checkpoint_paths(best=ckpt)

    # ── EVALUATE ───────────────────────────────────────────────────
    Evaluator = get_evaluator_class(config)
    evaluator = Evaluator(config, ckpt)

    # Time the evaluation pass
    eval_start = time.monotonic()
    results = evaluator.evaluate()
    eval_duration = time.monotonic() - eval_start
    timing_tracker.record_validation(epoch=-1, duration_s=eval_duration)  # -1 = final eval
    timing_tracker.save()

    print("\n===== Evaluation Results =====")
    for k, v in results.items():
        print(f"  {k}: {v}")

    save_evaluation_results(results, exp_path)
    save_evaluation_summary(results, exp_path)

    # ── Finalize tracker ───────────────────────────────────────────
    tracker.finish(final_metrics=results)
    saved_paths = tracker.save()
    tracker.print_summary()

    # ── Benchmark tables + paper assets ────────────────────────────
    logger.info("[Experiment] Exporting benchmark tables and paper assets...")
    try:
        # Load timing JSON for the benchmark exporter
        timing_data = _load_json_safe(os.path.join(exp_path, "runs", "timing.json"))
        timing_data["gpu_memory_max_used_gb"] = (
            tracker._gpu_mem.get("gpu_memory_max_used_gb", "N/A")
        )

        # Include gpu_name for runtime table
        timing_data["gpu_name"] = tracker._system_info.get("gpu_name", "N/A")

        benchmark.export_run_results(
            metrics=results,
            runtime_info=timing_data,
            model_info=tracker._model_info,
        )
    except Exception as e:
        logger.warning(f"[Experiment] Benchmark export failed (non-fatal): {e}")

    # ── Finalize reproducibility package ───────────────────────────
    try:
        repro.mark_complete(system_info=tracker.system_info)
    except Exception as e:
        logger.warning(f"[Experiment] Reproducibility finalize failed (non-fatal): {e}")

    elapsed = round((time.time() - start_time) / 3600, 2)
    logger.info(f"[Experiment] Done. Total time: {elapsed:.2f}h")

    return {
        "exp_dir":       exp_path,
        "status":        "completed",
        "checkpoint":    ckpt,
        "metrics":       results,
        "elapsed_hours": elapsed,
    }


# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════

def _attach_tracking_hooks(trainer, curve_tracker, timing_tracker, failure_logger,
                           diag_tracker=None) -> None:
    """
    Attach tracking callbacks to the trainer as attributes.

    The v4 trainer (and future trainers) check for these attributes at the
    end of each epoch and call them if present. Older trainers ignore them.

    Attaches:
        trainer._on_epoch_end(epoch, metrics, duration_s)
        trainer._on_validation_end(epoch, val_duration_s)

    These are duck-typed hooks — no base class required.
    """
    def _on_epoch_end(epoch: int, metrics: dict, duration_s: float = 0.0):
        try:
            curve_tracker.record(epoch, metrics)
            curve_tracker.save()   # incremental save so partial data survives crash
        except Exception:
            pass
        try:
            timing_tracker.record_epoch(epoch, duration_s)
        except Exception:
            pass
        try:
            failure_logger.record_epoch_completed(epoch)
        except Exception:
            pass
        # ── Diagnostics: generalization gap + stability ────────────
        if diag_tracker is not None:
            try:
                train_loss = metrics.get("train_loss")
                val_dice   = metrics.get("val_dice")
                train_dice = metrics.get("train_dice")
                # Only record when we have both train and val data
                if train_loss is not None and val_dice is not None:
                    diag_tracker.record_generalization(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=metrics.get("val_loss"),
                        train_dice=train_dice,
                        val_dice=val_dice,
                    )
            except Exception:
                pass
            try:
                # Record gradient clipping events from trainer metrics
                grad_clip_count = metrics.get("grad_clip_count", 0)
                if grad_clip_count and int(grad_clip_count) > 0:
                    for _ in range(int(grad_clip_count)):
                        diag_tracker.record_stability_event(
                            "grad_clip", epoch=epoch
                        )
            except Exception:
                pass
            try:
                # Record NaN detection
                nan_detected = metrics.get("nan_detected", False)
                if nan_detected:
                    diag_tracker.record_stability_event(
                        "nan", epoch=epoch
                    )
            except Exception:
                pass

    def _on_validation_end(epoch: int, val_duration_s: float = 0.0):
        try:
            timing_tracker.record_validation(epoch, val_duration_s)
        except Exception:
            pass

    # Attach as trainer attributes — trainer calls them if hasattr check passes
    try:
        trainer._on_epoch_end      = _on_epoch_end
        trainer._on_validation_end = _on_validation_end
    except Exception:
        pass  # read-only trainer — hooks silently unavailable


def _generate_artifact_plots(trainer, hookable, plotter, logger) -> None:
    """
    Generate research-specific plots from artifact data.
    Checks both the HookableTracker's collector and legacy trainer.artifact_collector.
    Safe to call on any trainer — returns silently if no artifacts present.
    """
    # Try HookableTracker's collector first, then legacy trainer attribute
    collector = None
    if hookable is not None:
        collector = getattr(hookable, "_collector", None)
        if collector is None:
            collector = getattr(hookable, "get_collector", lambda: None)()
    if collector is None:
        collector = getattr(trainer, "artifact_collector", None)
    if collector is None:
        return

    try:
        eig_timeline = collector.get_eigenvalue_timeline()
        cond_numbers = collector.get_condition_numbers()
        if eig_timeline or cond_numbers:
            plotter.plot_spectral(eig_timeline, cond_numbers)
    except Exception as e:
        logger.debug(f"[Experiment] Spectral plot skipped: {e}")

    try:
        graph_stats = collector.get_graph_stats()
        if graph_stats:
            plotter.plot_graph_sparsity(graph_stats)
    except Exception as e:
        logger.debug(f"[Experiment] Graph sparsity plot skipped: {e}")

    try:
        attn_stats = collector.get_attention_stats()
        if attn_stats:
            plotter.plot_attention_summary(attn_stats)
    except Exception as e:
        logger.debug(f"[Experiment] Attention plot skipped: {e}")

    try:
        collector.save_stats()
    except Exception as e:
        logger.debug(f"[Experiment] Artifact stats save skipped: {e}")


def _load_json_safe(path: str) -> dict:
    """Load a JSON file; return empty dict if missing or malformed."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_eval_checkpoint(exp_dir: str) -> Optional[str]:
    """
    Find the best checkpoint for evaluation in priority order:
      1. best_model.pt (new system)
      2. best.pth      (legacy trainer v3 EMA best)
      3. best_train.pth (legacy trainer v3 training best)
      4. Latest epoch_*.pth (legacy)
    """
    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    for name in ["best_model.pt", "best.pth", "best_train.pth"]:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p

    epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pth")))
    if epoch_ckpts:
        return epoch_ckpts[-1]

    return None


def _sync_tracker_from_trainer(tracker, trainer) -> None:
    """
    Read best metric information from the trainer and record it in the tracker.

    Works with any trainer version (v3, v4, or custom) by duck-typing.
    All attribute reads are guarded with getattr so older trainers don't crash.
    """
    try:
        # v4 trainer uses best_val_metric / best_val_epoch / ckpt_mgr.monitor
        best_metric = getattr(trainer, "best_val_metric", None)
        best_epoch  = getattr(trainer, "best_val_epoch",  None)

        # v3 trainer uses best_val_tc / no best_epoch attribute
        if best_metric is None:
            best_metric = getattr(trainer, "best_val_tc", None)
            best_epoch  = None

        # Try to get the monitor name from CheckpointManager if available
        monitor_name = "val_metric"
        if hasattr(trainer, "ckpt_mgr") and hasattr(trainer.ckpt_mgr, "monitor"):
            monitor_name = trainer.ckpt_mgr.monitor
        elif best_metric is not None and hasattr(trainer, "best_val_tc"):
            monitor_name = "tc_dice"

        if best_metric is not None:
            tracker.set_best(
                epoch=best_epoch if best_epoch is not None else -1,
                metric_name=monitor_name,
                value=float(best_metric),
            )
    except Exception as e:
        logging.getLogger("run_experiment").debug(
            f"[Tracker] Could not sync best metric from trainer: {e}"
        )



def _apply_config_overrides(config, overrides: Dict[str, Any]) -> None:
    """
    Apply dot-path overrides to a Config object in-place.

    e.g. overrides = {"training.lr": 1e-4, "model.feature_size": 48}
    """
    for dot_path, value in overrides.items():
        keys = dot_path.split(".")
        obj = config
        for k in keys[:-1]:
            if not hasattr(obj, k):
                # Create sub-config on the fly
                class _SubConfig:
                    pass
                setattr(obj, k, _SubConfig())
            obj = getattr(obj, k)
        setattr(obj, keys[-1], value)


def setup_logging(level=logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ══════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run a single experiment with optional resume support."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment YAML config"
    )
    parser.add_argument(
        "--resume", type=str, default="none",
        help=(
            "'auto'  → find latest resume checkpoint automatically\n"
            "'none'  → start fresh (default)\n"
            "path    → explicit checkpoint path"
        )
    )
    parser.add_argument(
        "--resume-from-dir", type=str, default=None,
        help="Explicit experiment directory to resume from."
    )
    args = parser.parse_args()

    result = run_experiment(
        config_path=args.config,
        resume=args.resume,
        resume_from_dir=args.resume_from_dir,
    )

    # The detailed summary box is already printed by tracker.print_summary() above.
    # Print a minimal final line for script exit confirmation.
    print(f"\n[DONE] status={result['status']} | "
          f"dir={result['exp_dir']} | "
          f"elapsed={result.get('elapsed_hours', 0):.2f}h")


if __name__ == "__main__":
    main()
