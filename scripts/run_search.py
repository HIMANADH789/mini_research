"""
run_search.py
=============
Automated hyperparameter search orchestrator.

Reads a search space YAML, generates experiment configs, runs each trial,
tracks status, and produces a ranked leaderboard.

Usage:
    # Grid search
    python scripts/run_search.py --space configs/search_space_example.yaml

    # Random search (override method and max trials at CLI)
    python scripts/run_search.py --space configs/search_space_example.yaml \\
        --method random --trials 8

    # Resume interrupted search (just re-run the same command)
    python scripts/run_search.py --space configs/search_space_example.yaml

    # Skip failed trials (don't retry them)
    python scripts/run_search.py --space configs/search_space_example.yaml --skip-failed

Design:
  - All trial configs are saved as YAML to search_results/<name>/trial_NNN/config.yaml
  - search_state.json tracks status of each trial (persistent across runs)
  - After each trial, summary.csv and summary.json are updated
  - Final leaderboard is printed at the end

Future extension:
  - Optuna: swap run_trial() internals, keep the rest
  - Bayesian / RL: replace SearchSpace.generate_trials() with a controller loop
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
import json
import time
import traceback
from datetime import datetime
from typing import Optional

from src.utils.search_utils import SearchSpace, TrialIndex, Trial
from src.utils.aggregate_results import aggregate
from src.utils.statistics_utils import StatisticalAnalyzer, _read_config_name

# Import run_experiment as a function (not subprocess) for cleaner error handling
from scripts.run_experiment import run_experiment, setup_logging


# ══════════════════════════════════════════════════════
#  SEARCH RUNNER
# ══════════════════════════════════════════════════════

def run_search(
    space_yaml_path: str,
    method_override: Optional[str] = None,
    max_trials_override: Optional[int] = None,
    skip_failed: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    """
    Main search loop.

    Parameters
    ----------
    space_yaml_path      : path to search space YAML
    method_override      : override search.method from CLI ('grid' | 'random')
    max_trials_override  : override search.max_trials from CLI
    skip_failed          : if True, do not retry previously failed trials
    output_dir           : root directory for search results
                           (default: search_results/<space_name>/)
    """
    setup_logging()
    log = logging.getLogger("run_search")

    log.info("=" * 70)
    log.info(" SABiT Research Pipeline — Hyperparameter Search")
    log.info(f" Space YAML: {space_yaml_path}")
    log.info(f" Time:       {datetime.now().isoformat()}")
    log.info("=" * 70)

    # ── Load search space ───────────────────────────────────────────
    space_name = os.path.splitext(os.path.basename(space_yaml_path))[0]

    if output_dir is None:
        output_dir = os.path.join("search_results", space_name)

    os.makedirs(output_dir, exist_ok=True)

    space = SearchSpace.from_yaml(space_yaml_path, output_dir=output_dir)

    # CLI overrides
    if method_override:
        space.method = method_override.lower()
        log.info(f"[Search] Method overridden to: {space.method}")
    if max_trials_override is not None:
        space.max_trials = max_trials_override
        log.info(f"[Search] max_trials overridden to: {space.max_trials}")

    # ── Generate trials ──────────────────────────────────────────────
    trials = space.generate_trials()

    # ── Trial index (persistent state) ──────────────────────────────
    state_path = os.path.join(output_dir, "search_state.json")
    index = TrialIndex(state_path)
    index.register_all(trials)

    # ── Filter to only trials that need to run ───────────────────────
    pending = index.pending_trials(trials, skip_failed=skip_failed)

    log.info(
        f"[Search] Total trials: {len(trials)} | "
        f"Pending: {len(pending)} | "
        f"Completed: {len(trials) - len(pending)}"
    )

    if not pending:
        log.info("[Search] All trials already completed. Nothing to do.")
        log.info("[Search] Re-running aggregate results...")
        _finalize(output_dir, space, log)
        return

    # ── Run each pending trial ───────────────────────────────────────
    search_start = time.time()
    completed_count = 0
    failed_count = 0

    for trial in pending:
        log.info("\n" + "─" * 60)
        log.info(f"[Search] Trial {trial.trial_id:03d}/{len(trials) - 1}")
        log.info(f"[Search] Overrides: {trial.overrides}")

        # Create trial directory and save expanded config
        trial_dir = os.path.join(output_dir, f"trial_{trial.trial_id:03d}")
        os.makedirs(trial_dir, exist_ok=True)
        trial.trial_dir = trial_dir

        trial_config_path = os.path.join(trial_dir, "config.yaml")
        trial.save_config(trial_config_path)
        log.info(f"[Search] Trial config saved → {trial_config_path}")

        index.mark_running(trial)

        trial_start = time.time()
        try:
            result = run_experiment(
                config_path=trial_config_path,
                resume="none",          # each trial starts fresh
                config_overrides=None,   # already baked into trial config
                exp_name_suffix=f"_trial_{trial.trial_id:03d}",
            )

            elapsed = round((time.time() - trial_start) / 3600, 2)
            result["elapsed_hours"] = elapsed
            result["trial_id"]      = trial.trial_id
            result["overrides"]     = trial.overrides

            index.mark_completed(trial, result)
            completed_count += 1

            primary = result.get("metrics", {}).get(space.metric, "N/A")
            log.info(
                f"[Search] Trial {trial.trial_id:03d} COMPLETED | "
                f"{space.metric}={primary} | {elapsed:.2f}h"
            )

        except KeyboardInterrupt:
            log.warning(f"[Search] Trial {trial.trial_id:03d} INTERRUPTED by user.")
            index.mark_failed(trial, "KeyboardInterrupt")
            # Save partial summary before exiting
            _save_intermediate_summary(output_dir, index, space, log)
            log.warning("[Search] Exiting. Re-run the same command to resume.")
            raise

        except Exception as e:
            elapsed = round((time.time() - trial_start) / 3600, 2)
            tb_str = traceback.format_exc()
            log.error(f"[Search] Trial {trial.trial_id:03d} FAILED after {elapsed:.2f}h")
            log.error(f"[Search] Error: {e}")
            log.debug(tb_str)

            index.mark_failed(trial, str(e))
            failed_count += 1

            # Save error log in trial dir
            err_path = os.path.join(trial_dir, "error.txt")
            with open(err_path, "w") as f:
                f.write(f"Trial {trial.trial_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Overrides: {trial.overrides}\n\n")
                f.write(tb_str)
            continue

        finally:
            # Always update the partial summary after each trial
            _save_intermediate_summary(output_dir, index, space, log)

    # ── Final summary ────────────────────────────────────────────────
    total_elapsed = round((time.time() - search_start) / 3600, 2)
    log.info("\n" + "=" * 60)
    log.info(f" SEARCH COMPLETE")
    log.info(f" Completed: {completed_count}/{len(pending)}")
    log.info(f" Failed:    {failed_count}/{len(pending)}")
    log.info(f" Total time: {total_elapsed:.2f}h")
    log.info("=" * 60)

    _finalize(output_dir, space, log)


# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════

def _save_intermediate_summary(output_dir: str, index: TrialIndex,
                                space: SearchSpace, log: logging.Logger) -> None:
    """
    Write a search_summary.json with all trial results seen so far.
    Called after each trial (success or failure) for live monitoring.
    """
    all_data = index.get_all()
    summary_path = os.path.join(output_dir, "search_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "metric":       space.metric,
                    "mode":         space.mode,
                    "trials":       list(all_data.values()),
                },
                f, indent=2, default=str,
            )
        log.debug(f"[Search] Intermediate summary → {summary_path}")
    except Exception as e:
        log.warning(f"[Search] Could not save intermediate summary: {e}")


def _finalize(output_dir: str, space: SearchSpace, log: logging.Logger) -> None:
    """
    Run aggregate_results over all experiment folders created by this search,
    and print final leaderboard.
    """
    # Collect experiment dirs from all completed trials
    exp_folders = []
    state_path = os.path.join(output_dir, "search_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        for entry in state.values():
            if entry.get("status") == "completed":
                exp_dir = entry.get("result", {}).get("exp_dir", "")
                if exp_dir and os.path.isdir(exp_dir):
                    exp_folders.append(exp_dir)

    if not exp_folders:
        log.info("[Search] No completed experiments to aggregate.")
        return

    # ── Cross-run statistical analysis ───────────────────────────────
    log.info(f"\n[Search] Running statistical analysis on {len(exp_folders)} experiments...")
    try:
        analyzer = StatisticalAnalyzer()
        for exp_dir in exp_folders:
            group_name = _read_config_name(exp_dir)
            analyzer.add_run_from_dir(group_name, exp_dir)
        analyzer.compute()
        analyzer.export(
            output_dir=output_dir,
            primary_metric=space.metric,
        )
        log.info(f"[Search] Statistics exported → {output_dir}/statistics_summary.*")
    except Exception as e:
        log.warning(f"[Search] Statistical analysis failed (non-fatal): {e}")

    # Write a minimal experiments-like structure for aggregate_results
    # We directly call aggregate on the parent experiments/ directory
    experiments_root = "experiments"
    if os.path.isdir(experiments_root):
        log.info(f"\n[Search] Aggregating results from {experiments_root}/...")
        try:
            aggregate(
                exp_dir=experiments_root,
                primary_metric=space.metric,
                mode=space.mode,
                top_n=5,
                output_dir=output_dir,
            )
        except Exception as e:
            log.warning(f"[Search] Aggregate failed: {e}")
    else:
        log.warning(f"[Search] experiments/ directory not found for aggregation.")


# ══════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter search over a search space YAML."
    )
    parser.add_argument(
        "--space", type=str, required=True,
        help="Path to search space YAML (e.g. configs/search_space_example.yaml)"
    )
    parser.add_argument(
        "--method", type=str, default=None, choices=["grid", "random"],
        help="Override search method from YAML (grid | random)"
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="Override max_trials from YAML"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output directory for search state and results "
             "(default: search_results/<space_name>/)"
    )
    parser.add_argument(
        "--skip-failed", action="store_true",
        help="Skip trials that previously failed (default: retry them)"
    )
    args = parser.parse_args()

    run_search(
        space_yaml_path=args.space,
        method_override=args.method,
        max_trials_override=args.trials,
        skip_failed=args.skip_failed,
        output_dir=args.out,
    )


if __name__ == "__main__":
    main()
