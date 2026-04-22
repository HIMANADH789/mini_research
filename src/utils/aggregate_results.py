"""
aggregate_results.py
====================
Reads all experiment folders and builds a ranked leaderboard.

Each experiment is expected to have:
    outputs/evaluation_results.json  — from evaluate.py
    training_state.json              — from CheckpointManager (optional, adds epoch info)
    config.yaml                      — saved at training start (optional, adds model name)

Usage:
    # From project root
    python src/utils/aggregate_results.py --exp-dir experiments/ --top 5

    # Save outputs to a custom location
    python src/utils/aggregate_results.py --exp-dir experiments/ --out reports/ --top 10

Outputs:
    summary.csv     — all experiments, all metrics, sorted by primary metric
    summary.json    — same data as JSON
    leaderboard     — top-N printed to stdout
"""

import os
import sys
import csv
import json
import glob
import argparse
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
#  PRIMARY METRIC KEYS (order matters for display)
# ═══════════════════════════════════════════════════════

PRIMARY_METRIC = "dice_mean"

METRIC_COLUMNS = [
    "dice_mean",
    "wt_dice",
    "tc_dice",
    "et_dice",
    "hd95",
    "hd95_wt",
    "hd95_tc",
    "hd95_et",
    "best_epoch",
    "epochs_completed",
    "total_time_hours",
]

# Common alias mappings: some evaluators save under different key names
ALIAS_MAP = {
    "mean_dice":    "dice_mean",
    "dice_wt":      "wt_dice",
    "dice_tc":      "tc_dice",
    "dice_et":      "et_dice",
    "hausdorff":    "hd95",
    "hd":           "hd95",
}


def _normalize_key(key: str) -> str:
    """Normalize metric key: lowercase, strip, apply aliases."""
    key = key.lower().strip()
    return ALIAS_MAP.get(key, key)


def _read_json(path: str) -> Optional[Dict]:
    """Read a JSON file, return None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"Could not read {path}: {e}")
        return None


def _read_yaml_field(yaml_path: str, field: str) -> Optional[Any]:
    """Read a single top-level field from a YAML file."""
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return d.get(field) if isinstance(d, dict) else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════
#  EXPERIMENT READER
# ═══════════════════════════════════════════════════════

def read_experiment(exp_dir: str) -> Optional[Dict]:
    """
    Read all available metrics from a single experiment directory.

    Returns a flat dict of metrics, or None if the experiment has
    no evaluation results.
    """
    exp_name = os.path.basename(exp_dir.rstrip(os.sep))

    # ── Evaluation results (required) ──────────────────────────────
    eval_path = os.path.join(exp_dir, "outputs", "evaluation_results.json")
    if not os.path.exists(eval_path):
        logger.debug(f"[Aggregator] No evaluation results in {exp_name} — skipping.")
        return None

    raw_eval = _read_json(eval_path) or {}
    metrics: Dict[str, Any] = {"exp_name": exp_name, "exp_dir": exp_dir}

    # Normalize all metric keys
    for k, v in raw_eval.items():
        normalized = _normalize_key(k)
        if isinstance(v, (int, float)):
            metrics[normalized] = round(float(v), 6)
        else:
            metrics[normalized] = v  # keep strings etc.

    # ── Training state (optional) ───────────────────────────────────
    state_path = os.path.join(exp_dir, "training_state.json")
    state = _read_json(state_path) or {}
    if "best_epoch" in state:
        metrics["best_epoch"]        = state["best_epoch"]
    if "last_epoch" in state:
        metrics["epochs_completed"]  = state["last_epoch"] + 1
    if "best_metric" in state:
        # Prefer evaluation result, but fill in if missing
        monitor = state.get("monitor", "dice_mean")
        normalized_monitor = _normalize_key(monitor)
        if normalized_monitor not in metrics:
            metrics[normalized_monitor] = round(float(state["best_metric"]), 6)

    # ── Config info (optional) ──────────────────────────────────────
    config_path = os.path.join(exp_dir, "config.yaml")
    if os.path.exists(config_path):
        model_name = _read_yaml_field(config_path, "name")
        if model_name:
            metrics["model_name"] = str(model_name)

    # ── File timestamps for total_time estimate ──────────────────────
    try:
        start_time = os.path.getctime(exp_dir)
        if os.path.exists(state_path):
            end_time = os.path.getmtime(state_path)
            elapsed_h = (end_time - start_time) / 3600.0
            metrics["total_time_hours"] = round(elapsed_h, 2)
    except Exception:
        pass

    return metrics


# ═══════════════════════════════════════════════════════
#  AGGREGATOR
# ═══════════════════════════════════════════════════════

def aggregate(
    exp_dir: str,
    primary_metric: str = PRIMARY_METRIC,
    mode: str = "max",
    top_n: int = 5,
    output_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Scan experiment directory, collect all results, sort by primary metric.

    Parameters
    ----------
    exp_dir        : root folder containing exp_* subdirectories
    primary_metric : metric name to rank by (e.g. 'dice_mean')
    mode           : 'max' or 'min'
    top_n          : number of top experiments to print
    output_dir     : where to save summary.csv / summary.json
                     (defaults to exp_dir)

    Returns
    -------
    Sorted list of metric dicts.
    """
    if not os.path.isdir(exp_dir):
        logger.error(f"[Aggregator] exp_dir not found: {exp_dir}")
        return []

    if output_dir is None:
        output_dir = exp_dir

    # ── Scan experiment folders ─────────────────────────────────────
    exp_dirs = sorted([
        d for d in glob.glob(os.path.join(exp_dir, "exp_*"))
        if os.path.isdir(d)
    ])

    if not exp_dirs:
        logger.warning(f"[Aggregator] No exp_* folders found in {exp_dir}")
        return []

    logger.info(f"[Aggregator] Found {len(exp_dirs)} experiment folders.")

    rows = []
    for d in exp_dirs:
        result = read_experiment(d)
        if result is not None:
            rows.append(result)

    if not rows:
        logger.warning("[Aggregator] No experiments with evaluation results found.")
        return []

    # ── Sort by primary metric ──────────────────────────────────────
    reverse = (mode == "max")
    rows.sort(
        key=lambda r: float(r.get(primary_metric, -1e9 if mode == "max" else 1e9)),
        reverse=reverse,
    )

    # ── Rank column ─────────────────────────────────────────────────
    for i, row in enumerate(rows):
        row["rank"] = i + 1

    # ── Save CSV ────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "summary.csv")
    _save_csv(rows, csv_path, primary_metric)
    logger.info(f"[Aggregator] CSV saved → {csv_path}")

    # ── Save JSON ───────────────────────────────────────────────────
    json_path = os.path.join(output_dir, "summary.json")
    _save_json(rows, json_path)
    logger.info(f"[Aggregator] JSON saved → {json_path}")

    # ── Print leaderboard ───────────────────────────────────────────
    _print_leaderboard(rows, primary_metric, top_n)

    return rows


def _save_csv(rows: List[Dict], path: str, primary_metric: str) -> None:
    """Write summary CSV with a consistent column order."""
    if not rows:
        return

    # Build ordered column list: rank, exp_name, model_name, primary, rest
    priority = ["rank", "exp_name", "model_name", primary_metric]
    other_metrics = [c for c in METRIC_COLUMNS if c != primary_metric]
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    columns = []
    for c in priority + other_metrics:
        if c in all_keys:
            columns.append(c)
    # Append any remaining keys
    for k in sorted(all_keys):
        if k not in columns and k != "exp_dir":
            columns.append(k)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_json(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": datetime.now().isoformat(), "results": rows},
            f, indent=2, default=str,
        )


def _print_leaderboard(rows: List[Dict], primary_metric: str, top_n: int) -> None:
    """Print a clean leaderboard to stdout."""
    show = rows[:top_n]

    col_widths = {
        "rank":           5,
        "exp_name":       35,
        "model_name":     18,
        primary_metric:   12,
        "wt_dice":        10,
        "tc_dice":        10,
        "et_dice":        10,
        "hd95":           10,
        "best_epoch":     12,
        "epochs_completed": 18,
    }

    display_cols = ["rank", "exp_name", "model_name", primary_metric,
                    "wt_dice", "tc_dice", "et_dice", "hd95",
                    "best_epoch", "epochs_completed"]

    # Filter to only columns that exist in any row
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    display_cols = [c for c in display_cols if c in all_keys]

    header = "".join(c.ljust(col_widths.get(c, 12)) for c in display_cols)

    sep = "=" * len(header)
    print(f"\n{sep}")
    print(f" LEADERBOARD  (top {min(top_n, len(show))}/{len(rows)})  |  ranked by {primary_metric}")
    print(sep)
    print(header)
    print("-" * len(header))

    for row in show:
        line_parts = []
        for c in display_cols:
            val = row.get(c, "")
            if isinstance(val, float):
                s = f"{val:.4f}"
            else:
                s = str(val)
            w = col_widths.get(c, 12)
            line_parts.append(s.ljust(w))
        print("".join(line_parts))

    print(sep + "\n")


# ═══════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Aggregate experiment results into a ranked leaderboard."
    )
    parser.add_argument(
        "--exp-dir", type=str, default="experiments",
        help="Root directory containing exp_* subdirectories (default: experiments/)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output directory for summary.csv / summary.json. "
             "Defaults to --exp-dir."
    )
    parser.add_argument(
        "--metric", type=str, default=PRIMARY_METRIC,
        help=f"Primary metric to rank by (default: {PRIMARY_METRIC})"
    )
    parser.add_argument(
        "--mode", type=str, choices=["max", "min"], default="max",
        help="Whether higher or lower metric is better (default: max)"
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of top experiments to show in leaderboard (default: 5)"
    )
    args = parser.parse_args()

    aggregate(
        exp_dir=args.exp_dir,
        primary_metric=args.metric,
        mode=args.mode,
        top_n=args.top,
        output_dir=args.out,
    )


if __name__ == "__main__":
    # Allow running from project root
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    main()
