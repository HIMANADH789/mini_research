"""
statistics_utils.py
===================
Multi-run statistical analysis for the mini_research pipeline.

When you have multiple runs of the same (or similar) configurations,
this module computes the statistical evidence needed for paper writing:

    - mean ± std of all metrics across runs
    - Best, median, worst run identification
    - 95% confidence intervals (t-distribution for small N, normal for N≥30)
    - Paired improvement vs baseline (delta, p-value proxy)
    - Rank ordering of model variants

Outputs:
    statistics_summary.csv    — one row per config family
    statistics_summary.json   — same + raw per-run data
    statistics_summary.md     — markdown table for paper

Usage:
    from src.utils.statistics_utils import StatisticalAnalyzer

    analyzer = StatisticalAnalyzer()
    analyzer.add_run("swinunetr", exp_dir="experiments/exp_001_swinunetr")
    analyzer.add_run("swinunetr", exp_dir="experiments/exp_002_swinunetr")
    analyzer.add_run("unet3d",    exp_dir="experiments/exp_003_unet3d")

    analyzer.compute()
    analyzer.export("reports/statistics/")

    # Or: auto-scan experiments/ folder
    analyzer = StatisticalAnalyzer.from_exp_dir("experiments/")
    analyzer.compute()
    analyzer.export("reports/statistics/")
"""

import os
import csv
import json
import math
import glob
import logging
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
#  STATISTICS HELPERS
# ══════════════════════════════════════════════════════════════

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _median(values: List[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2


def _confidence_interval_95(values: List[float]) -> Tuple[float, float]:
    """
    95% confidence interval using t-distribution (for small N).
    For N >= 30, uses z=1.96 (normal approximation).
    Returns (lower_bound, upper_bound).
    """
    n = len(values)
    if n < 2:
        m = values[0] if values else float("nan")
        return (m, m)

    m   = _mean(values)
    s   = _std(values)
    sem = s / math.sqrt(n)

    # t-critical values for 95% CI (two-tailed, df = n-1)
    # Pre-computed for small N; normal approximation for large N
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 25: 2.060, 29: 2.045,
    }
    df = n - 1
    if df <= 29:
        t = t_table.get(df, t_table.get(min(t_table.keys(), key=lambda k: abs(k - df)), 2.0))
    else:
        t = 1.96  # normal approximation for large N

    margin = t * sem
    return (round(m - margin, 6), round(m + margin, 6))


# ══════════════════════════════════════════════════════════════
#  METRIC READER
# ══════════════════════════════════════════════════════════════

METRIC_ALIAS = {
    "mean_dice": "dice_mean",
    "dice_wt":   "wt_dice",
    "dice_tc":   "tc_dice",
    "dice_et":   "et_dice",
    "hausdorff": "hd95",
}

STANDARD_METRICS = [
    "dice_mean", "wt_dice", "tc_dice", "et_dice",
    "hd95", "wt_hd95", "tc_hd95", "et_hd95",
]


def _read_metrics(exp_dir: str) -> Optional[Dict[str, float]]:
    """
    Read evaluation metrics from an experiment directory.
    Tries multiple known locations in priority order.
    """
    # 1. outputs/evaluation_results.json (primary)
    paths_to_try = [
        os.path.join(exp_dir, "outputs", "evaluation_results.json"),
        os.path.join(exp_dir, "runs", "metrics.json"),
        os.path.join(exp_dir, "metrics.json"),
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Flatten nested dicts (some metrics files have sub-dicts)
                metrics: Dict[str, float] = {}
                for k, v in raw.items():
                    if isinstance(v, dict):
                        # Try to get "final_metrics" sub-key
                        if k == "final_metrics":
                            for mk, mv in v.items():
                                nk = METRIC_ALIAS.get(mk.lower(), mk.lower())
                                if isinstance(mv, (int, float)):
                                    metrics[nk] = float(mv)
                    elif isinstance(v, (int, float)):
                        nk = METRIC_ALIAS.get(k.lower(), k.lower())
                        metrics[nk] = float(v)
                if metrics:
                    return metrics
            except Exception:
                continue
    return None


def _read_config_name(exp_dir: str) -> str:
    """Extract model name from config.yaml."""
    cfg_path = os.path.join(exp_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        return os.path.basename(exp_dir)
    try:
        import yaml
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("name", os.path.basename(exp_dir))
    except Exception:
        return os.path.basename(exp_dir)


# ══════════════════════════════════════════════════════════════
#  STATISTICAL ANALYZER
# ══════════════════════════════════════════════════════════════

class StatisticalAnalyzer:
    """
    Collects metrics from multiple experiment runs and computes
    statistical summaries for paper tables.

    Usage:
        analyzer = StatisticalAnalyzer()
        analyzer.add_run("my_model", {"dice_mean": 0.842, "tc_dice": 0.831})
        analyzer.add_run("my_model", {"dice_mean": 0.835, "tc_dice": 0.820})
        analyzer.add_run("baseline", {"dice_mean": 0.810, "tc_dice": 0.798})
        analyzer.compute()
        analyzer.export("reports/statistics/")
    """

    def __init__(self):
        # group_name → list of {exp_dir, metrics, metadata}
        self._groups: Dict[str, List[Dict]] = defaultdict(list)
        self._results: Dict[str, Dict] = {}

    # ─────────────────────────────────────────────────────────
    # DATA INGESTION
    # ─────────────────────────────────────────────────────────

    def add_run(
        self,
        group_name: str,
        metrics: Dict[str, float],
        exp_dir: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Add a single run's metrics to a named group.

        Parameters
        ----------
        group_name : model name / config family (e.g. "swinunetr")
        metrics    : dict of metric_name → float value
        exp_dir    : optional experiment directory path
        metadata   : optional extra info (seed, lr, etc.)
        """
        entry = {
            "metrics":  {k: float(v) for k, v in metrics.items()
                         if isinstance(v, (int, float))},
            "exp_dir":  exp_dir or "",
            "metadata": metadata or {},
        }
        # Normalize metric keys
        normalized_metrics: Dict[str, float] = {}
        for k, v in entry["metrics"].items():
            nk = METRIC_ALIAS.get(k.lower(), k.lower())
            normalized_metrics[nk] = v
        entry["metrics"] = normalized_metrics
        self._groups[group_name].append(entry)

    def add_run_from_dir(self, group_name: str, exp_dir: str) -> bool:
        """
        Add a run by reading metrics from an experiment directory.
        Returns True if metrics were found, False otherwise.
        """
        metrics = _read_metrics(exp_dir)
        if metrics is None:
            logger.debug(f"[Stats] No metrics found in {exp_dir}")
            return False
        self.add_run(group_name, metrics, exp_dir=exp_dir)
        return True

    @classmethod
    def from_exp_dir(cls, experiments_root: str) -> "StatisticalAnalyzer":
        """
        Auto-scan an experiments/ directory and group runs by model name.

        Runs are grouped by config 'name' field. Multiple exp_XXX_swinunetr
        folders → "swinunetr" group.
        """
        analyzer = cls()
        exp_dirs = sorted(glob.glob(os.path.join(experiments_root, "exp_*")))
        for exp_dir in exp_dirs:
            if not os.path.isdir(exp_dir):
                continue
            group = _read_config_name(exp_dir)
            added = analyzer.add_run_from_dir(group, exp_dir)
            if added:
                logger.debug(f"[Stats] Added {os.path.basename(exp_dir)} → group '{group}'")

        total_runs = sum(len(v) for v in analyzer._groups.values())
        logger.info(
            f"[Stats] Loaded {total_runs} runs across "
            f"{len(analyzer._groups)} groups from {experiments_root}"
        )
        return analyzer

    # ─────────────────────────────────────────────────────────
    # COMPUTATION
    # ─────────────────────────────────────────────────────────

    def compute(self, baseline_group: Optional[str] = None) -> Dict[str, Dict]:
        """
        Compute statistical summaries for all groups.

        Parameters
        ----------
        baseline_group : if provided, compute improvement over this group

        Returns
        -------
        dict of group_name → statistical summary
        """
        self._results = {}

        baseline_means: Dict[str, float] = {}
        if baseline_group and baseline_group in self._groups:
            baseline_runs = self._groups[baseline_group]
            all_keys = set()
            for r in baseline_runs:
                all_keys.update(r["metrics"].keys())
            for key in all_keys:
                vals = [r["metrics"][key] for r in baseline_runs if key in r["metrics"]]
                if vals:
                    baseline_means[key] = _mean(vals)

        for group_name, runs in self._groups.items():
            self._results[group_name] = self._compute_group(
                group_name, runs, baseline_means
            )

        return self._results

    def _compute_group(
        self,
        group_name: str,
        runs: List[Dict],
        baseline_means: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compute full statistics for one group of runs."""
        if not runs:
            return {}

        # Collect all metric keys across runs
        all_keys: set = set()
        for r in runs:
            all_keys.update(r["metrics"].keys())

        stats: Dict[str, Any] = {
            "group_name":  group_name,
            "n_runs":      len(runs),
            "exp_dirs":    [r["exp_dir"] for r in runs],
            "metrics":     {},
        }

        for key in sorted(all_keys):
            vals = [r["metrics"][key] for r in runs if key in r["metrics"]]
            if not vals:
                continue

            m    = _mean(vals)
            s    = _std(vals)
            ci   = _confidence_interval_95(vals)
            best_val = max(vals)
            best_run = runs[[r["metrics"].get(key, -999) for r in runs].index(best_val)]["exp_dir"]
            worst_val = min(vals)
            med  = _median(vals)

            metric_stats: Dict[str, Any] = {
                "mean":    round(m, 6),
                "std":     round(s, 6),
                "mean_std_str": f"{m:.4f} ± {s:.4f}",
                "ci_95":   [round(ci[0], 6), round(ci[1], 6)],
                "ci_95_str": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                "best":    round(best_val, 6),
                "best_run": os.path.basename(best_run) if best_run else "N/A",
                "worst":   round(worst_val, 6),
                "median":  round(med, 6),
                "all_values": [round(v, 6) for v in vals],
                "n":       len(vals),
            }

            # Improvement vs baseline
            if key in baseline_means and baseline_means:
                delta    = m - baseline_means[key]
                rel_impr = delta / abs(baseline_means[key]) * 100 if baseline_means[key] != 0 else 0
                metric_stats["delta_vs_baseline"]   = round(delta, 6)
                metric_stats["rel_improvement_pct"] = round(rel_impr, 3)

            stats["metrics"][key] = metric_stats

        # Overall rank score (average of standard metrics where available)
        rank_vals = [
            stats["metrics"][k]["mean"]
            for k in STANDARD_METRICS[:4]  # dice metrics
            if k in stats["metrics"]
        ]
        stats["rank_score"] = round(_mean(rank_vals), 6) if rank_vals else 0.0

        return stats

    # ─────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────

    def export(
        self,
        output_dir: str,
        primary_metric: str = "dice_mean",
    ) -> Dict[str, str]:
        """
        Export statistical summaries to CSV, JSON, and Markdown.

        Parameters
        ----------
        output_dir     : directory to save outputs
        primary_metric : metric used for ranking groups

        Returns
        -------
        dict of filename → absolute path
        """
        if not self._results:
            self.compute()

        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        # Sort groups by primary metric (descending)
        sorted_groups = sorted(
            self._results.items(),
            key=lambda item: item[1].get("metrics", {})
                             .get(primary_metric, {})
                             .get("mean", -999),
            reverse=True,
        )

        # Add ranks
        for rank, (gname, stats) in enumerate(sorted_groups, start=1):
            stats["rank"] = rank

        # ── CSV ─────────────────────────────────────────────
        csv_path = os.path.join(output_dir, "statistics_summary.csv")
        rows = self._build_csv_rows(sorted_groups, primary_metric)
        _write_csv(rows, csv_path)
        paths["statistics_summary.csv"] = csv_path

        # ── JSON ────────────────────────────────────────────
        json_path = os.path.join(output_dir, "statistics_summary.json")
        _atomic_json(
            {
                "generated_at": _now_iso(),
                "primary_metric": primary_metric,
                "n_groups": len(self._results),
                "total_runs": sum(s["n_runs"] for s in self._results.values()),
                "groups": {name: stats for name, stats in sorted_groups},
            },
            json_path,
        )
        paths["statistics_summary.json"] = json_path

        # ── Markdown ────────────────────────────────────────
        md_path = os.path.join(output_dir, "statistics_summary.md")
        _write_text(self._build_markdown_table(sorted_groups, primary_metric), md_path)
        paths["statistics_summary.md"] = md_path

        logger.info(
            f"[Stats] Exported statistics for {len(self._results)} groups → {output_dir}"
        )
        return paths

    def _build_csv_rows(
        self,
        sorted_groups: List[Tuple[str, Dict]],
        primary_metric: str,
    ) -> List[Dict[str, Any]]:
        rows = []
        for gname, stats in sorted_groups:
            row: Dict[str, Any] = {
                "rank":       stats.get("rank", "N/A"),
                "group_name": gname,
                "n_runs":     stats.get("n_runs", 0),
            }
            for key, mdata in stats.get("metrics", {}).items():
                row[f"{key}_mean"]    = mdata.get("mean", "N/A")
                row[f"{key}_std"]     = mdata.get("std", "N/A")
                row[f"{key}_mean_std"] = mdata.get("mean_std_str", "N/A")
                row[f"{key}_best"]    = mdata.get("best", "N/A")
                row[f"{key}_ci95_lo"] = mdata.get("ci_95", ["N/A", "N/A"])[0]
                row[f"{key}_ci95_hi"] = mdata.get("ci_95", ["N/A", "N/A"])[1]
                if "delta_vs_baseline" in mdata:
                    row[f"{key}_delta"]   = mdata["delta_vs_baseline"]
            rows.append(row)
        return rows

    def _build_markdown_table(
        self,
        sorted_groups: List[Tuple[str, Dict]],
        primary_metric: str,
    ) -> str:
        """Build a markdown table suitable for a LaTeX/paper draft."""
        # Determine which standard metrics are available in any group
        available_metrics = []
        for key in STANDARD_METRICS:
            for _, stats in sorted_groups:
                if key in stats.get("metrics", {}):
                    available_metrics.append(key)
                    break

        lines = [
            f"# Statistical Summary",
            f"",
            f"**Generated:** {_now_iso()}  ",
            f"**Primary Metric:** {primary_metric}  ",
            f"**Runs per group:** see n_runs column  ",
            f"",
            f"Values shown as **mean ± std** (best in brackets)",
            f"",
            "| Rank | Model | N |" + "".join(f" {m} |" for m in available_metrics),
            "|------|-------|---|" + "".join(f"---------|" for _ in available_metrics),
        ]

        for gname, stats in sorted_groups:
            rank = stats.get("rank", "-")
            n    = stats.get("n_runs", 0)
            row  = f"| {rank} | **{gname}** | {n} |"
            for key in available_metrics:
                mdata = stats.get("metrics", {}).get(key, {})
                if mdata:
                    mean = mdata.get("mean", 0)
                    std  = mdata.get("std", 0)
                    best = mdata.get("best", 0)
                    cell = f" {mean:.4f}±{std:.4f} [{best:.4f}] |"
                else:
                    cell = " N/A |"
                row += cell
            lines.append(row)

        lines += [
            "",
            "---",
            "",
            "## Per-Group Details",
            "",
        ]
        for gname, stats in sorted_groups:
            lines.append(f"### {gname} (n={stats.get('n_runs',0)})")
            for key, mdata in stats.get("metrics", {}).items():
                ci = mdata.get("ci_95_str", "N/A")
                delta_str = ""
                if "delta_vs_baseline" in mdata:
                    d = mdata["delta_vs_baseline"]
                    pct = mdata.get("rel_improvement_pct", 0)
                    delta_str = f" | Δ vs baseline: {d:+.4f} ({pct:+.1f}%)"
                lines.append(
                    f"- **{key}**: {mdata.get('mean_std_str','N/A')} "
                    f"| 95% CI: {ci}{delta_str}"
                )
            lines.append("")

        return "\n".join(lines)

    def print_summary(self, primary_metric: str = "dice_mean") -> None:
        """Print a concise leaderboard to stdout."""
        if not self._results:
            self.compute()

        sorted_groups = sorted(
            self._results.items(),
            key=lambda item: item[1].get("metrics", {})
                             .get(primary_metric, {})
                             .get("mean", -999),
            reverse=True,
        )

        W = 80
        print("\n" + "=" * W)
        print(f"  STATISTICAL SUMMARY  |  ranked by {primary_metric}")
        print("=" * W)
        header = f"{'Rank':<5} {'Model':<25} {'N':>3}  {'Mean':>8}  {'±Std':>7}  {'Best':>8}  {'95% CI':<20}"
        print(header)
        print("-" * W)

        for rank, (gname, stats) in enumerate(sorted_groups, start=1):
            mdata = stats.get("metrics", {}).get(primary_metric, {})
            if not mdata:
                continue
            mean = mdata.get("mean", 0)
            std  = mdata.get("std", 0)
            best = mdata.get("best", 0)
            n    = stats.get("n_runs", 0)
            ci   = mdata.get("ci_95", ["N/A", "N/A"])
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if len(ci) == 2 else "N/A"
            print(
                f"{rank:<5} {gname:<25} {n:>3}  "
                f"{mean:>8.4f}  ±{std:>6.4f}  {best:>8.4f}  {ci_str:<20}"
            )
        print("=" * W + "\n")


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    all_keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def _write_text(content: str, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        os.close(fd)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
