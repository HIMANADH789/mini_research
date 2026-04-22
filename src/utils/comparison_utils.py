"""
comparison_utils.py
===================
Cross-experiment comparison and paper-ready export utilities.

Provides:
  1. ExperimentComparator — scans multiple experiment folders, extracts metrics,
     generates side-by-side comparison tables in CSV, JSON, Markdown, and LaTeX.
  2. AblationAnalyzer — given a list of experiments with labeled changes,
     generates ablation study tables with delta improvements.
  3. latex_table() — format any DataFrame as publication-ready LaTeX.

Usage:
    from src.utils.comparison_utils import ExperimentComparator

    comp = ExperimentComparator()
    comp.add_experiment("experiments/exp_001_unet3d")
    comp.add_experiment("experiments/exp_005_resunet")
    comp.add_experiment("experiments/exp_012_sabit")
    comp.export("results/comparison", primary_metric="tc_dice")

This generates:
    results/comparison/
        comparison_table.csv
        comparison_table.json
        comparison_table.md
        comparison_table.tex
        runtime_comparison.csv
        params_comparison.csv
"""

import os
import json
import csv
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  SAFE FILE I/O
# ══════════════════════════════════════════════════════════════

def _load_json_safe(path: str) -> dict:
    """Load JSON, return {} on any error."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _safe(fn, default=None):
    """Execute fn, return default on exception."""
    try:
        return fn()
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT COMPARATOR
# ══════════════════════════════════════════════════════════════

class ExperimentComparator:
    """
    Scans experiment folders and produces publication-ready comparison tables.

    Each experiment folder is expected to contain:
        - summary.json    (best metrics, model info)
        - metrics.json    (final evaluation results)
        - model_info.json (param counts, architecture)
        - runtime.json or runs/timing.json (timing info)
        - system_info.json (hardware)
        - model_profile.json (v5 trainer param breakdown)
    """

    def __init__(self):
        self._experiments: List[Dict[str, Any]] = []

    def add_experiment(self, exp_dir: str, label: Optional[str] = None) -> None:
        """
        Add an experiment folder to the comparison.

        Parameters
        ----------
        exp_dir : str
            Path to the experiment folder.
        label : str, optional
            Human-readable name for this experiment in tables.
            Defaults to the folder basename.
        """
        if not os.path.isdir(exp_dir):
            logger.warning(f"[Comparator] Experiment folder not found: {exp_dir}")
            return

        exp_name = label or os.path.basename(exp_dir)

        # Load all available metadata files
        summary    = _load_json_safe(os.path.join(exp_dir, "summary.json"))
        metrics    = _load_json_safe(os.path.join(exp_dir, "metrics.json"))
        model_info = _load_json_safe(os.path.join(exp_dir, "model_info.json"))
        system     = _load_json_safe(os.path.join(exp_dir, "system_info.json"))
        profile    = _load_json_safe(os.path.join(exp_dir, "model_profile.json"))

        # Try both timing locations
        timing = _load_json_safe(os.path.join(exp_dir, "runtime.json"))
        if not timing:
            timing = _load_json_safe(os.path.join(exp_dir, "runs", "timing.json"))

        # Training state for epoch info
        state = _load_json_safe(os.path.join(exp_dir, "training_state.json"))
        if not state:
            state = _load_json_safe(os.path.join(exp_dir, "checkpoints", "training_state.json"))

        entry = {
            "name":        exp_name,
            "exp_dir":     exp_dir,
            "summary":     summary,
            "metrics":     metrics,
            "model_info":  model_info,
            "system":      system,
            "timing":      timing,
            "profile":     profile,
            "state":       state,
        }

        self._experiments.append(entry)
        logger.info(f"[Comparator] Added: {exp_name} ({exp_dir})")

    def add_experiments_from_dir(self, parent_dir: str) -> int:
        """
        Scan a parent directory for experiment folders and add all found.
        Returns the number of experiments added.
        """
        if not os.path.isdir(parent_dir):
            return 0

        count = 0
        for entry in sorted(os.listdir(parent_dir)):
            full = os.path.join(parent_dir, entry)
            if os.path.isdir(full) and (
                os.path.exists(os.path.join(full, "summary.json")) or
                os.path.exists(os.path.join(full, "metrics.json"))
            ):
                self.add_experiment(full)
                count += 1

        return count

    # ──────────────────────────────────────────────────────
    # TABLE BUILDERS
    # ──────────────────────────────────────────────────────

    def build_metrics_table(self, primary_metric: str = "tc_dice") -> List[Dict]:
        """
        Build a comparison table of evaluation metrics.

        Returns a list of dicts (one per experiment) with standardized keys.
        """
        rows = []
        for exp in self._experiments:
            m = exp["metrics"]
            s = exp["summary"]

            row = {
                "Model":       exp["name"],
                "Dice_WT":     _safe(lambda: round(m.get("wt_dice", m.get("dice_wt", 0)), 4)),
                "Dice_TC":     _safe(lambda: round(m.get("tc_dice", m.get("dice_tc", 0)), 4)),
                "Dice_ET":     _safe(lambda: round(m.get("et_dice", m.get("dice_et", 0)), 4)),
                "Dice_Mean":   _safe(lambda: round(m.get("mean_dice", m.get("dice_mean", 0)), 4)),
                "HD95_WT":     _safe(lambda: round(m.get("hd95_wt", m.get("wt_hd95", 0)), 2)),
                "HD95_TC":     _safe(lambda: round(m.get("hd95_tc", m.get("tc_hd95", 0)), 2)),
                "HD95_ET":     _safe(lambda: round(m.get("hd95_et", m.get("et_hd95", 0)), 2)),
                "Best_Epoch":  _safe(lambda: s.get("best_epoch", "N/A")),
            }
            rows.append(row)

        # Sort by primary metric descending
        _metric_key = {
            "tc_dice": "Dice_TC", "wt_dice": "Dice_WT", "et_dice": "Dice_ET",
            "mean_dice": "Dice_Mean", "dice_mean": "Dice_Mean",
        }
        sort_key = _metric_key.get(primary_metric, "Dice_TC")
        rows.sort(key=lambda r: r.get(sort_key, 0) or 0, reverse=True)

        return rows

    def build_runtime_table(self) -> List[Dict]:
        """Build a comparison of runtime / hardware info."""
        rows = []
        for exp in self._experiments:
            t = exp["timing"]
            s = exp["system"]

            row = {
                "Model":            exp["name"],
                "GPU":              _safe(lambda: s.get("gpu_name", "N/A")),
                "Params_M":         _safe(lambda: round(
                    exp["profile"].get("total_params", exp["model_info"].get("total_params", 0))
                    / 1e6, 2
                )),
                "Trainable_M":      _safe(lambda: round(
                    exp["profile"].get("trainable_params",
                                       exp["model_info"].get("trainable_params", 0))
                    / 1e6, 2
                )),
                "Memory_MB":        _safe(lambda: round(
                    exp["profile"].get("memory_mb", 0), 1
                )),
                "Train_Hours":      _safe(lambda: round(
                    t.get("total_train_time_s", t.get("wall_time_s", 0)) / 3600, 2
                )),
                "Peak_VRAM_GB":     _safe(lambda: round(
                    s.get("peak_vram_gb", t.get("gpu_memory_max_used_gb", 0)), 2
                )),
                "Epochs":           _safe(lambda: exp["state"].get("last_epoch", "N/A")),
            }
            rows.append(row)

        return rows

    def build_params_table(self) -> List[Dict]:
        """Build per-module parameter breakdown (from model_profile.json)."""
        rows = []
        for exp in self._experiments:
            profile = exp["profile"]
            breakdown = profile.get("breakdown", {})
            row = {"Model": exp["name"]}
            for module_name, info in breakdown.items():
                row[module_name] = f"{info.get('params', 0):,} ({info.get('percent', 0)}%)"
            rows.append(row)
        return rows

    # ──────────────────────────────────────────────────────
    # EXPORT
    # ──────────────────────────────────────────────────────

    def export(self, output_dir: str, primary_metric: str = "tc_dice") -> None:
        """
        Export all comparison tables to output_dir.

        Generates:
            comparison_table.csv / .json / .md / .tex
            runtime_comparison.csv
            params_comparison.csv
        """
        os.makedirs(output_dir, exist_ok=True)

        if not self._experiments:
            logger.warning("[Comparator] No experiments to compare.")
            return

        # ── Metrics table ─────────────────────────────────
        metrics_rows = self.build_metrics_table(primary_metric)
        self._write_csv(metrics_rows, os.path.join(output_dir, "comparison_table.csv"))
        self._write_json(metrics_rows, os.path.join(output_dir, "comparison_table.json"))
        self._write_markdown(metrics_rows, os.path.join(output_dir, "comparison_table.md"),
                             title="Segmentation Results Comparison")
        self._write_latex(metrics_rows, os.path.join(output_dir, "comparison_table.tex"),
                          caption="Comparison of segmentation methods on BraTS 2021.",
                          label="tab:comparison",
                          bold_best=True, primary_metric=primary_metric)

        # ── Runtime table ─────────────────────────────────
        runtime_rows = self.build_runtime_table()
        self._write_csv(runtime_rows, os.path.join(output_dir, "runtime_comparison.csv"))
        self._write_markdown(runtime_rows, os.path.join(output_dir, "runtime_comparison.md"),
                             title="Runtime & Resource Comparison")
        self._write_latex(runtime_rows, os.path.join(output_dir, "runtime_comparison.tex"),
                          caption="Computational cost comparison.",
                          label="tab:runtime")

        # ── Params breakdown ──────────────────────────────
        params_rows = self.build_params_table()
        if any(len(r) > 1 for r in params_rows):
            self._write_csv(params_rows, os.path.join(output_dir, "params_comparison.csv"))

        logger.info(f"[Comparator] Exported comparison tables to {output_dir}")

    # ──────────────────────────────────────────────────────
    # FORMAT WRITERS
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _write_csv(rows: List[Dict], path: str) -> None:
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_json(rows: List[Dict], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)

    @staticmethod
    def _write_markdown(rows: List[Dict], path: str, title: str = "") -> None:
        if not rows:
            return
        keys = list(rows[0].keys())

        lines = []
        if title:
            lines.append(f"# {title}\n")
            lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

        # Header
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join(["---"] * len(keys)) + "|")

        # Rows
        for row in rows:
            vals = [str(row.get(k, "")) for k in keys]
            lines.append("| " + " | ".join(vals) + " |")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    @staticmethod
    def _write_latex(rows: List[Dict], path: str,
                     caption: str = "", label: str = "tab:results",
                     bold_best: bool = False, primary_metric: str = "") -> None:
        """Write a LaTeX table ready for paper insertion."""
        if not rows:
            return

        keys = list(rows[0].keys())

        # Find best values per column (for bolding)
        best_vals = {}
        if bold_best:
            for k in keys:
                if k == "Model":
                    continue
                vals = []
                for r in rows:
                    v = r.get(k)
                    if isinstance(v, (int, float)) and v is not None:
                        vals.append(v)
                if vals:
                    # HD95 columns: lower is better
                    if "HD95" in k or "hd95" in k:
                        best_vals[k] = min(vals)
                    else:
                        best_vals[k] = max(vals)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular}{" + "l" + "c" * (len(keys) - 1) + "}",
            r"\toprule",
        ]

        # Header row
        header = " & ".join(k.replace("_", r"\_") for k in keys) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        # Data rows
        for row in rows:
            cells = []
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, float):
                    cell = f"{v:.4f}" if v < 1 else f"{v:.2f}"
                else:
                    cell = str(v) if v is not None else ""

                # Bold best value
                if bold_best and k in best_vals and isinstance(v, (int, float)):
                    if v == best_vals[k]:
                        cell = r"\textbf{" + cell + "}"

                cells.append(cell)

            lines.append(" & ".join(cells) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════
#  ABLATION ANALYZER
# ══════════════════════════════════════════════════════════════

class AblationAnalyzer:
    """
    Generate ablation study tables from labeled experiments.

    Usage:
        ablation = AblationAnalyzer(baseline="experiments/exp_001_baseline")
        ablation.add_variant("experiments/exp_002_no_prior",    "w/o Prior")
        ablation.add_variant("experiments/exp_003_no_spectral", "w/o Spectral Optim.")
        ablation.add_variant("experiments/exp_004_no_gating",   "w/o Gating")
        ablation.add_variant("experiments/exp_005_full",        "Full Model (Ours)")
        ablation.export("results/ablation")
    """

    def __init__(self, baseline: str, baseline_label: str = "Baseline"):
        self._baseline_dir   = baseline
        self._baseline_label = baseline_label
        self._variants: List[Dict] = []

    def add_variant(self, exp_dir: str, label: str) -> None:
        """Add an ablation variant with a descriptive label."""
        self._variants.append({"dir": exp_dir, "label": label})

    def export(self, output_dir: str, metrics: Optional[List[str]] = None) -> None:
        """
        Generate ablation table with delta improvements vs baseline.

        Default metrics: ["Dice_WT", "Dice_TC", "Dice_ET", "HD95_WT"]
        """
        os.makedirs(output_dir, exist_ok=True)

        if metrics is None:
            metrics = ["Dice_WT", "Dice_TC", "Dice_ET", "HD95_WT"]

        # Load baseline metrics
        base_m = _load_json_safe(os.path.join(self._baseline_dir, "metrics.json"))
        base = self._extract_metrics(base_m)

        rows = [{"Variant": self._baseline_label, **{k: base.get(k, 0) for k in metrics}}]

        for var in self._variants:
            var_m = _load_json_safe(os.path.join(var["dir"], "metrics.json"))
            var_vals = self._extract_metrics(var_m)

            row = {"Variant": var["label"]}
            for k in metrics:
                val = var_vals.get(k, 0)
                delta = val - base.get(k, 0) if val and base.get(k) else 0

                # Format with delta
                if "HD95" in k:
                    # Lower is better for HD95
                    delta_str = f" ({delta:+.2f})" if delta != 0 else ""
                    row[k] = f"{val:.2f}{delta_str}" if val else "N/A"
                else:
                    delta_str = f" ({delta:+.4f})" if delta != 0 else ""
                    row[k] = f"{val:.4f}{delta_str}" if val else "N/A"

            rows.append(row)

        # Write ablation table
        ExperimentComparator._write_csv(rows, os.path.join(output_dir, "ablation_table.csv"))
        ExperimentComparator._write_markdown(rows, os.path.join(output_dir, "ablation_table.md"),
                                             title="Ablation Study")
        ExperimentComparator._write_latex(rows, os.path.join(output_dir, "ablation_table.tex"),
                                          caption="Ablation study on BraTS 2021 validation set.",
                                          label="tab:ablation")

        logger.info(f"[Ablation] Exported ablation tables to {output_dir}")

    @staticmethod
    def _extract_metrics(m: Dict) -> Dict:
        """Normalize metric keys to standard format."""
        return {
            "Dice_WT":   m.get("wt_dice", m.get("dice_wt", 0)),
            "Dice_TC":   m.get("tc_dice", m.get("dice_tc", 0)),
            "Dice_ET":   m.get("et_dice", m.get("dice_et", 0)),
            "Dice_Mean": m.get("mean_dice", m.get("dice_mean", 0)),
            "HD95_WT":   m.get("hd95_wt", m.get("wt_hd95", 0)),
            "HD95_TC":   m.get("hd95_tc", m.get("tc_hd95", 0)),
            "HD95_ET":   m.get("hd95_et", m.get("et_hd95", 0)),
        }
