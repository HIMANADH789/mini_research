"""
benchmark_utils.py
==================
Generates ready-to-use benchmark tables for research papers.

Creates all table types in CSV, JSON, and Markdown formats:

    experiment_dir/tables/
        benchmark_results.csv / .json / .md    — main model comparison
        ablation_results.csv  / .json / .md    — ablation study
        runtime_results.csv   / .json / .md    — speed / efficiency
        params_results.csv    / .json / .md    — parameter comparison
        memory_results.csv    / .json / .md    — GPU memory comparison

    experiment_dir/paper_assets/
        best_metrics.txt                       — plain-text summary (for writing)
        experiment_notes.md                    — auto-generated run notes
        appendix_assets_list.md                — list of appendix figures/tables
        tables/                                — copies of all tables above

Usage:
    from src.utils.benchmark_utils import BenchmarkExporter

    exporter = BenchmarkExporter(exp_path, config)
    exporter.export_run_results(
        metrics=evaluation_results,
        runtime_info=timing_dict,
        model_info=model_info_dict,
    )

    # Multi-run comparison:
    BenchmarkExporter.build_comparison_table(
        exp_dirs=["experiments/exp_001", "experiments/exp_002"],
        output_path="reports/benchmark_results.csv",
    )
"""

import os
import csv
import json
import logging
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _atomic_write(text: str, path: str) -> None:
    """Atomic text file write."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        os.close(fd)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _atomic_json(obj: Any, path: str) -> None:
    """Atomic JSON write."""
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


def _write_csv(rows: List[Dict], path: str, fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fieldnames is None:
        all_keys: List[str] = []
        for row in rows:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
        fieldnames = all_keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(
    rows: List[Dict],
    title: str,
    path: str,
    column_order: Optional[List[str]] = None,
    float_precision: int = 4,
) -> None:
    """Write a clean Markdown table from a list of dicts."""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Determine columns
    if column_order is None:
        all_keys: List[str] = []
        for row in rows:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
        col_order = all_keys
    else:
        col_order = [c for c in column_order if any(c in row for row in rows)]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.{float_precision}f}"
        return str(v) if v is not None else ""

    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| " + " | ".join(c for c in col_order) + " |",
        "|" + "|".join("---" for _ in col_order) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c, "")) for c in col_order) + " |")
    lines.append("")

    _atomic_write("\n".join(lines), path)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
#  METRIC DEFINITIONS
# ══════════════════════════════════════════════════════════════

BRATS_METRICS = [
    "dice_mean", "wt_dice", "tc_dice", "et_dice",
    "hd95",      "wt_hd95", "tc_hd95", "et_hd95",
]

METRIC_ALIAS = {
    "mean_dice": "dice_mean",
    "dice_wt":   "wt_dice",
    "dice_tc":   "tc_dice",
    "dice_et":   "et_dice",
    "hausdorff": "hd95",
}

# Common column ordering for benchmark tables
BENCH_COLS = ["experiment", "model_name", "seed", "timestamp"] + BRATS_METRICS + \
             ["total_params", "trainable_params", "total_training_h", "avg_epoch_s"]

RUNTIME_COLS = ["experiment", "model_name", "total_params_M",
                "total_training_h", "avg_epoch_s",
                "avg_inference_s", "gpu_memory_peak_gb", "gpu_name"]

PARAMS_COLS  = ["experiment", "model_name", "total_params", "trainable_params",
                "frozen_params", "parameter_memory_mb"]

MEMORY_COLS  = ["experiment", "model_name", "gpu_name",
                "gpu_memory_peak_gb", "gpu_memory_reserved_gb",
                "parameter_memory_mb"]


# ══════════════════════════════════════════════════════════════
#  BENCHMARK EXPORTER
# ══════════════════════════════════════════════════════════════

class BenchmarkExporter:
    """
    Exports benchmark tables and paper assets for a single experiment run.

    Generates CSV + JSON + Markdown versions of all tables.
    """

    def __init__(self, exp_path: str, config):
        self.exp_path     = exp_path
        self.config       = config
        self.tables_dir   = os.path.join(exp_path, "tables")
        self.paper_dir    = os.path.join(exp_path, "paper_assets")
        self.paper_tables = os.path.join(self.paper_dir, "tables")
        self.runs_dir     = os.path.join(exp_path, "runs")

        for d in [self.tables_dir, self.paper_dir, self.paper_tables, self.runs_dir]:
            os.makedirs(d, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # MAIN EXPORT
    # ─────────────────────────────────────────────────────────

    def export_run_results(
        self,
        metrics: Dict[str, Any],
        runtime_info: Optional[Dict] = None,
        model_info: Optional[Dict] = None,
        notes: str = "",
    ) -> None:
        """
        Export all tables and paper assets for this run.

        Generates:
            tables/benchmark_results.{csv,json,md}
            tables/runtime_results.{csv,json,md}
            tables/params_results.{csv,json,md}
            tables/memory_results.{csv,json,md}
            paper_assets/best_metrics.txt
            paper_assets/experiment_notes.md
            paper_assets/appendix_assets_list.md
            paper_assets/tables/ (copies of all tables)
        """
        bench_row   = self._build_benchmark_row(metrics, runtime_info, model_info)
        runtime_row = self._build_runtime_row(runtime_info, model_info)
        params_row  = self._build_params_row(model_info)
        memory_row  = self._build_memory_row(runtime_info, model_info)

        # ── Benchmark table ───────────────────────────────────────
        self._export_table(
            name="benchmark_results",
            rows=[bench_row],
            title=f"Benchmark Results — {os.path.basename(self.exp_path)}",
            col_order=BENCH_COLS,
        )

        # ── Runtime table ─────────────────────────────────────────
        if runtime_row:
            self._export_table(
                name="runtime_results",
                rows=[runtime_row],
                title=f"Runtime Results — {os.path.basename(self.exp_path)}",
                col_order=RUNTIME_COLS,
            )

        # ── Params table ──────────────────────────────────────────
        if params_row:
            self._export_table(
                name="params_results",
                rows=[params_row],
                title=f"Parameter Summary — {os.path.basename(self.exp_path)}",
                col_order=PARAMS_COLS,
            )

        # ── Memory table ──────────────────────────────────────────
        if memory_row:
            self._export_table(
                name="memory_results",
                rows=[memory_row],
                title=f"Memory Results — {os.path.basename(self.exp_path)}",
                col_order=MEMORY_COLS,
            )

        # ── Paper assets ─────────────────────────────────────────
        self._write_best_metrics_txt(metrics, model_info)
        self._write_experiment_notes_md(metrics, runtime_info, model_info, notes)
        self._write_appendix_assets_list()

        logger.info(f"[Benchmark] Tables + paper assets → {self.tables_dir}/")

    def _export_table(
        self,
        name: str,
        rows: List[Dict],
        title: str,
        col_order: Optional[List[str]] = None,
    ) -> None:
        """Export a table in CSV + JSON + Markdown, both in tables/ and paper_assets/tables/."""
        base  = os.path.join(self.tables_dir, name)
        pbase = os.path.join(self.paper_tables, name)

        for path in [f"{base}.csv", f"{pbase}.csv"]:
            try:
                _write_csv(rows, path, fieldnames=col_order)
            except Exception as e:
                logger.warning(f"[Benchmark] CSV write failed for {name}: {e}")

        for path in [f"{base}.json", f"{pbase}.json"]:
            try:
                _atomic_json(
                    {"title": title, "generated_at": _now_iso(), "rows": rows},
                    path,
                )
            except Exception as e:
                logger.warning(f"[Benchmark] JSON write failed for {name}: {e}")

        for path in [f"{base}.md", f"{pbase}.md"]:
            try:
                _write_markdown_table(rows, title, path, column_order=col_order)
            except Exception as e:
                logger.warning(f"[Benchmark] Markdown write failed for {name}: {e}")

    # ─────────────────────────────────────────────────────────
    # ROW BUILDERS
    # ─────────────────────────────────────────────────────────

    def _normalize_metrics(self, metrics: Dict) -> Dict[str, Any]:
        out = {}
        for k, v in metrics.items():
            nk = METRIC_ALIAS.get(k.lower(), k.lower())
            out[nk] = round(v, 4) if isinstance(v, float) else v
        return out

    def _build_benchmark_row(
        self,
        metrics: Dict,
        runtime_info: Optional[Dict],
        model_info: Optional[Dict],
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "experiment":  os.path.basename(self.exp_path),
            "model_name":  _safe(lambda: self.config.name, "unknown"),
            "seed":        _safe(lambda: self.config.seed,  "N/A"),
            "timestamp":   _now_iso(),
        }
        row.update(self._normalize_metrics(metrics))

        if model_info:
            row["total_params"]     = model_info.get("total_params", "N/A")
            row["trainable_params"] = model_info.get("trainable_params", "N/A")

        if runtime_info:
            total_s = runtime_info.get("total_training_s")
            if total_s and isinstance(total_s, (int, float)):
                row["total_training_h"] = round(float(total_s) / 3600, 2)
            avg_ep = runtime_info.get("avg_epoch_duration_s")
            if avg_ep:
                row["avg_epoch_s"] = avg_ep
            row["gpu_name"] = runtime_info.get("gpu_name", "N/A")
            row["gpu_memory_peak_gb"] = runtime_info.get("gpu_memory_max_used_gb", "N/A")

        return row

    def _build_runtime_row(
        self,
        runtime_info: Optional[Dict],
        model_info: Optional[Dict],
    ) -> Optional[Dict[str, Any]]:
        if not runtime_info and not model_info:
            return None
        row: Dict[str, Any] = {
            "experiment": os.path.basename(self.exp_path),
            "model_name": _safe(lambda: self.config.name, "unknown"),
        }
        if model_info:
            total = model_info.get("total_params")
            if isinstance(total, int):
                row["total_params_M"] = round(total / 1e6, 2)
            row["gpu_name"] = model_info.get("gpu_name", "N/A")

        if runtime_info:
            total_s = runtime_info.get("total_training_s")
            if total_s:
                row["total_training_h"]  = round(float(total_s) / 3600, 2)
            row["avg_epoch_s"]           = runtime_info.get("avg_epoch_duration_s", "N/A")
            row["avg_inference_s"]       = runtime_info.get("avg_inference_time_s", "N/A")
            row["gpu_memory_peak_gb"]    = runtime_info.get("gpu_memory_max_used_gb", "N/A")
            row["avg_samples_per_sec"]   = runtime_info.get("avg_samples_per_sec", "N/A")

            # Fine-grained timing if available
            row["avg_dataloader_s"]      = runtime_info.get("avg_dataloader_time_s", "N/A")
            row["avg_forward_s"]         = runtime_info.get("avg_forward_pass_time_s", "N/A")
            row["avg_backward_s"]        = runtime_info.get("avg_backward_pass_time_s", "N/A")
            row["avg_optimizer_s"]       = runtime_info.get("avg_optimizer_step_s", "N/A")
        return row

    def _build_params_row(self, model_info: Optional[Dict]) -> Optional[Dict[str, Any]]:
        if not model_info:
            return None
        row: Dict[str, Any] = {
            "experiment":   os.path.basename(self.exp_path),
            "model_name":   _safe(lambda: self.config.name, "unknown"),
        }
        total = model_info.get("total_params", "N/A")
        if isinstance(total, int):
            row["total_params"] = total
            row["total_params_M"] = round(total / 1e6, 2)
        else:
            row["total_params"] = total

        row["trainable_params"]    = model_info.get("trainable_params", "N/A")
        row["frozen_params"]       = model_info.get("frozen_params", "N/A")
        row["parameter_memory_mb"] = model_info.get("parameter_memory_estimate_mb", "N/A")
        row["model_class"]         = model_info.get("model_class", "N/A")
        return row

    def _build_memory_row(
        self,
        runtime_info: Optional[Dict],
        model_info: Optional[Dict],
    ) -> Optional[Dict[str, Any]]:
        if not runtime_info and not model_info:
            return None
        row: Dict[str, Any] = {
            "experiment":   os.path.basename(self.exp_path),
            "model_name":   _safe(lambda: self.config.name, "unknown"),
        }
        if runtime_info:
            row["gpu_name"]               = runtime_info.get("gpu_name", "N/A")
            row["gpu_memory_peak_gb"]     = runtime_info.get("gpu_memory_max_used_gb", "N/A")
            row["gpu_memory_reserved_gb"] = runtime_info.get("gpu_memory_reserved_gb", "N/A")
        if model_info:
            row["parameter_memory_mb"]    = model_info.get("parameter_memory_estimate_mb", "N/A")
            row["total_params_M"]         = round(
                model_info.get("total_params", 0) / 1e6, 2
            ) if isinstance(model_info.get("total_params"), int) else "N/A"
        return row

    # ─────────────────────────────────────────────────────────
    # PAPER ASSETS
    # ─────────────────────────────────────────────────────────

    def _write_best_metrics_txt(
        self,
        metrics: Dict,
        model_info: Optional[Dict],
    ) -> None:
        """Write a plain-text metric summary for copy-paste into paper drafts."""
        normalized = self._normalize_metrics(metrics)
        lines = [
            f"# Best Metrics — {os.path.basename(self.exp_path)}",
            f"Generated: {_now_iso()}",
            "",
        ]
        if model_info:
            total = model_info.get("total_params", "N/A")
            if isinstance(total, int):
                total = f"{total:,}"
            mem = model_info.get("parameter_memory_estimate_mb", "N/A")
            lines += [
                f"Model:            {model_info.get('model_name', 'N/A')}",
                f"Total Parameters: {total}",
                f"Frozen Parameters: {model_info.get('frozen_params', 'N/A')}",
                f"Parameter Memory: {mem} MB",
                "",
            ]
        lines.append("=== Segmentation Metrics ===")
        for key in BRATS_METRICS:
            v = normalized.get(key)
            if v is not None:
                lines.append(f"  {key:<20}: {v:.4f}" if isinstance(v, float) else f"  {key:<20}: {v}")

        lines += ["", "=== All Metrics ==="]
        for k, v in normalized.items():
            if k not in BRATS_METRICS:
                lines.append(f"  {k:<20}: {v:.4f}" if isinstance(v, float) else f"  {k:<20}: {v}")

        path = os.path.join(self.paper_dir, "best_metrics.txt")
        try:
            _atomic_write("\n".join(lines) + "\n", path)
        except Exception as e:
            logger.warning(f"[Benchmark] Could not write best_metrics.txt: {e}")

    def _write_experiment_notes_md(
        self,
        metrics: Dict,
        runtime_info: Optional[Dict],
        model_info: Optional[Dict],
        notes: str = "",
    ) -> None:
        """Auto-generate experiment notes markdown."""
        exp_name  = os.path.basename(self.exp_path)
        model     = _safe(lambda: self.config.name, "unknown")
        seed      = _safe(lambda: self.config.seed,  "N/A")
        epochs    = _safe(lambda: self.config.training.epochs, "N/A")
        lr        = _safe(lambda: self.config.training.lr,     "N/A")
        opt_type  = _safe(lambda: getattr(self.config.optimizer, "type", "N/A"), "N/A")
        normalized = self._normalize_metrics(metrics)

        md = [
            f"# Experiment Notes: {exp_name}",
            "",
            f"**Generated:** {_now_iso()}  ",
            f"**Model:** {model}  ",
            f"**Seed:** {seed}  ",
            "",
            "---",
            "",
            "## Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Model     | {model} |",
            f"| Seed      | {seed}  |",
            f"| Epochs    | {epochs} |",
            f"| LR        | {lr} |",
            f"| Optimizer | {opt_type} |",
        ]
        if model_info:
            total = model_info.get("total_params", "N/A")
            if isinstance(total, int):
                total = f"{total:,}"
            md.append(f"| Parameters | {total} |")
            mem = model_info.get("parameter_memory_estimate_mb", "N/A")
            md.append(f"| Param Memory | {mem} MB |")

        md += ["", "---", "", "## Results", "", "| Metric | Value |", "|--------|-------|"]
        for key in BRATS_METRICS:
            v = normalized.get(key)
            if v is not None:
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                md.append(f"| {key:<12} | {val_str} |")

        if runtime_info:
            md += [
                "", "---", "", "## Runtime", "",
                "| Stat | Value |", "|------|-------|",
                f"| Total Training Time | {runtime_info.get('total_training_human', 'N/A')} |",
                f"| Avg Epoch | {runtime_info.get('avg_epoch_duration_s', 'N/A')} s |",
                f"| Peak GPU Memory | {runtime_info.get('gpu_memory_max_used_gb', 'N/A')} GB |",
                f"| Samples/sec | {runtime_info.get('avg_samples_per_sec', 'N/A')} |",
            ]

        if notes:
            md += ["", "---", "", "## Notes", "", notes]

        md += [
            "", "---", "", "## Reproducibility", "",
            "```bash",
            f"python scripts/run_experiment.py --config configs/[your_config].yaml",
            "```",
            "",
            f"Config hash: see `reproducibility.json`  ",
            f"Git commit:  see `system_info.json`  ",
            "",
        ]

        path = os.path.join(self.paper_dir, "experiment_notes.md")
        try:
            _atomic_write("\n".join(md) + "\n", path)
        except Exception as e:
            logger.warning(f"[Benchmark] Could not write experiment_notes.md: {e}")

    def _write_appendix_assets_list(self) -> None:
        """Generate a list of all appendix figures and tables."""
        exp_name = os.path.basename(self.exp_path)

        lines = [
            f"# Appendix Assets List — {exp_name}",
            "",
            f"**Generated:** {_now_iso()}",
            "",
            "Complete list of all generated files available for appendix inclusion.",
            "",
            "## Training Curves (curves/plot_data/)",
            "",
            "| File | Description | Format |",
            "|------|-------------|--------|",
            "| loss_curve_data.json | Raw train/val loss series | JSON |",
            "| dice_curve_data.json | All Dice score series | JSON |",
            "| lr_schedule_data.json | Learning rate schedule | JSON |",
            "| grad_norm_curve_data.json | Gradient norm series | JSON |",
            "| gpu_memory_curve_data.json | GPU memory timeline | JSON |",
            "| epoch_duration_curve_data.json | Epoch duration series | JSON |",
            "| throughput_curve_data.json | Training throughput | JSON |",
            "",
            "## Generated Plots (plots/ and paper_assets/plots/)",
            "",
            "| File | Description | Format |",
            "|------|-------------|--------|",
            "| plots/loss_curve.png | Loss curve (dark style) | PNG |",
            "| plots/dice_curve.png | Dice curves (dark style) | PNG |",
            "| plots/dashboard.png | Overview dashboard | PNG |",
            "| paper_assets/plots/paper_loss_curve.png | Loss curve (paper style) | PNG 300dpi |",
            "| paper_assets/plots/paper_dice_curve.png | Dice curve (paper style) | PNG 300dpi |",
            "| paper_assets/plots/paper_loss_curve.svg | Loss curve (vector) | SVG |",
            "| paper_assets/plots/paper_dice_curve.svg | Dice curve (vector) | SVG |",
            "",
            "## Tables (tables/)",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| benchmark_results.csv/.json/.md | Main metrics comparison |",
            "| runtime_results.csv/.json/.md | Speed and efficiency |",
            "| params_results.csv/.json/.md | Parameter counts |",
            "| memory_results.csv/.json/.md | GPU memory usage |",
            "",
            "## Diagnostics (diagnostics/)",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| optimization_diagnostics.json | Gradient stats, update norms |",
            "| generalization_diagnostics.json | Train-val gap analysis |",
            "| stability_diagnostics.json | NaN incidents, clip events |",
            "| calibration_diagnostics.json | ECE, confidence histograms |",
            "| diagnostics_summary.json | Summary of all diagnostics |",
            "",
            "## Research Artifacts (artifacts/)",
            "",
            "| Directory | Contents |",
            "|-----------|----------|",
            "| artifacts/graph/ | Adjacency matrix snapshots (.npz) |",
            "| artifacts/spectral/ | Eigenvalue arrays (.npy) |",
            "| artifacts/evidential/ | Uncertainty histograms (.npz) |",
            "| artifacts/attention/ | Attention heatmaps (.npy) |",
            "",
            "## Paper Templates (paper_assets/)",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| best_metrics.txt | Copy-paste metrics for paper |",
            "| experiment_notes.md | Auto-generated run summary |",
            "| methodology_notes_template.md | Methodology section template |",
            "| limitations_notes_template.md | Limitations section template |",
            "| reproducibility_checklist.md | NeurIPS/ICML checklist |",
            "| compute_budget_summary.md | Compute cost breakdown |",
            "",
        ]

        path = os.path.join(self.paper_dir, "appendix_assets_list.md")
        try:
            _atomic_write("\n".join(lines), path)
        except Exception as e:
            logger.warning(f"[Benchmark] Could not write appendix_assets_list.md: {e}")

    # ─────────────────────────────────────────────────────────
    # MULTI-RUN COMPARISON (static)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_comparison_table(
        exp_dirs: List[str],
        output_path: str = "reports/benchmark_results.csv",
        primary_metric: str = "dice_mean",
    ) -> str:
        """
        Build a ranked multi-run comparison table from a list of experiment directories.
        Reads evaluation_results.json from each exp_dir.
        Outputs ranked CSV + JSON + Markdown sorted by primary_metric.
        """
        rows = []
        for exp_dir in exp_dirs:
            # Try multiple locations for evaluation results
            for eval_name in ["evaluation_results.json", "metrics.json"]:
                for subdir in ["outputs", "runs", ""]:
                    if subdir:
                        eval_path = os.path.join(exp_dir, subdir, eval_name)
                    else:
                        eval_path = os.path.join(exp_dir, eval_name)
                    if os.path.exists(eval_path):
                        break
                else:
                    continue
                break
            else:
                continue

            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception:
                continue

            row: Dict[str, Any] = {"experiment": os.path.basename(exp_dir)}

            # Config name
            cfg_path = os.path.join(exp_dir, "config.yaml")
            if os.path.exists(cfg_path):
                try:
                    import yaml
                    with open(cfg_path, "r") as f:
                        cfg = yaml.safe_load(f)
                    row["model_name"] = cfg.get("name", "unknown")
                    row["seed"]       = cfg.get("seed", "N/A")
                except Exception:
                    pass

            # Metrics
            for k, v in raw.items():
                nk = METRIC_ALIAS.get(k.lower(), k.lower())
                row[nk] = round(v, 4) if isinstance(v, float) else v

            # Runtime from timing.json
            timing_path = os.path.join(exp_dir, "runs", "timing.json")
            if os.path.exists(timing_path):
                try:
                    with open(timing_path, "r") as f:
                        timing = json.load(f)
                    total_s = timing.get("total_training_s")
                    if total_s:
                        row["total_training_h"] = round(float(total_s) / 3600, 2)
                    row["avg_epoch_s"] = timing.get("avg_epoch_duration_s", "N/A")
                except Exception:
                    pass

            rows.append(row)

        # Sort by primary metric
        rows.sort(
            key=lambda r: float(r.get(primary_metric, -1)) if isinstance(r.get(primary_metric), (int, float)) else -1,
            reverse=True,
        )
        for i, row in enumerate(rows):
            row["rank"] = i + 1

        base = os.path.splitext(output_path)[0]
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        _write_csv(rows, output_path, fieldnames=["rank"] + BENCH_COLS)
        _atomic_json(
            {"primary_metric": primary_metric, "rows": rows},
            base + ".json",
        )
        _write_markdown_table(
            rows,
            f"Benchmark Comparison — ranked by {primary_metric}",
            base + ".md",
            column_order=["rank"] + BENCH_COLS,
        )

        logger.info(
            f"[Benchmark] Comparison table → {output_path} ({len(rows)} experiments)"
        )
        return output_path


# ══════════════════════════════════════════════════════════════
#  ABLATION TABLE (standalone helper)
# ══════════════════════════════════════════════════════════════

def build_ablation_table(
    entries: List[Dict[str, Any]],
    output_path: str,
    primary_metric: str = "dice_mean",
) -> str:
    """
    Build an ablation study table in CSV + JSON + Markdown.

    Parameters
    ----------
    entries     : list of dicts, minimum keys: {"variant": str, metric_keys...}
    output_path : base path (without extension); .csv / .json / .md will be added

    Example:
        entries = [
            {"variant": "Full Model",       "dice_mean": 0.842, "tc_dice": 0.831},
            {"variant": "- Spectral Opt",   "dice_mean": 0.827, "tc_dice": 0.814},
            {"variant": "- Structure Prior","dice_mean": 0.819, "tc_dice": 0.801},
            {"variant": "Baseline",         "dice_mean": 0.808, "tc_dice": 0.793},
        ]
        build_ablation_table(entries, "reports/ablation_results")
    """
    if not entries:
        return output_path

    # Sort by primary metric descending
    entries = sorted(entries,
                     key=lambda r: float(r.get(primary_metric, 0)), reverse=True)

    # Add rank and delta columns
    best_val = float(entries[0].get(primary_metric, 0))
    for i, entry in enumerate(entries):
        entry["rank"] = i + 1
        d = float(entry.get(primary_metric, 0))
        entry[f"delta_{primary_metric}"] = round(d - best_val, 4)

    base = output_path.rstrip(".csv").rstrip(".json").rstrip(".md")
    if base.endswith("."):
        base = base[:-1]

    csv_path  = base + ".csv"
    json_path = base + ".json"
    md_path   = base + ".md"

    _write_csv(entries, csv_path)
    _atomic_json(
        {"primary_metric": primary_metric, "rows": entries},
        json_path,
    )
    _write_markdown_table(
        entries,
        f"Ablation Study — ranked by {primary_metric}",
        md_path,
    )

    logger.info(f"[Benchmark] Ablation table → {csv_path}")
    return csv_path


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
