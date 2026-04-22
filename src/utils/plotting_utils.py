"""
plotting_utils.py
=================
On-demand plot generation system for the mini_research pipeline.

DESIGN PHILOSOPHY — Raw Data First:
    All plot data is saved as structured JSON to the curves/ directory.
    Plots (PNG, SVG, PDF) are generated ON DEMAND from this raw data.
    This means:
      - No large image files stored during training
      - You can regenerate any plot at any time in any format
      - You can change styles/colors after training without re-running
      - Raw data persists forever; rendered images are disposable

Workflow:
    1. Training saves curves/training_curves.json  (done by CurveTracker)
    2. After training: plotter.save_plot_data(...)   saves individual series JSON
    3. Anytime later: plotter.generate_png(name)     renders PNG from saved data
                      plotter.generate_svg(name)     renders SVG from saved data
                      plotter.generate_all_png()     renders all saved plots as PNG
                      plotter.generate_all_svg()     renders all saved plots as SVG

Plot data is saved under: experiment_dir/curves/plot_data/
    loss_curve_data.json
    dice_curve_data.json
    lr_schedule_data.json
    grad_norm_curve_data.json
    gpu_memory_curve_data.json
    epoch_duration_curve_data.json
    dashboard_data.json
    spectral_artifacts_data.json
    graph_sparsity_data.json
    attention_artifacts_data.json
    throughput_curve_data.json
    param_norm_curve_data.json

On-demand renders go to:
    experiment_dir/plots/   (dark style PNG, standard quality)
    experiment_dir/paper_assets/plots/  (white style, 300 dpi PNG + SVG)

Usage:
    from src.utils.plotting_utils import PlotManager

    plotter = PlotManager(exp_path)
    plotter.save_all_plot_data(curve_json_path)   # save raw data (fast, no images)
    plotter.generate_all_png()                     # renders all plots as PNG
    plotter.generate_svg("loss_curve")             # renders specific plot as SVG
    plotter.generate_paper_plots()                 # renders publication-ready plots
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  MATPLOTLIB IMPORT (optional)
# ══════════════════════════════════════════════════════════════

def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend (no display needed)
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as e:
        logger.debug(f"[PlotManager] Skipped: {e}")
        return default


# ══════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════

def _load_curves(json_path: str) -> List[Dict]:
    """Load training_curves.json → list of epoch dicts."""
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("epochs", [])
    except Exception:
        return []


def _extract_series(rows: List[Dict], key: str) -> Tuple[List[int], List[float]]:
    """
    Extract (epochs, values) for a given metric key.
    Skips epochs where the key is None or missing.
    """
    epochs, values = [], []
    for row in rows:
        v = row.get(key)
        if v is not None:
            try:
                epochs.append(int(row["epoch"]))
                values.append(float(v))
            except (TypeError, ValueError):
                pass
    return epochs, values


def _atomic_json(obj: Any, path: str) -> None:
    import tempfile
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
#  PLOT MANAGER
# ══════════════════════════════════════════════════════════════

class PlotManager:
    """
    Raw-data-first plot manager.

    Core concept: save structured plot data (series, labels, metadata)
    to JSON. Generate rendered images (PNG/SVG/PDF) on demand from that data.
    This decouples data capture from rendering — you never lose plotting data,
    and can always regenerate in a different format or style.

    Parameters
    ----------
    exp_path : str
        Root experiment directory.
    """

    # ── Dark research style (for monitoring / exploratory plots) ──
    COLORS = {
        "train":    "#4C9BE8",
        "val":      "#F4A261",
        "tc":       "#E63946",
        "wt":       "#2A9D8F",
        "et":       "#E9C46A",
        "bg":       "#A8DADC",
        "lr":       "#8338EC",
        "grad":     "#FB5607",
        "gpu":      "#06D6A0",
        "time":     "#FFBE0B",
        "prior":    "#3A86FF",
        "spectral": "#FF006E",
        "attn":     "#8338EC",
        "throughput": "#00B4D8",
        "param_norm": "#CDDFA0",
    }

    DARK_STYLE = {
        "figure.facecolor":  "#1A1A2E",
        "axes.facecolor":    "#16213E",
        "axes.edgecolor":    "#444466",
        "axes.labelcolor":   "#DDDDEE",
        "xtick.color":       "#AAAACC",
        "ytick.color":       "#AAAACC",
        "text.color":        "#DDDDEE",
        "grid.color":        "#333355",
        "grid.alpha":        0.4,
        "legend.facecolor":  "#1A1A2E",
        "legend.edgecolor":  "#444466",
        "lines.linewidth":   2.0,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "font.family":       "DejaVu Sans",
    }

    # ── Paper style (white, publication-ready) ──
    PAPER_STYLE = {
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#333333",
        "axes.labelcolor":  "#111111",
        "xtick.color":      "#333333",
        "ytick.color":      "#333333",
        "text.color":       "#111111",
        "grid.color":       "#CCCCCC",
        "grid.alpha":       0.5,
        "lines.linewidth":  2.5,
        "axes.titlesize":   14,
        "axes.labelsize":   12,
        "font.family":      "DejaVu Sans",
    }

    def __init__(self, exp_path: str):
        self.exp_path      = exp_path
        self.plots_dir     = os.path.join(exp_path, "plots")
        self.paper_dir     = os.path.join(exp_path, "paper_assets", "plots")
        self.curves_dir    = os.path.join(exp_path, "curves")
        self.data_dir      = os.path.join(exp_path, "curves", "plot_data")

        for d in [self.plots_dir, self.paper_dir, self.curves_dir, self.data_dir]:
            os.makedirs(d, exist_ok=True)

        # Registry of saved plot data {plot_name: data_file_path}
        self._registry: Dict[str, str] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Re-discover existing plot data files on init."""
        if not os.path.exists(self.data_dir):
            return
        for fname in os.listdir(self.data_dir):
            if fname.endswith("_data.json"):
                name = fname[:-len("_data.json")]
                self._registry[name] = os.path.join(self.data_dir, fname)

    # ─────────────────────────────────────────────────────────
    # STEP 1: SAVE RAW PLOT DATA
    # ─────────────────────────────────────────────────────────

    def save_all_plot_data(self, curve_json_path: str) -> List[str]:
        """
        Extract all plot series from training_curves.json and save as
        individual structured JSON files under curves/plot_data/.

        This is FAST (no rendering) and should be called after training.
        Returns list of saved data file paths.
        """
        rows = _load_curves(curve_json_path)
        if not rows:
            logger.warning("[PlotManager] No curve data — plot data not saved.")
            return []

        saved = []
        for save_fn in [
            self._save_loss_data,
            self._save_dice_data,
            self._save_lr_data,
            self._save_grad_norm_data,
            self._save_gpu_memory_data,
            self._save_epoch_duration_data,
            self._save_throughput_data,
            self._save_param_norm_data,
            self._save_dashboard_data,
        ]:
            path = _safe(lambda f=save_fn, r=rows: f(r))
            if path:
                saved.append(path)

        logger.info(f"[PlotManager] Saved {len(saved)} plot data files → {self.data_dir}/")
        return saved

    def save_spectral_data(
        self,
        eigenvalues_per_epoch: List[Dict],
        condition_numbers: List[float],
    ) -> Optional[str]:
        """Save spectral artifact data for on-demand plotting."""
        data = {
            "plot_type":              "spectral_artifacts",
            "title":                  "Spectral Artifacts",
            "eigenvalues_per_epoch":  eigenvalues_per_epoch,
            "condition_numbers":      condition_numbers,
            "x_label":                "Epoch",
        }
        return self._save_plot_data("spectral_artifacts", data)

    def save_graph_sparsity_data(self, sparsity_per_epoch: List[Dict]) -> Optional[str]:
        """Save graph sparsity data for on-demand plotting."""
        data = {
            "plot_type":          "graph_sparsity",
            "title":              "Graph Sparsity over Training",
            "sparsity_per_epoch": sparsity_per_epoch,
            "x_label":            "Epoch",
        }
        return self._save_plot_data("graph_sparsity", data)

    def save_attention_data(self, attention_stats: List[Dict]) -> Optional[str]:
        """Save attention artifact data for on-demand plotting."""
        data = {
            "plot_type":       "attention_artifacts",
            "title":           "Attention Statistics over Training",
            "attention_stats": attention_stats,
            "x_label":         "Epoch",
        }
        return self._save_plot_data("attention_artifacts", data)

    # ─────────────────────────────────────────────────────────
    # STEP 2: GENERATE PLOTS ON DEMAND
    # ─────────────────────────────────────────────────────────

    def generate_all_png(self) -> List[str]:
        """
        Generate ALL saved plot data files as dark-style PNG images.
        Returns list of generated file paths.
        """
        plt = _try_import_matplotlib()
        if plt is None:
            logger.warning("[PlotManager] matplotlib not installed — cannot generate PNG.")
            return []

        saved = []
        for name, data_path in self._registry.items():
            path = _safe(lambda n=name, dp=data_path, p=plt: self._render_plot(n, dp, p, "png", "dark"))
            if path:
                saved.append(path)
        logger.info(f"[PlotManager] Generated {len(saved)} PNG plots → {self.plots_dir}/")
        return saved

    def generate_all_svg(self) -> List[str]:
        """
        Generate ALL saved plot data files as publication-ready SVG images.
        SVG is vector format — can be edited in Inkscape/Illustrator or embedded in LaTeX.
        Returns list of generated file paths.
        """
        plt = _try_import_matplotlib()
        if plt is None:
            logger.warning("[PlotManager] matplotlib not installed — cannot generate SVG.")
            return []

        saved = []
        for name, data_path in self._registry.items():
            path = _safe(lambda n=name, dp=data_path, p=plt: self._render_plot(n, dp, p, "svg", "paper"))
            if path:
                saved.append(path)
        logger.info(f"[PlotManager] Generated {len(saved)} SVG plots → {self.paper_dir}/")
        return saved

    def generate_png(self, plot_name: str) -> Optional[str]:
        """Generate a single plot as PNG. plot_name must match saved data."""
        plt = _try_import_matplotlib()
        if plt is None:
            return None
        data_path = self._registry.get(plot_name)
        if data_path is None:
            logger.warning(f"[PlotManager] No data for plot '{plot_name}'. "
                           f"Available: {list(self._registry.keys())}")
            return None
        return _safe(lambda: self._render_plot(plot_name, data_path, plt, "png", "dark"))

    def generate_svg(self, plot_name: str) -> Optional[str]:
        """
        Generate a single plot as SVG (vector, publication-ready).
        SVG files can be converted to PDF with Inkscape or embedded in LaTeX.
        plot_name must match saved data (e.g. 'loss_curve', 'dice_curve').
        """
        plt = _try_import_matplotlib()
        if plt is None:
            return None
        data_path = self._registry.get(plot_name)
        if data_path is None:
            logger.warning(f"[PlotManager] No data for SVG plot '{plot_name}'. "
                           f"Available: {list(self._registry.keys())}")
            return None
        return _safe(lambda: self._render_plot(plot_name, data_path, plt, "svg", "paper"))

    def generate_paper_plots(self) -> List[str]:
        """
        Generate publication-ready plots (white bg, 300 dpi PNG + SVG).
        Output goes to paper_assets/plots/.
        """
        plt = _try_import_matplotlib()
        if plt is None:
            return []

        target_plots = ["loss_curve", "dice_curve", "lr_schedule", "grad_norm_curve"]
        saved = []
        for name in target_plots:
            if name not in self._registry:
                continue
            data_path = self._registry[name]
            # PNG 300 dpi
            path_png = _safe(lambda n=name, dp=data_path, p=plt:
                             self._render_plot(n, dp, p, "png", "paper", dpi=300))
            if path_png:
                saved.append(path_png)
            # SVG (vector)
            path_svg = _safe(lambda n=name, dp=data_path, p=plt:
                             self._render_plot(n, dp, p, "svg", "paper"))
            if path_svg:
                saved.append(path_svg)

        logger.info(f"[PlotManager] Paper plots → {self.paper_dir}/")
        return saved

    # ─────────────────────────────────────────────────────────
    # BACKWARD COMPATIBILITY — old API still works
    # ─────────────────────────────────────────────────────────

    def plot_all(self, curve_json_path: str) -> List[str]:
        """
        Legacy method: save data + immediately generate all PNG plots.
        Equivalent to: save_all_plot_data() + generate_all_png()
        """
        self.save_all_plot_data(curve_json_path)
        return self.generate_all_png()

    def plot_paper_figures(self, curve_json_path: str) -> List[str]:
        """Legacy: save data + generate paper-quality figures."""
        self.save_all_plot_data(curve_json_path)
        return self.generate_paper_plots()

    def plot_spectral(
        self,
        eigenvalues_per_epoch: List[Dict],
        condition_numbers: List[float],
    ) -> Optional[str]:
        """Legacy: save spectral data + generate PNG."""
        self.save_spectral_data(eigenvalues_per_epoch, condition_numbers)
        return self.generate_png("spectral_artifacts")

    def plot_graph_sparsity(self, sparsity_per_epoch: List[Dict]) -> Optional[str]:
        self.save_graph_sparsity_data(sparsity_per_epoch)
        return self.generate_png("graph_sparsity")

    def plot_attention_summary(self, attention_stats: List[Dict]) -> Optional[str]:
        self.save_attention_data(attention_stats)
        return self.generate_png("attention_artifacts")

    def list_available_plots(self) -> List[str]:
        """Return names of all plots with saved data."""
        self._load_registry()
        return sorted(self._registry.keys())

    # ─────────────────────────────────────────────────────────
    # INTERNAL: SAVE PLOT DATA
    # ─────────────────────────────────────────────────────────

    def _save_plot_data(self, name: str, data: Dict) -> Optional[str]:
        """Save structured plot data JSON. Returns path."""
        path = os.path.join(self.data_dir, f"{name}_data.json")
        try:
            _atomic_json(data, path)
            self._registry[name] = path
            return path
        except Exception as e:
            logger.warning(f"[PlotManager] Could not save plot data '{name}': {e}")
            return None

    def _save_series_data(
        self,
        name: str,
        series: List[Dict],
        title: str,
        x_label: str = "Epoch",
        y_label: str = "",
        plot_type: str = "line",
        extra: Optional[Dict] = None,
    ) -> Optional[str]:
        data: Dict[str, Any] = {
            "plot_type": plot_type,
            "title":     title,
            "x_label":   x_label,
            "y_label":   y_label,
            "series":    series,
        }
        if extra:
            data.update(extra)
        return self._save_plot_data(name, data)

    def _save_loss_data(self, rows: List[Dict]) -> Optional[str]:
        series = []
        ep_tr, loss_tr = _extract_series(rows, "train_loss")
        if ep_tr:
            series.append({"label": "Train Loss", "color": self.COLORS["train"],
                           "linestyle": "-", "epochs": ep_tr, "values": loss_tr})
        ep_vl, loss_vl = _extract_series(rows, "val_loss")
        if ep_vl:
            series.append({"label": "Val Loss", "color": self.COLORS["val"],
                           "linestyle": "--", "epochs": ep_vl, "values": loss_vl})
        if not series:
            return None
        return self._save_series_data("loss_curve", series,
                                      "Training & Validation Loss", y_label="Loss")

    def _save_dice_data(self, rows: List[Dict]) -> Optional[str]:
        dice_keys = [
            ("val_dice",     "Val Mean Dice",  self.COLORS["val"],   "-"),
            ("val_tc_dice",  "Val TC Dice",    self.COLORS["tc"],    "--"),
            ("val_wt_dice",  "Val WT Dice",    self.COLORS["wt"],    "-."),
            ("val_et_dice",  "Val ET Dice",    self.COLORS["et"],    ":"),
            ("train_dice",   "Train Dice",     self.COLORS["train"], "-"),
        ]
        series = []
        for key, label, color, ls in dice_keys:
            ep, vals = _extract_series(rows, key)
            if ep:
                series.append({"label": label, "color": color, "linestyle": ls,
                               "epochs": ep, "values": vals})
        if not series:
            return None
        return self._save_series_data("dice_curve", series,
                                      "Dice Score Curves", y_label="Dice",
                                      extra={"y_min": 0})

    def _save_lr_data(self, rows: List[Dict]) -> Optional[str]:
        ep, vals = _extract_series(rows, "lr")
        if not ep:
            return None
        series = [{"label": "Learning Rate", "color": self.COLORS["lr"],
                   "linestyle": "-", "epochs": ep, "values": vals}]
        return self._save_series_data("lr_schedule", series,
                                      "Learning Rate Schedule",
                                      y_label="LR",
                                      extra={"y_log_scale": True})

    def _save_grad_norm_data(self, rows: List[Dict]) -> Optional[str]:
        ep, vals = _extract_series(rows, "grad_norm")
        if not ep:
            return None
        series = [{"label": "Gradient Norm", "color": self.COLORS["grad"],
                   "linestyle": "-", "epochs": ep, "values": vals}]
        return self._save_series_data("grad_norm_curve", series,
                                      "Gradient Norm over Training",
                                      y_label="Gradient Norm")

    def _save_gpu_memory_data(self, rows: List[Dict]) -> Optional[str]:
        ep, vals = _extract_series(rows, "gpu_mem_gb")
        if not ep:
            return None
        series = [{"label": "GPU Memory (GB)", "color": self.COLORS["gpu"],
                   "linestyle": "-", "fill": True, "epochs": ep, "values": vals}]
        return self._save_series_data("gpu_memory_curve", series,
                                      "GPU Memory Usage over Training",
                                      y_label="GPU Memory (GB)")

    def _save_epoch_duration_data(self, rows: List[Dict]) -> Optional[str]:
        ep, vals = _extract_series(rows, "epoch_duration_s")
        if not ep:
            return None
        series = [{"label": "Epoch Duration (s)", "color": self.COLORS["time"],
                   "linestyle": "-", "scatter": True, "epochs": ep, "values": vals}]
        return self._save_series_data("epoch_duration_curve", series,
                                      "Epoch Duration over Training",
                                      y_label="Duration (s)")

    def _save_throughput_data(self, rows: List[Dict]) -> Optional[str]:
        ep, vals = _extract_series(rows, "samples_per_sec")
        if not ep:
            return None
        series = [{"label": "Samples/sec", "color": self.COLORS["throughput"],
                   "linestyle": "-", "epochs": ep, "values": vals}]
        return self._save_series_data("throughput_curve", series,
                                      "Training Throughput",
                                      y_label="Samples / second")

    def _save_param_norm_data(self, rows: List[Dict]) -> Optional[str]:
        ep, vals = _extract_series(rows, "param_norm")
        if not ep:
            return None
        series = [{"label": "Parameter Norm", "color": self.COLORS["param_norm"],
                   "linestyle": "-", "epochs": ep, "values": vals}]
        return self._save_series_data("param_norm_curve", series,
                                      "Parameter Norm over Training",
                                      y_label="Parameter Norm")

    def _save_dashboard_data(self, rows: List[Dict]) -> Optional[str]:
        """Save a multi-panel dashboard data layout."""
        panels = []
        panel_specs = [
            ("Loss",          [("train_loss","Train",self.COLORS["train"],"-"),
                               ("val_loss","Val",self.COLORS["val"],"--")]),
            ("Dice Scores",   [("val_dice","Val Dice",self.COLORS["val"],"-"),
                               ("val_tc_dice","TC",self.COLORS["tc"],"--"),
                               ("val_wt_dice","WT",self.COLORS["wt"],"-.")]),
            ("Learning Rate", [("lr","LR",self.COLORS["lr"],"-")]),
            ("Grad Norm",     [("grad_norm","Grad Norm",self.COLORS["grad"],"-")]),
        ]
        for title, series_specs in panel_specs:
            panel_series = []
            for key, label, color, ls in series_specs:
                ep, vals = _extract_series(rows, key)
                if ep:
                    panel_series.append({"label": label, "color": color,
                                        "linestyle": ls, "epochs": ep, "values": vals})
            panels.append({"title": title, "series": panel_series})

        data = {
            "plot_type": "dashboard",
            "title":     "Training Dashboard",
            "panels":    panels,
        }
        return self._save_plot_data("dashboard", data)

    # ─────────────────────────────────────────────────────────
    # INTERNAL: RENDER FROM SAVED DATA
    # ─────────────────────────────────────────────────────────

    def _render_plot(
        self,
        name: str,
        data_path: str,
        plt,
        fmt: str = "png",
        style: str = "dark",
        dpi: int = 150,
    ) -> Optional[str]:
        """
        Render a plot from its data JSON.

        Parameters
        ----------
        name      : plot identifier (e.g. 'loss_curve')
        data_path : path to the _data.json file
        plt       : matplotlib.pyplot module
        fmt       : 'png' | 'svg' | 'pdf'
        style     : 'dark' | 'paper'
        dpi       : resolution (ignored for SVG)
        """
        if not os.path.exists(data_path):
            return None

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        plot_type = data.get("plot_type", "line")
        style_dict = self.PAPER_STYLE if style == "paper" else self.DARK_STYLE
        plt.rcParams.update(style_dict)

        if plot_type == "dashboard":
            return self._render_dashboard(name, data, plt, fmt, style, dpi)
        elif plot_type == "spectral_artifacts":
            return self._render_spectral(name, data, plt, fmt, style, dpi)
        elif plot_type == "graph_sparsity":
            return self._render_graph_sparsity(name, data, plt, fmt, style, dpi)
        elif plot_type == "attention_artifacts":
            return self._render_attention(name, data, plt, fmt, style, dpi)
        else:
            return self._render_line(name, data, plt, fmt, style, dpi)

    def _render_line(self, name, data, plt, fmt, style, dpi) -> Optional[str]:
        """Render a standard line/scatter plot."""
        series  = data.get("series", [])
        if not series:
            return None
        title   = data.get("title", name)
        x_label = data.get("x_label", "Epoch")
        y_label = data.get("y_label", "")
        y_log   = data.get("y_log_scale", False)
        y_min   = data.get("y_min", None)

        fig, ax = plt.subplots(figsize=(10, 5))
        bg = self.DARK_STYLE["figure.facecolor"] if style == "dark" else "white"
        fig.patch.set_facecolor(bg)

        for s in series:
            ep   = s.get("epochs", [])
            vals = s.get("values", [])
            if not ep:
                continue
            label = s.get("label", "")
            color = s.get("color", "#888888")
            ls    = s.get("linestyle", "-")
            fill  = s.get("fill", False)
            scatter = s.get("scatter", False)

            if scatter:
                ax.scatter(ep, vals, color=color, s=20, alpha=0.8, label=label)
                # Rolling mean
                if len(vals) >= 5:
                    window = 5
                    smoothed = [
                        sum(vals[max(0, i - window):i + 1]) / len(vals[max(0, i - window):i + 1])
                        for i in range(len(vals))
                    ]
                    ax.plot(ep, smoothed, color=color, linewidth=2)
            elif fill:
                ax.fill_between(ep, vals, alpha=0.35, color=color)
                ax.plot(ep, vals, color=color, linewidth=2, label=label, linestyle=ls)
            elif y_log:
                ax.semilogy(ep, vals, color=color, linewidth=2, label=label, linestyle=ls)
            else:
                ax.plot(ep, vals, color=color, linewidth=2, label=label, linestyle=ls)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        if len(series) > 1:
            ax.legend()
        ax.grid(True)

        return self._save_fig(plt, fig, name, fmt, style, dpi)

    def _render_dashboard(self, name, data, plt, fmt, style, dpi) -> Optional[str]:
        """Render a 2×2 dashboard."""
        panels = data.get("panels", [])
        if not panels:
            return None

        bg = self.DARK_STYLE["figure.facecolor"] if style == "dark" else "white"
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(data.get("title", "Dashboard"), fontsize=15)
        fig.patch.set_facecolor(bg)

        for i, panel in enumerate(panels[:4]):
            ax = axes[i // 2][i % 2]
            for s in panel.get("series", []):
                ep   = s.get("epochs", [])
                vals = s.get("values", [])
                if not ep:
                    continue
                ax.plot(ep, vals, color=s.get("color", "#888"),
                        label=s.get("label", ""), linestyle=s.get("linestyle", "-"))
            ax.set_title(panel.get("title", ""))
            ax.legend(fontsize=8)
            ax.grid(True)

        return self._save_fig(plt, fig, name, fmt, style, dpi)

    def _render_spectral(self, name, data, plt, fmt, style, dpi) -> Optional[str]:
        """Render spectral artifacts: eigenvalue timeline + condition number."""
        eig_data = data.get("eigenvalues_per_epoch", [])
        cond_nums = data.get("condition_numbers", [])
        if not eig_data and not cond_nums:
            return None

        bg = self.DARK_STYLE["figure.facecolor"] if style == "dark" else "white"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(bg)

        if eig_data:
            epochs   = [d["epoch"] for d in eig_data]
            all_eigs = [d.get("eigenvalues", []) for d in eig_data]
            max_k    = min(max((len(e) for e in all_eigs), default=0), 5)
            try:
                cmap = plt.get_cmap("plasma")
            except Exception:
                cmap = None
            for k in range(max_k):
                vals      = [e[k] if len(e) > k else None for e in all_eigs]
                valid_ep  = [ep for ep, v in zip(epochs, vals) if v is not None]
                valid_v   = [v for v in vals if v is not None]
                color = cmap(k / max(max_k - 1, 1)) if cmap else self.COLORS["spectral"]
                ax1.plot(valid_ep, valid_v, color=color, label=f"λ_{k+1}", linewidth=1.8)
            ax1.set_title("Eigenvalue Evolution (top-k)")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Eigenvalue")
            ax1.legend(fontsize=8)
            ax1.grid(True)

        if cond_nums:
            cond_ep = list(range(len(cond_nums)))
            ax2.semilogy(cond_ep, cond_nums, color=self.COLORS["spectral"], linewidth=2)
            ax2.set_title("Condition Number")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Condition Number (log)")
            ax2.grid(True)

        return self._save_fig(plt, fig, name, fmt, style, dpi)

    def _render_graph_sparsity(self, name, data, plt, fmt, style, dpi) -> Optional[str]:
        sparsity_data = data.get("sparsity_per_epoch", [])
        if not sparsity_data:
            return None

        bg = self.DARK_STYLE["figure.facecolor"] if style == "dark" else "white"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(bg)

        epochs   = [d["epoch"]               for d in sparsity_data]
        sparsity = [d.get("sparsity_ratio", 0) for d in sparsity_data]
        deg_mean = [d.get("mean_degree", 0)   for d in sparsity_data]
        deg_max  = [d.get("max_degree", 0)    for d in sparsity_data]

        ax1.plot(epochs, sparsity, color=self.COLORS["prior"], linewidth=2)
        ax1.set_title("Graph Sparsity Ratio")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Sparsity (fraction of zeros)")
        ax1.grid(True)

        ax2.plot(epochs, deg_mean, color=self.COLORS["wt"], label="Mean degree")
        ax2.fill_between(epochs, 0, deg_max, alpha=0.2, color=self.COLORS["prior"],
                         label="Max degree")
        ax2.set_title("Graph Degree Statistics")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Degree")
        ax2.legend()
        ax2.grid(True)

        return self._save_fig(plt, fig, name, fmt, style, dpi)

    def _render_attention(self, name, data, plt, fmt, style, dpi) -> Optional[str]:
        attn_stats = data.get("attention_stats", [])
        if not attn_stats:
            return None

        bg = self.DARK_STYLE["figure.facecolor"] if style == "dark" else "white"
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor(bg)

        epochs = [d["epoch"]                   for d in attn_stats]
        entrop = [d.get("mean_entropy", None)   for d in attn_stats]
        magnit = [d.get("mean_magnitude", None) for d in attn_stats]
        hstd   = [d.get("head_std", None)       for d in attn_stats]

        for ax, vals, title, color in [
            (axes[0], entrop, "Attention Entropy",   self.COLORS["attn"]),
            (axes[1], magnit, "Attention Magnitude", self.COLORS["val"]),
            (axes[2], hstd,   "Head Std Dev",        self.COLORS["grad"]),
        ]:
            valid_ep   = [e for e, v in zip(epochs, vals) if v is not None]
            valid_vals = [v for v in vals if v is not None]
            if valid_ep:
                ax.plot(valid_ep, valid_vals, color=color, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True)

        return self._save_fig(plt, fig, name, fmt, style, dpi)

    def _save_fig(
        self,
        plt,
        fig,
        name: str,
        fmt: str,
        style: str,
        dpi: int,
    ) -> Optional[str]:
        """Save figure to appropriate directory."""
        # Paper plots go to paper_assets/plots/; dark style to plots/
        if style == "paper":
            out_dir = self.paper_dir
            fname   = f"paper_{name}.{fmt}"
        else:
            out_dir = self.plots_dir
            fname   = f"{name}.{fmt}"

        path = os.path.join(out_dir, fname)
        try:
            plt.tight_layout()
            bg = fig.get_facecolor()
            if fmt == "svg":
                plt.savefig(path, format="svg", bbox_inches="tight")
            else:
                plt.savefig(path, dpi=dpi, bbox_inches="tight",
                            facecolor=bg, format=fmt)
            plt.close("all")
            return path
        except Exception as e:
            logger.warning(f"[PlotManager] Could not save {fname}: {e}")
            plt.close("all")
            return None
