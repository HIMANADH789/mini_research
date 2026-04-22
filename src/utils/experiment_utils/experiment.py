"""
experiment.py  (experiment_utils)
==================================
Creates a new self-contained experiment directory with ALL
subdirectories required by the publication-grade evidence system.

Folder structure created:

    experiments/
     └── exp_001_swinunetr/          ← exp_path
          config.yaml
          notes.md                   ← placeholder (editor fills this)
          checkpoints/
          logs/
          outputs/
          runs/                      ← metrics, timing, failure JSON
          curves/                    ← raw plot data JSON (for on-demand rendering)
              plot_data/
          plots/                     ← rendered PNG plots
          tables/                    ← CSV + JSON + Markdown tables
          artifacts/                 ← tensor snapshots (.npy, .npz)
              graph/
              spectral/
              evidential/
              attention/
          diagnostics/               ← scientific diagnostics JSON
          paper_assets/              ← paper-ready assets
              plots/
              tables/
          raw_exports/               ← raw data exports (e.g. full predictions)
"""

import os
from datetime import datetime


def create_experiment(config, suffix: str = "") -> str:
    """
    Create a new experiment directory with all required subdirectories.

    Parameters
    ----------
    config : Config
        Parsed config object. Must have a `name` attribute.
    suffix : str
        Optional suffix appended to the folder name.
        Used by run_search.py to tag trial folders (e.g. "_trial_003").

    Returns
    -------
    str : absolute path to the new experiment directory.
    """
    base_dir = "experiments"
    os.makedirs(base_dir, exist_ok=True)

    exp_id   = len(os.listdir(base_dir)) + 1
    name_tag = str(getattr(config, "name", "exp"))
    exp_name = f"exp_{exp_id:03d}_{name_tag}{suffix}"
    exp_path = os.path.join(base_dir, exp_name)

    # Safety: if name collides (rare but possible with concurrent runs),
    # append a timestamp fragment.
    if os.path.exists(exp_path):
        ts = datetime.now().strftime("%H%M%S")
        exp_path = exp_path + f"_{ts}"

    # ── Create all directories ───────────────────────────────
    subdirs = [
        # Core
        "checkpoints",
        "logs",
        "outputs",
        # Evidence system
        "runs",
        "curves",
        "curves/plot_data",
        "plots",
        "tables",
        # Artifact storage (tensor snapshots)
        "artifacts",
        "artifacts/graph",
        "artifacts/spectral",
        "artifacts/evidential",
        "artifacts/attention",
        # Scientific diagnostics
        "diagnostics",
        # Paper assets
        "paper_assets",
        "paper_assets/plots",
        "paper_assets/tables",
        # Raw exports (full-volume predictions, embeddings, etc.)
        "raw_exports",
    ]

    os.makedirs(exp_path)
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_path, subdir), exist_ok=True)

    # ── Create placeholder notes.md ─────────────────────────
    _write_notes_placeholder(exp_path, config, exp_id)

    return exp_path


def _write_notes_placeholder(exp_path: str, config, exp_id: int) -> None:
    """Write an editable notes.md placeholder at experiment root."""
    path = os.path.join(exp_path, "notes.md")
    if os.path.exists(path):
        return  # Never overwrite researcher's notes

    model  = getattr(config, "name",  "unknown")
    seed   = getattr(config, "seed",  "N/A")
    epochs = getattr(getattr(config, "training", object()), "epochs", "N/A")
    lr     = getattr(getattr(config, "training", object()), "lr",     "N/A")
    exp_name = os.path.basename(exp_path)

    content = f"""# Experiment Notes: {exp_name}

**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Model:** {model} | **Seed:** {seed} | **Epochs:** {epochs} | **LR:** {lr}

---

## Hypothesis / Goal

<!-- What are you testing in this experiment? -->


## Key Changes from Previous Run

<!-- What is different from the baseline? -->


## Observations

<!-- Notes during training: loss behavior, GPU utilization, anomalies -->


## Results Summary

<!-- Fill after evaluation completes -->


## Next Steps

<!-- Based on these results, what to try next? -->


## Issues / Anomalies

<!-- NaN incidents, OOM errors, unexpected behavior -->
"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass