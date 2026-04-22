"""
repro_utils.py
==============
Reproducibility package builder for the mini_research pipeline.

Saves everything needed to reproduce an experiment from scratch:

    experiment_dir/
        reproducibility.json   — full reproducibility manifest
        notes.md               — experiment notes (editable by researcher)
        paper_assets/
            reproducibility_checklist.md   — NeurIPS/ICML-style checklist
            methodology_notes_template.md  — paper section template
            limitations_notes_template.md  — paper section template
            compute_budget_summary.md      — compute cost breakdown

Usage:
    from src.utils.repro_utils import ReproducibilityBuilder

    repro = ReproducibilityBuilder(exp_path, config)
    repro.build()  # saves reproduciblity.json + paper templates
    repro.update_dataset_hashes(data_root)  # call after data loading
"""

import os
import sys
import json
import time
import hashlib
import logging
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.system_utils import (
    collect_system_info,
    collect_git_info,
    collect_environment_info,
    config_hash,
    atomic_json_write,
    format_duration,
)

logger = logging.getLogger(__name__)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
#  REPRODUCIBILITY BUILDER
# ══════════════════════════════════════════════════════════════

class ReproducibilityBuilder:
    """
    Builds and saves the reproducibility package for one experiment.

    Call order:
        repro = ReproducibilityBuilder(exp_path, config)
        repro.build()                          # at experiment start
        repro.update_dataset_hashes(data_root) # after data loading (optional)
        repro.mark_complete(system_info)       # at experiment end
    """

    def __init__(self, exp_path: str, config):
        self.exp_path   = exp_path
        self.config     = config
        self.paper_dir  = os.path.join(exp_path, "paper_assets")
        self._data: Dict[str, Any] = {}

        os.makedirs(self.paper_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # BUILD
    # ─────────────────────────────────────────────────────────

    def build(self) -> None:
        """
        Collect and save the complete reproducibility manifest.
        Safe to call at training start — all collection is guarded.
        """
        self._data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "exp_path":     self.exp_path,
            "experiment_name": _safe(lambda: self.config.name, "unknown"),

            # ── Config ────────────────────────────────────────
            "config_hash":        _safe(lambda: config_hash(self.config), "unavailable"),
            "seed":               _safe(lambda: self.config.seed,  "N/A"),
            "config_snapshot":    _safe(self._dump_config, {}),

            # ── Code provenance ──────────────────────────────
            "git":                _safe(collect_git_info, {}),

            # ── Environment ──────────────────────────────────
            "environment":        _safe(collect_environment_info, {}),

            # ── Random seeds ─────────────────────────────────
            "random_seeds":       _safe(self._collect_seeds, {}),

            # ── Dataset hashes (populated later) ────────────
            "dataset_hashes":     {},

            # ── Status ───────────────────────────────────────
            "status":             "running",
            "completed_at":       None,
        }

        self._save()
        self._write_notes_md()
        self._write_paper_templates()
        logger.info(f"[Repro] Reproducibility package saved → {self.exp_path}/")

    def update_dataset_hashes(self, data_root: Optional[str] = None) -> None:
        """
        Hash the dataset directory to detect data changes between runs.
        Uses MD5 of all file sizes + names (not full content — fast).
        """
        if data_root is None:
            data_root = _safe(
                lambda: getattr(self.config.data, "train_root", None), None
            )
        if not data_root or not os.path.exists(str(data_root)):
            return

        try:
            hashes = _hash_directory(str(data_root))
            self._data["dataset_hashes"] = hashes
            self._save()
            logger.info(f"[Repro] Dataset hashes updated for {data_root}")
        except Exception as e:
            logger.debug(f"[Repro] Dataset hashing failed: {e}")

    def mark_complete(self, system_info: Optional[Dict] = None) -> None:
        """Update status to completed with final system info snapshot."""
        self._data["status"]       = "completed"
        self._data["completed_at"] = datetime.now(timezone.utc).isoformat()
        if system_info:
            self._data["final_system_snapshot"] = system_info
        self._save()

    # ─────────────────────────────────────────────────────────
    # PAPER ASSET TEMPLATES
    # ─────────────────────────────────────────────────────────

    def _write_notes_md(self) -> None:
        """Create editable notes.md at experiment root."""
        path = os.path.join(self.exp_path, "notes.md")
        if os.path.exists(path):
            return  # Don't overwrite researcher's notes

        exp_name = os.path.basename(self.exp_path)
        config   = self.config
        seed     = _safe(lambda: config.seed, "N/A")
        model    = _safe(lambda: config.name, "unknown")
        epochs   = _safe(lambda: config.training.epochs, "N/A")
        lr       = _safe(lambda: config.training.lr, "N/A")

        content = f"""# Experiment Notes: {exp_name}

**Created:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
**Model:** {model} | **Seed:** {seed} | **Epochs:** {epochs} | **LR:** {lr}

---

## Hypothesis / Goal

<!-- What are you testing in this experiment? -->


## Key Changes from Previous Run

<!-- What is different from the baseline? -->


## Observations During Training

<!-- What did you notice while the experiment ran? (loss behavior, GPU utilization, etc.) -->


## Results Summary

<!-- Fill after evaluation completes -->


## Next Steps

<!-- Based on these results, what should be tried next? -->


## Issues / Anomalies

<!-- Any NaN incidents, OOM errors, unexpected behavior? -->

"""
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.debug(f"[Repro] Could not write notes.md: {e}")

    def _write_paper_templates(self) -> None:
        """Write paper asset templates to paper_assets/."""
        self._write_methodology_template()
        self._write_limitations_template()
        self._write_reproducibility_checklist()
        self._write_compute_budget_template()

    def _write_methodology_template(self) -> None:
        model  = _safe(lambda: self.config.name, "MODEL_NAME")
        epochs = _safe(lambda: self.config.training.epochs, "N")
        lr     = _safe(lambda: self.config.training.lr, "LR")
        opt    = _safe(lambda: self.config.optimizer.type, "OPTIMIZER")

        tmpl = f"""# Methodology Notes Template

<!-- FILL THIS IN FOR YOUR PAPER'S METHODOLOGY SECTION -->

## Model Architecture

We propose {model}, a [describe architecture briefly].
The model consists of [describe main components].

## Training Details

We trained for {epochs} epochs using {opt} with an initial
learning rate of {lr}. [Describe LR schedule].
All experiments used a fixed random seed for reproducibility.

## Loss Function

[Describe the loss function used].

## Data Augmentation

[Describe augmentation pipeline].

## Evaluation Protocol

[Describe evaluation setup: patch size, overlap, TTA, etc.].

---
*Auto-generated template — edit before including in paper.*
"""
        path = os.path.join(self.paper_dir, "methodology_notes_template.md")
        _write_text_safe(tmpl, path)

    def _write_limitations_template(self) -> None:
        tmpl = """# Limitations Notes Template

<!-- FILL THIS IN FOR YOUR PAPER'S LIMITATIONS SECTION -->

## Computational Cost

Training requires [N] GPU-hours on [GPU model].
This may limit accessibility for researchers without GPU resources.

## Dataset Scope

Our experiments are conducted on [DATASET].
Performance on other datasets may differ due to domain shift.

## Architectural Assumptions

[Describe any assumptions that may limit generalizability].

## Negative Transfer / Failure Cases

[Describe cases where the model performs poorly or unexpectedly].

## Future Work

- [ ] Evaluate on additional benchmarks
- [ ] Extend to [other modalities / tasks]
- [ ] Reduce computational cost through [distillation / pruning]

---
*Auto-generated template — edit before including in paper.*
"""
        path = os.path.join(self.paper_dir, "limitations_notes_template.md")
        _write_text_safe(tmpl, path)

    def _write_reproducibility_checklist(self) -> None:
        exp_name    = os.path.basename(self.exp_path)
        config_hash_val = self._data.get("config_hash", "N/A")
        git_commit  = self._data.get("git", {}).get("git_commit", "N/A")
        seed        = _safe(lambda: self.config.seed, "N/A")

        checklist = f"""# Reproducibility Checklist — {exp_name}

**Auto-generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
**Config Hash:** `{config_hash_val}`
**Git Commit:** `{git_commit}`
**Seed:** `{seed}`

---

## NeurIPS / ICML-Style Checklist

### Code and Data
- [ ] Code is publicly available or will be released upon acceptance
- [ ] Dataset is publicly available (or access method is documented)
- [ ] Data preprocessing script is included
- [ ] `config.yaml` for this experiment is included
- [ ] `reproducibility.json` (this package) is included

### Training
- [ ] Random seed is fixed and documented (seed = {seed})
- [ ] Hyperparameters are fully reported in the paper
- [ ] Hardware specs are documented (`system_info.json`)
- [ ] Number of runs and variance are reported
- [ ] Training time is reported (`runtime.json`)

### Evaluation
- [ ] Evaluation protocol is fully described
- [ ] Test-time augmentation (if any) is documented
- [ ] Evaluation metrics are defined precisely
- [ ] Statistical significance is tested (see `statistics_summary.json`)

### Model
- [ ] Model architecture is fully described
- [ ] Parameter count is reported (`model_info.json`)
- [ ] Best model checkpoint is preserved
- [ ] Model class path: documented in `model_info.json`

### Results
- [ ] Mean ± std reported across multiple runs
- [ ] Best run identified and checkpoint preserved
- [ ] Ablation results are reproducible

---

## Files in This Package

| File | Contents |
|------|----------|
| `config.yaml` | Exact experiment config |
| `reproducibility.json` | Full environment snapshot |
| `system_info.json` | Hardware/software environment |
| `model_info.json` | Architecture + parameter details |
| `runtime.json` | Training time breakdown |
| `training_curves.csv` | Per-epoch training metrics |
| `best_metrics.txt` | Best evaluation results |

---
*Fill in all checkboxes before paper submission.*
"""
        path = os.path.join(self.paper_dir, "reproducibility_checklist.md")
        _write_text_safe(checklist, path)

    def _write_compute_budget_template(self) -> None:
        model  = _safe(lambda: self.config.name, "MODEL_NAME")
        epochs = _safe(lambda: self.config.training.epochs, "N")

        tmpl = f"""# Compute Budget Summary — {model}

**Model:** {model}
**Epochs:** {epochs}
**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

---

## Hardware Used

| Resource | Details |
|----------|---------|
| GPU | *(see system_info.json → gpu_name)* |
| GPU Memory | *(see system_info.json → gpu_memory_total_gb)* |
| CPU | *(see system_info.json → cpu_name)* |
| RAM | *(see system_info.json → ram_total_gb)* |

## Training Time

| Stage | Time |
|-------|------|
| Total training | *(see runtime.json → total_duration_human)* |
| Avg per epoch | *(see runtime.json → avg_epoch_duration_s)* |
| Evaluation | *(see runtime.json → avg_validation_time_s)* |

## Computational Cost Estimate

Approximate GPU-hours: **[fill from runtime.json]**
Cloud equivalent cost: ~$[fill based on GPU-hour rate]

---
*Fill values from corresponding JSON files after training completes.*
"""
        path = os.path.join(self.paper_dir, "compute_budget_summary.md")
        _write_text_safe(tmpl, path)

    # ─────────────────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────────────────

    def _dump_config(self) -> Dict:
        """Serialize the config object to a dict for JSON storage."""
        try:
            import yaml
            cfg_str = yaml.dump(
                vars(self.config) if hasattr(self.config, "__dict__")
                else {"config": str(self.config)},
                default_flow_style=False, sort_keys=True
            )
            return yaml.safe_load(cfg_str) or {}
        except Exception:
            try:
                return {"raw": str(self.config)}
            except Exception:
                return {}

    def _collect_seeds(self) -> Dict[str, Any]:
        """Collect all random seeds in use."""
        seeds: Dict[str, Any] = {}
        seeds["config_seed"] = _safe(lambda: self.config.seed, "N/A")
        try:
            import random
            seeds["python_random_state_type"] = type(random.getstate()[0]).__name__
        except Exception:
            pass
        try:
            import numpy as np
            seeds["numpy_seed_info"] = "see rng_states in checkpoint"
        except Exception:
            pass
        try:
            import torch
            seeds["torch_seed_info"] = "see rng_states in checkpoint"
            seeds["cudnn_deterministic"] = _safe(
                lambda: torch.backends.cudnn.deterministic, "N/A"
            )
            seeds["cudnn_benchmark"]     = _safe(
                lambda: torch.backends.cudnn.benchmark, "N/A"
            )
        except Exception:
            pass
        return seeds

    def _save(self) -> None:
        path = os.path.join(self.exp_path, "reproducibility.json")
        atomic_json_write(self._data, path)


# ══════════════════════════════════════════════════════════════
#  DATASET HASHING
# ══════════════════════════════════════════════════════════════

def _hash_directory(directory: str, max_files: int = 5000) -> Dict[str, Any]:
    """
    Compute a lightweight hash of a directory for reproducibility.
    Hashes file names + sizes (not content) for speed.
    Works on large medical imaging datasets (100GB+) without reading files.
    """
    if not os.path.isdir(directory):
        return {"error": f"Directory not found: {directory}"}

    hasher = hashlib.md5()
    file_count = 0
    total_size = 0
    extensions: Dict[str, int] = {}

    for root, dirs, files in os.walk(directory):
        dirs.sort()  # deterministic traversal order
        for fname in sorted(files):
            if file_count >= max_files:
                break
            full_path = os.path.join(root, fname)
            try:
                size = os.path.getsize(full_path)
                total_size += size
                entry = f"{fname}:{size}"
                hasher.update(entry.encode())

                ext = os.path.splitext(fname)[1].lower()
                extensions[ext] = extensions.get(ext, 0) + 1
                file_count += 1
            except Exception:
                pass

    return {
        "directory":   directory,
        "hash_md5":    hasher.hexdigest(),
        "file_count":  file_count,
        "total_size_gb": round(total_size / 1e9, 3),
        "extensions":  extensions,
        "hashed_at":   datetime.now(timezone.utc).isoformat(),
        "note": "Hash based on file names + sizes (not content) for speed.",
    }


# ══════════════════════════════════════════════════════════════
#  TEXT WRITE HELPER
# ══════════════════════════════════════════════════════════════

def _write_text_safe(content: str, path: str) -> None:
    """Write text file atomically. Never raises."""
    try:
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
    except Exception as e:
        logger.warning(f"[Repro] Could not write {path}: {e}")
