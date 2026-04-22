"""
search_utils.py
===============
Flexible search space system for hyperparameter search.

Supports:
  - Grid search     (all combinations via itertools.product)
  - Random search   (reproducible sampling with seed)
  - Dot-path config overrides (e.g. "training.lr" = 1e-4)
  - Nested list params (e.g. prior.weights: [[0.4,0.4,0.2], [0.5,0.3,0.2]])
  - Per-trial full YAML config generation (for reproducibility)

Designed for future extension to:
  - Optuna: replace generate_*() with trial.suggest_*(); apply_overrides() stays the same
  - RL controller: same apply_overrides() interface, different config source

Usage:
    from src.utils.search_utils import SearchSpace, TrialIndex

    space = SearchSpace.from_yaml("configs/search_space_example.yaml")
    trials = space.generate_trials()          # list of Trial objects

    for trial in trials:
        print(trial.trial_id, trial.overrides)
        expanded_cfg = trial.build_config()   # full YAML-safe dict
"""

import os
import copy
import json
import yaml
import random
import hashlib
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  DOT-PATH CONFIG OVERRIDE
# ══════════════════════════════════════════════════════════════

def _set_dot_path(cfg: Dict, dot_path: str, value: Any) -> None:
    """
    Set a value in a nested dict using a dot-path key.

    Example:
        _set_dot_path(cfg, "training.lr", 1e-4)
        _set_dot_path(cfg, "model.rank_k", 64)
        _set_dot_path(cfg, "prior.weights", [0.4, 0.4, 0.2])
    """
    keys = dot_path.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _get_dot_path(cfg: Dict, dot_path: str, default=None) -> Any:
    """Get a value from a nested dict using a dot-path key."""
    keys = dot_path.split(".")
    d = cfg
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def apply_overrides(base_cfg: Dict, overrides: Dict[str, Any]) -> Dict:
    """
    Deep-copy base config and apply a flat dict of dot-path overrides.

    Parameters
    ----------
    base_cfg  : dict loaded from base YAML
    overrides : {dot_path: value} e.g. {"training.lr": 1e-4, "model.rank_k": 64}

    Returns
    -------
    New dict with overrides applied.
    """
    cfg = copy.deepcopy(base_cfg)
    for dot_path, value in overrides.items():
        _set_dot_path(cfg, dot_path, value)
    return cfg


# ══════════════════════════════════════════════════════════════
#  TRIAL
# ══════════════════════════════════════════════════════════════

@dataclass
class Trial:
    """
    Represents a single hyperparameter search trial.

    Attributes
    ----------
    trial_id    : zero-based integer
    overrides   : flat {dot_path: value} dict
    base_config : base config dict (from base_config YAML)
    trial_dir   : experiment directory for this trial (set at runtime)
    """
    trial_id:    int
    overrides:   Dict[str, Any]
    base_config: Dict
    trial_dir:   str = ""

    def build_config(self) -> Dict:
        """Return the full expanded config dict for this trial."""
        return apply_overrides(self.base_config, self.overrides)

    def config_hash(self) -> str:
        """Short deterministic hash of the overrides dict."""
        s = json.dumps(self.overrides, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:8]

    def save_config(self, path: str) -> None:
        """Write the full expanded config YAML for this trial."""
        cfg = self.build_config()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> Dict:
        """Serializable representation of this trial."""
        return {
            "trial_id":    self.trial_id,
            "overrides":   self.overrides,
            "config_hash": self.config_hash(),
            "trial_dir":   self.trial_dir,
        }


# ══════════════════════════════════════════════════════════════
#  SEARCH SPACE
# ══════════════════════════════════════════════════════════════

class SearchSpace:
    """
    Parses a search space YAML and generates trials.

    Search space YAML format:

    search:
      method: grid          # grid | random
      max_trials: 20        # only used for random search
      metric: dice_mean     # metric to optimize
      mode: max             # max | min
      seed: 42              # for reproducibility

    base_config: configs/exp_swinunetr.yaml

    space:
      training.lr: [1e-4, 5e-5]
      model.feature_size: [24, 48]
      loss.focal_weight: [0.3, 0.5]
      prior.weights:
        - [0.4, 0.4, 0.2]
        - [0.5, 0.3, 0.2]
    """

    def __init__(
        self,
        method: str,
        metric: str,
        mode: str,
        base_config_path: str,
        space: Dict[str, List],
        max_trials: int = 20,
        seed: int = 42,
        output_dir: str = "search_results",
    ):
        self.method           = method.lower()
        self.metric           = metric
        self.mode             = mode.lower()
        self.base_config_path = base_config_path
        self.raw_space        = space
        self.max_trials       = max_trials
        self.seed             = seed
        self.output_dir       = output_dir

        assert self.method in {"grid", "random"}, \
            f"method must be 'grid' or 'random', got: {self.method}"
        assert self.mode in {"max", "min"}, \
            f"mode must be 'max' or 'min', got: {self.mode}"

        # Load base config dict
        with open(base_config_path, "r", encoding="utf-8") as f:
            self.base_config: Dict = yaml.safe_load(f)

    @classmethod
    def from_yaml(cls, space_yaml_path: str, output_dir: str = "search_results") -> "SearchSpace":
        """Factory: load SearchSpace from a search space YAML file."""
        with open(space_yaml_path, "r", encoding="utf-8") as f:
            data: Dict = yaml.safe_load(f)

        search_cfg    = data.get("search", {})
        method        = search_cfg.get("method", "grid")
        max_trials    = int(search_cfg.get("max_trials", 20))
        metric        = search_cfg.get("metric", "dice_mean")
        mode          = search_cfg.get("mode", "max")
        seed          = int(search_cfg.get("seed", 42))
        base_cfg_path = data.get("base_config", "")
        space         = data.get("space", {})

        if not base_cfg_path:
            raise ValueError("search_space YAML must specify 'base_config'")
        if not os.path.isabs(base_cfg_path):
            # Resolve relative to YAML location
            base_cfg_path = os.path.join(
                os.path.dirname(os.path.abspath(space_yaml_path)),
                base_cfg_path
            )

        logger.info(
            f"[SearchSpace] method={method} | metric={metric} ({mode}) | "
            f"max_trials={max_trials} | seed={seed}"
        )

        return cls(
            method=method,
            metric=metric,
            mode=mode,
            base_config_path=base_cfg_path,
            space=space,
            max_trials=max_trials,
            seed=seed,
            output_dir=output_dir,
        )

    # ─────────────────────────────────────────────────────────
    # TRIAL GENERATION
    # ─────────────────────────────────────────────────────────

    def _all_combinations(self) -> List[Dict[str, Any]]:
        """Return all grid combinations as a list of override dicts."""
        if not self.raw_space:
            return [{}]

        keys   = list(self.raw_space.keys())
        values = [self.raw_space[k] if isinstance(self.raw_space[k], list)
                  else [self.raw_space[k]]
                  for k in keys]

        combos = []
        for combo in itertools.product(*values):
            overrides = {k: v for k, v in zip(keys, combo)}
            combos.append(overrides)

        return combos

    def generate_trials(self) -> List[Trial]:
        """
        Generate a list of Trial objects based on the search method.

        For grid search : returns all combinations (up to max_trials).
        For random search: randomly samples max_trials unique combos.
        """
        all_combos = self._all_combinations()

        if self.method == "grid":
            selected = all_combos[: self.max_trials]
            if len(all_combos) > self.max_trials:
                logger.warning(
                    f"[SearchSpace] Grid has {len(all_combos)} combinations but "
                    f"max_trials={self.max_trials}. Taking first {self.max_trials}."
                )
        else:  # random
            rng = random.Random(self.seed)
            n   = min(self.max_trials, len(all_combos))
            selected = rng.sample(all_combos, n)

        trials = []
        for i, overrides in enumerate(selected):
            trial = Trial(
                trial_id=i,
                overrides=overrides,
                base_config=self.base_config,
            )
            trials.append(trial)

        logger.info(f"[SearchSpace] Generated {len(trials)} trials ({self.method} search).")
        return trials

    def is_better(self, new_val: float, current_best: float) -> bool:
        """Return True if new_val is better than current_best given mode."""
        if self.mode == "max":
            return new_val > current_best
        return new_val < current_best


# ══════════════════════════════════════════════════════════════
#  TRIAL INDEX (search state tracker)
# ══════════════════════════════════════════════════════════════

class TrialIndex:
    """
    Persistent tracker of trial statuses across search runs.

    Statuses:
      pending   → not yet started
      running   → currently training (or was running when interrupted)
      completed → training + evaluation finished
      failed    → raised an exception

    Stored as:
        search_results/search_state.json

    This is what makes search resumable: just re-run the same command.
    Completed trials are skipped. Running trials (interrupted) are retried.
    """

    STATUS_PENDING   = "pending"
    STATUS_RUNNING   = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED    = "failed"

    def __init__(self, state_path: str):
        self.state_path = state_path
        self._data: Dict[str, Dict] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        import tempfile
        dir_ = os.path.dirname(self.state_path)
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            os.close(fd)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, default=str)
            os.replace(tmp, self.state_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def register(self, trial: Trial) -> None:
        """Register a trial as pending if not already tracked."""
        key = str(trial.trial_id)
        if key not in self._data:
            self._data[key] = {
                "trial_id":    trial.trial_id,
                "status":      self.STATUS_PENDING,
                "overrides":   trial.overrides,
                "config_hash": trial.config_hash(),
                "trial_dir":   trial.trial_dir,
                "result":      None,
                "error":       None,
                "started_at":  None,
                "finished_at": None,
            }
        self._save()

    def register_all(self, trials: List[Trial]) -> None:
        for t in trials:
            self.register(t)

    def get_status(self, trial: Trial) -> str:
        return self._data.get(str(trial.trial_id), {}).get("status", self.STATUS_PENDING)

    def mark_running(self, trial: Trial) -> None:
        from datetime import datetime
        key = str(trial.trial_id)
        self._data[key]["status"]     = self.STATUS_RUNNING
        self._data[key]["trial_dir"]  = trial.trial_dir
        self._data[key]["started_at"] = datetime.now().isoformat()
        self._save()

    def mark_completed(self, trial: Trial, result: Dict) -> None:
        from datetime import datetime
        key = str(trial.trial_id)
        self._data[key]["status"]      = self.STATUS_COMPLETED
        self._data[key]["result"]      = result
        self._data[key]["trial_dir"]   = trial.trial_dir
        self._data[key]["finished_at"] = datetime.now().isoformat()
        self._save()

    def mark_failed(self, trial: Trial, error: str) -> None:
        from datetime import datetime
        key = str(trial.trial_id)
        self._data[key]["status"]      = self.STATUS_FAILED
        self._data[key]["error"]       = error
        self._data[key]["finished_at"] = datetime.now().isoformat()
        self._save()

    def get_all(self) -> Dict[str, Dict]:
        return self._data

    def pending_trials(self, trials: List[Trial], skip_failed: bool = False) -> List[Trial]:
        """
        Return only trials that should be (re)run.

        skip_failed: if True, do not retry failed trials.
        """
        to_run = []
        for t in trials:
            status = self.get_status(t)
            if status == self.STATUS_COMPLETED:
                logger.info(f"[TrialIndex] Trial {t.trial_id:03d} already completed — skipping.")
                continue
            if status == self.STATUS_FAILED and skip_failed:
                logger.info(f"[TrialIndex] Trial {t.trial_id:03d} failed — skipping (--skip-failed).")
                continue
            to_run.append(t)
        return to_run
