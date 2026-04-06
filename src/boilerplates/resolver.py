"""
Version Resolver
================
Reads config.versions.* and dynamically imports the correct versioned module.

Config example:
    versions:
      data: v0
      trainer: segmentation_v0
      evaluation: v0

Adding a new version:
  1. Create src/boilerplates/data/data_v1/ with dataloader.py + brats_dataset.py
  2. Set `versions.data: v1` in your config YAML
  No code changes needed here.
"""

import importlib


def _get_versions(config):
    return getattr(config, "versions", None)


def build_dataloader(config, split="train"):
    """Load build_dataloader from the versioned data module and call it."""
    versions = _get_versions(config)
    version = getattr(versions, "data", "v0") if versions else "v0"
    module = importlib.import_module(
        f"src.boilerplates.data.data_{version}.dataloader"
    )
    return module.build_dataloader(config, split)


def get_trainer_class(config):
    """Return the Trainer class from the versioned trainer module."""
    versions = _get_versions(config)
    version = getattr(versions, "trainer", "segmentation_v0") if versions else "segmentation_v0"
    module = importlib.import_module(
        f"src.boilerplates.trainers.{version}.segmentation_trainer"
    )
    return module.Trainer


def get_evaluator_class(config):
    """Return the Evaluator class from the versioned evaluation module."""
    versions = _get_versions(config)
    version = getattr(versions, "evaluation", "v0") if versions else "v0"
    module = importlib.import_module(
        f"src.boilerplates.evaluation.evaluation_{version}.evaluator"
    )
    return module.Evaluator
