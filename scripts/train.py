import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import shutil
import glob

# CONFIG + UTILS
from src.utils.experiment_utils.config import load_config
from src.utils.experiment_utils.seed import set_seed
from src.utils.experiment_utils.experiment import create_experiment
from src.utils.experiment_utils.logger import get_logger
from src.utils.experiment_utils.io import save_environment

# CHECKPOINT
from src.utils.checkpoint_utils import CheckpointManager

# RESOLVER
from src.boilerplates.resolver import get_trainer_class


def _resolve_resume(resume_arg: str, exp_path: str):
    """
    Resolve --resume argument to an actual checkpoint path or None.

    resume_arg:
      "none" or "false"  → no resume (default)
      "auto"             → search inside exp_path for a resume checkpoint
      "path/to/ckpt.pt"  → use directly
    """
    if not resume_arg or resume_arg.lower() in ("none", "false", ""):
        return None

    if resume_arg.lower() == "auto":
        found = CheckpointManager.find_resume_checkpoint(exp_path)
        if found:
            print(f"[Resume] Auto-detected checkpoint: {found}")
        else:
            print("[Resume] No checkpoint found — starting fresh.")
        return found

    if os.path.isfile(resume_arg):
        return resume_arg

    raise FileNotFoundError(f"[Resume] Checkpoint not found: {resume_arg}")


def main(config_path: str, resume: str = "none", resume_from_dir: str = None):

    # LOAD CONFIG
    config = load_config(config_path)

    # SEED
    set_seed(config.seed)

    # ── Resolve experiment path ───────────────────────────────────────
    # If we have an existing dir to resume from, reuse it; otherwise create new.
    if resume_from_dir and os.path.isdir(resume_from_dir):
        exp_path = resume_from_dir
        print(f"[Train] Reusing existing experiment folder: {exp_path}")
    else:
        exp_path = create_experiment(config)
        print(f"[Train] Created new experiment folder: {exp_path}")

    # ── Resolve checkpoint path ───────────────────────────────────────
    resume_path = _resolve_resume(resume, exp_path)

    # SAVE CONFIG
    shutil.copy(config_path, os.path.join(exp_path, "config.yaml"))

    # SAVE SCRIPT
    shutil.copy(__file__, os.path.join(exp_path, "train_script.py"))

    # SAVE ENV
    save_environment(exp_path)

    # LOGGER
    logger = get_logger(os.path.join(exp_path, "logs", "train.log"))

    logger.info("Starting Experiment")
    logger.info(f"Experiment Path: {exp_path}")
    if resume_path:
        logger.info(f"Resuming from:   {resume_path}")

    # TRAINER — version resolved from config.versions.trainer
    Trainer = get_trainer_class(config)
    trainer = Trainer(config, exp_path, logger)

    # RESUME (call before train() — restores model/optimizer/scheduler/scaler/EMA)
    if resume_path and hasattr(trainer, "resume_from"):
        trainer.resume_from(resume_path)
    elif resume_path:
        # Fallback for older trainers that don't have resume_from()
        logger.warning(
            "[Train] Trainer does not support resume_from(). "
            "Loading model weights only (legacy fallback)."
        )
        import torch
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)
        trainer.model.load_state_dict(state, strict=False)

    # TRAIN
    trainer.train()

    logger.info("Training Completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a single experiment. See run_experiment.py for full search support."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML config")
    parser.add_argument(
        "--resume", type=str, default="none",
        help=(
            "Resume training. Options:\n"
            "  'none'  — start fresh (default)\n"
            "  'auto'  — find latest resume checkpoint automatically\n"
            "  <path>  — explicit checkpoint .pt file"
        )
    )
    parser.add_argument(
        "--resume-from-dir", type=str, default=None,
        help="Existing experiment directory to resume from (optional)."
    )

    args = parser.parse_args()
    main(args.config, resume=args.resume, resume_from_dir=args.resume_from_dir)