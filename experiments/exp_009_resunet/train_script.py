import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import shutil

# CONFIG + UTILS
from src.utils.experiment_utils.config import load_config
from src.utils.experiment_utils.seed import set_seed
from src.utils.experiment_utils.experiment import create_experiment
from src.utils.experiment_utils.logger import get_logger
from src.utils.experiment_utils.io import save_environment

# RESOLVER
from src.boilerplates.resolver import get_trainer_class


def main(config_path):

    # LOAD CONFIG
    config = load_config(config_path)

    # SEED
    set_seed(config.seed)

    # CREATE EXPERIMENT
    exp_path = create_experiment(config)

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

    # TRAINER — version resolved from config.versions.trainer
    Trainer = get_trainer_class(config)
    trainer = Trainer(config, exp_path, logger)

    # TRAIN
    trainer.train()

    logger.info("Training Completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    main(args.config)