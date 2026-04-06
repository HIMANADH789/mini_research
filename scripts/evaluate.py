import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

from src.utils.experiment_utils.config import load_config
from src.utils.experiment_utils.output import (
    save_evaluation_results,
    save_evaluation_summary
)

from src.boilerplates.resolver import get_evaluator_class


def main(config_path, checkpoint_path):

    config = load_config(config_path)

    # EVALUATOR — version resolved from config.versions.evaluation
    Evaluator = get_evaluator_class(config)
    evaluator = Evaluator(config, checkpoint_path)

    results = evaluator.evaluate()

    print("\n===== Evaluation Results =====")
    for k, v in results.items():
        print(f"{k}: {v}")

    # EXPERIMENT DIR
    exp_dir = os.path.dirname(os.path.dirname(checkpoint_path))

    # SAVE
    save_evaluation_results(results, exp_dir)
    save_evaluation_summary(results, exp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)

    args = parser.parse_args()

    main(args.config, args.ckpt)