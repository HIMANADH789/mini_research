import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.experiment_utils.config import load_config
from src.boilerplates.evaluation.evaluator import Evaluator


def main():

    # ======================
    # CONFIG + CHECKPOINT
    # ======================
    config_path = "configs/exp_unet3d.yaml"
    checkpoint_path = "experiments/exp_001_unet3d/checkpoints/epoch_99.pth"

    # ======================
    # LOAD CONFIG
    # ======================
    config = load_config(config_path)

    # ======================
    # EVALUATOR
    # ======================
    evaluator = Evaluator(config, checkpoint_path)

    # ======================
    # RUN
    # ======================
    results = evaluator.evaluate()

    # ======================
    # PRINT RESULTS
    # ======================
    print("\n" + "=" * 50)
    print("🔥 UNET3D EVALUATION RESULTS")
    print("=" * 50)

    for k, v in results.items():
        print(f"{k}: {v}")

    print("=" * 50)


if __name__ == "__main__":
    main()