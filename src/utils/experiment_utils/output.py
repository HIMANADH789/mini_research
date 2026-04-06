import os
import json
from datetime import datetime


def get_output_dir(exp_path):
    output_dir = os.path.join(exp_path, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_evaluation_results(results, exp_path, filename="evaluation_results.json"):
    output_dir = get_output_dir(exp_path)

    save_path = os.path.join(output_dir, filename)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[INFO] Evaluation results saved to {save_path}")

    return save_path


def save_evaluation_summary(results, exp_path):
    output_dir = get_output_dir(exp_path)

    save_path = os.path.join(output_dir, "evaluation_summary.txt")

    with open(save_path, "w") as f:
        f.write("===== Evaluation Summary =====\n\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")

        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    print(f"[INFO] Evaluation summary saved to {save_path}")