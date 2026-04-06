import os
import subprocess
import glob

# ======================
# CONFIG FILES
# ======================

# v0 — original baseline (experiments 1–5)
CONFIGS_V0 = [
    "configs/exp_unet3d.yaml",
    "configs/exp_attention_unet.yaml",
    "configs/exp_unetpp.yaml",
]

# v1 — improved pipeline: WeightedDiceFocalCE + tumor_ratio 0.8 + cosine LR
CONFIGS_V1 = [
    "configs/exp_unet3d_v1.yaml",
    "configs/exp_attention_unet_v1.yaml",
    "configs/exp_unetpp_v1.yaml",
]

# Phase 1 strong CNN baselines:
#   data_v1 (augmentation + percentile clip + TC-biased sampling)
#   segmentation_v2 (warmup + grad clip + per-class Dice log + deep sup)
#   evaluation_v1 (75% overlap + CC filter + per-sample metrics)
CONFIGS_PHASE1_BASELINES = [
    "configs/exp_resunet.yaml",       # Exp 9:  ResUNet (5-level residual)
    "configs/exp_unetpp_ds.yaml",     # Exp 10: UNet++ with deep supervision
]

# Active run — set to the group you want to train next
CONFIGS = CONFIGS_PHASE1_BASELINES


def run_command(cmd):
    print(f"\n Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def get_latest_experiment():
    exp_dirs = sorted(glob.glob("experiments/exp_*"))
    if not exp_dirs:
        raise ValueError("No experiment directories found.")
    return exp_dirs[-1]


def get_best_or_latest_checkpoint(exp_dir):
    """
    Prefer best.pth (saved by segmentation_v2 trainer on lowest loss).
    Fall back to the last epoch checkpoint if best.pth not present.
    """
    best_ckpt = os.path.join(exp_dir, "checkpoints", "best.pth")
    if os.path.exists(best_ckpt):
        return best_ckpt

    ckpts = sorted(glob.glob(os.path.join(exp_dir, "checkpoints", "epoch_*.pth")))
    if not ckpts:
        raise ValueError(f"No checkpoints found in {exp_dir}")
    return ckpts[-1]


def main():

    for config in CONFIGS:

        print("\n" + "="*60)
        print(f" STARTING MODEL: {config}")
        print("="*60)

        # ── Train ────────────────────────────────────────────
        train_cmd = f"python scripts/train.py --config {config}"
        run_command(train_cmd)

        # ── Locate experiment dir ─────────────────────────────
        exp_dir = get_latest_experiment()
        print(f"[INFO] Latest experiment: {exp_dir}")

        # ── Locate checkpoint ─────────────────────────────────
        ckpt = get_best_or_latest_checkpoint(exp_dir)
        print(f"[INFO] Using checkpoint: {ckpt}")

        # ── Evaluate ──────────────────────────────────────────
        eval_cmd = f"python scripts/evaluate.py --config {config} --ckpt {ckpt}"
        run_command(eval_cmd)

        print(f"\n COMPLETED: {config}")


if __name__ == "__main__":
    main()
