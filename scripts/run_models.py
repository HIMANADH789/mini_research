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

# Phase 4 — Strong CNN Baselines:
#   data_v1 (augmentation + percentile clip + TC-biased sampling)
#   segmentation_v2 (warmup + grad clip + per-class Dice log + deep sup)
#   evaluation_v1   (75% overlap + CC filter + per-sample metrics)
CONFIGS_PHASE_CNN = [
    "configs/exp_resunet.yaml",       # Exp 9:  ResUNet  (5-level residual, TC=0.816)
    "configs/exp_unetpp_ds.yaml",     # Exp 10: UNet++ DS (dense + DS, TC=0.825) BEST CNN
]

# Phase 5 — Transformer Baselines:
#   data_v2        (extended augmentation: rotation, gamma, zoom)
#   segmentation_v3 (AdamW, warmup=20, val-based best checkpoint, poly LR option)
#   evaluation_v2   (TTA 8-flip + standard BraTS WT/TC/ET metrics)
CONFIGS_PHASE_TRANSFORMER = [
    "configs/exp_swinunetr.yaml",     # Exp 11: SwinUNETR (3D Swin + UNet decoder)
]

# Active run — set to the group you want to train next
CONFIGS = CONFIGS_PHASE_TRANSFORMER


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


def get_best_checkpoint(exp_dir):
    """
    Checkpoint priority:
      1. best.pth       — saved by trainer v3 based on validation TC Dice
      2. best_train.pth — saved by trainer v3 based on training loss (fallback)
      3. Last epoch     — final epoch checkpoint (legacy fallback)
    """
    # Priority 1: validation-based best (trainer v3)
    best_val = os.path.join(exp_dir, "checkpoints", "best.pth")
    if os.path.exists(best_val):
        return best_val

    # Priority 2: training-loss best (trainer v3 fallback)
    best_train = os.path.join(exp_dir, "checkpoints", "best_train.pth")
    if os.path.exists(best_train):
        return best_train

    # Priority 3: last epoch checkpoint (trainer v0-v2 compatibility)
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
        ckpt = get_best_checkpoint(exp_dir)
        print(f"[INFO] Using checkpoint: {ckpt}")

        # ── Evaluate ──────────────────────────────────────────
        eval_cmd = f"python scripts/evaluate.py --config {config} --ckpt {ckpt}"
        run_command(eval_cmd)

        print(f"\n COMPLETED: {config}")


if __name__ == "__main__":
    main()
