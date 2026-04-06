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

# Remaining experiments — unet3d_v1 already completed as exp_006
# exp_007 = attention_unet_v1,  exp_008 = unetpp_v1
CONFIGS = [
    "configs/exp_attention_unet_v1.yaml",
    "configs/exp_unetpp_v1.yaml",
]


def run_command(cmd):
    print(f"\n🚀 Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def get_latest_experiment():
    exp_dirs = sorted(glob.glob("experiments/exp_*"))
    return exp_dirs[-1]


def get_latest_checkpoint(exp_dir):
    ckpts = sorted(glob.glob(os.path.join(exp_dir, "checkpoints", "epoch_*.pth")))

    if len(ckpts) == 0:
        raise ValueError(f"No checkpoints found in {exp_dir}")

    return ckpts[-1]


def main():

    for config in CONFIGS:

        print("\n" + "="*50)
        print(f"🔥 STARTING MODEL: {config}")
        print("="*50)

        # ======================
        # TRAIN
        # ======================
        train_cmd = f"python scripts/train.py --config {config}"
        run_command(train_cmd)

        # ======================
        # GET EXP DIR
        # ======================
        exp_dir = get_latest_experiment()
        print(f"[INFO] Latest experiment: {exp_dir}")

        # ======================
        # GET CHECKPOINT
        # ======================
        ckpt = get_latest_checkpoint(exp_dir)
        print(f"[INFO] Using checkpoint: {ckpt}")

        # ======================
        # EVALUATE
        # ======================
        eval_cmd = f"python scripts/evaluate.py --config {config} --ckpt {ckpt}"
        run_command(eval_cmd)

        print(f"\n✅ COMPLETED: {config}")


if __name__ == "__main__":
    main()