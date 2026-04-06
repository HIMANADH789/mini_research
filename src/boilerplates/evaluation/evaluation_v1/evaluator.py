"""
Evaluator v1
============
Improvements over v0:
  - Configurable sliding window overlap (default 75%):
      stride = patch_size * (1 - overlap)
      Higher overlap → better boundary consistency (especially HD95)
      at the cost of ~4x more inference passes.
  - Connected component post-processing (optional):
      Removes isolated false-positive regions below a minimum voxel size.
      Applied per tumor class after argmax. Does not affect background.
      Config: evaluation.cc_filter: true, evaluation.cc_min_size: 50
  - Saves per-sample metrics to outputs/per_sample_metrics.json
      Enables analysis of variance across subjects, not just mean.
  - Best checkpoint auto-detection:
      If checkpoint_path ends with "best.pth", logs it explicitly.

Config fields consumed (beyond v0):
  evaluation:
    overlap:     0.75   # sliding window overlap fraction (default 0.75)
    cc_filter:   true   # apply connected component filtering (default false)
    cc_min_size: 50     # minimum component size in voxels (default 50)
"""

import os
import json

import torch
import numpy as np
from tqdm import tqdm

try:
    from scipy.ndimage import label as scipy_label
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from src.boilerplates.losses.metrics import (
    dice_score,
    hausdorff_distance_95,
    sensitivity,
    specificity,
)
from src.boilerplates.resolver import build_dataloader
from src.boilerplates.model_builder.build import build_model
from src.utils.experiment_utils.device import get_device


# ──────────────────────────────────────────────────────────
# CONNECTED COMPONENT FILTERING
# ──────────────────────────────────────────────────────────

def connected_component_filter(pred, num_classes, min_size=50):
    """
    Remove isolated components smaller than min_size voxels for each
    tumor class (classes 1, 2, 3 — not background).

    Args:
        pred:        np.ndarray [D, H, W] int64 — argmax prediction
        num_classes: int
        min_size:    int — minimum component size to keep

    Returns:
        np.ndarray [D, H, W] filtered prediction
    """
    if not SCIPY_AVAILABLE:
        return pred

    filtered = pred.copy()
    for c in range(1, num_classes):   # skip background (class 0)
        binary = (pred == c).astype(np.uint8)
        labeled, n_components = scipy_label(binary)

        for comp_id in range(1, n_components + 1):
            comp_mask = (labeled == comp_id)
            if comp_mask.sum() < min_size:
                filtered[comp_mask] = 0   # remove small component → background

    return filtered


# ──────────────────────────────────────────────────────────
# EVALUATOR
# ──────────────────────────────────────────────────────────

class Evaluator:

    def __init__(self, config, checkpoint_path):
        self.config          = config
        self.checkpoint_path = checkpoint_path
        self.device          = get_device(config)

        # ── Evaluation config ─────────────────────────────────
        eval_cfg        = getattr(config, "evaluation", None)
        overlap         = float(getattr(eval_cfg, "overlap",     0.75))
        self.cc_filter  = bool( getattr(eval_cfg, "cc_filter",   False))
        self.cc_min_size = int( getattr(eval_cfg, "cc_min_size", 50))

        # ── Model ─────────────────────────────────────────────
        self.model = build_model(config).to(self.device)

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # ── Data ──────────────────────────────────────────────
        self.loader      = build_dataloader(config, split="val")
        self.patch_size  = list(config.data.patch_size)
        self.num_classes = config.model.out_channels

        # Stride from overlap
        self.stride = [
            max(1, int(p * (1.0 - overlap))) for p in self.patch_size
        ]

    # ──────────────────────────────────────────────────────
    # GAUSSIAN WEIGHT MAP
    # ──────────────────────────────────────────────────────

    def get_gaussian_weight(self, shape):
        D, H, W = shape
        z = np.linspace(-1, 1, D)
        y = np.linspace(-1, 1, H)
        x = np.linspace(-1, 1, W)
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        dist   = zz**2 + yy**2 + xx**2
        weight = np.exp(-dist / (2 * 0.5**2))
        return torch.tensor(weight, dtype=torch.float32)

    # ──────────────────────────────────────────────────────
    # SLIDING WINDOW INFERENCE
    # ──────────────────────────────────────────────────────

    def sliding_window_inference(self, volume):
        """
        Args:
            volume: Tensor [1, C, D, H, W]

        Returns:
            logits: Tensor [1, num_classes, D, H, W]
        """
        _, C, D, H, W = volume.shape
        pd, ph, pw    = self.patch_size
        sd, sh, sw    = self.stride

        output     = torch.zeros((1, self.num_classes, D, H, W), device=self.device)
        weight_map = torch.zeros_like(output)
        gaussian   = self.get_gaussian_weight((pd, ph, pw)).to(self.device)

        for z in range(0, D, sd):
            for y in range(0, H, sh):
                for x in range(0, W, sw):
                    z1, y1, x1 = z, y, x
                    z2 = min(z + pd, D)
                    y2 = min(y + ph, H)
                    x2 = min(x + pw, W)

                    patch = volume[:, :, z1:z2, y1:y2, x1:x2]

                    pad_d = pd - (z2 - z1)
                    pad_h = ph - (y2 - y1)
                    pad_w = pw - (x2 - x1)

                    if pad_d or pad_h or pad_w:
                        patch = torch.nn.functional.pad(
                            patch, (0, pad_w, 0, pad_h, 0, pad_d)
                        )

                    pred = self.model(patch)

                    # If model returns list (training mode was somehow on), use main
                    if isinstance(pred, list):
                        pred = pred[0]

                    pred   = pred[:, :, :z2-z1, :y2-y1, :x2-x1]
                    weight = gaussian[:z2-z1, :y2-y1, :x2-x1]

                    output[:, :, z1:z2, y1:y2, x1:x2]     += pred * weight
                    weight_map[:, :, z1:z2, y1:y2, x1:x2] += weight

        return output / (weight_map + 1e-5)

    # ──────────────────────────────────────────────────────
    # EVALUATE
    # ──────────────────────────────────────────────────────

    def evaluate(self):
        """
        Run evaluation on the validation set.

        Returns:
            dict with keys:
              dice_per_class, mean_dice, hd95, sensitivity, specificity
        """
        total_dice = []
        total_hd   = []
        total_sens = []
        total_spec = []
        per_sample  = []

        cc_label = " + CC filter" if self.cc_filter else ""
        overlap_pct = int((1 - self.stride[0] / self.patch_size[0]) * 100)
        print(f"[Evaluator v1] Overlap: {overlap_pct}%{cc_label} | "
              f"Checkpoint: {os.path.basename(self.checkpoint_path)}")

        with torch.no_grad():
            for i, (img, seg) in enumerate(tqdm(self.loader, desc="Evaluating")):
                img = img.to(self.device)
                seg = seg.to(self.device)

                out  = self.sliding_window_inference(img)
                pred = torch.argmax(out, dim=1)

                # Optional CC post-processing
                if self.cc_filter:
                    pred_np = pred[0].cpu().numpy().astype(np.int64)
                    pred_np = connected_component_filter(
                        pred_np, self.num_classes, self.cc_min_size
                    )
                    pred = torch.tensor(
                        pred_np, dtype=torch.long, device=self.device
                    ).unsqueeze(0)

                # Metrics
                dices = dice_score(out, seg)
                hd    = hausdorff_distance_95(pred[0], seg[0])
                sens  = sensitivity(pred, seg)
                spec  = specificity(pred, seg)

                total_dice.append(dices)
                total_hd.append(hd)
                total_sens.append(sens)
                total_spec.append(spec)

                per_sample.append({
                    "sample_idx":   i,
                    "dice_per_class": [round(float(d), 4) for d in dices],
                    "mean_dice":    round(float(torch.tensor(dices).mean()), 4),
                    "hd95":         round(float(hd), 4),
                    "sensitivity":  round(float(sens), 4),
                    "specificity":  round(float(spec), 4),
                })

        avg_dice = torch.tensor(total_dice).mean(dim=0)

        results = {
            "dice_per_class": avg_dice.tolist(),
            "mean_dice":      float(avg_dice.mean()),
            "hd95":           float(sum(total_hd) / len(total_hd)),
            "sensitivity":    float(sum(total_sens) / len(total_sens)),
            "specificity":    float(sum(total_spec) / len(total_spec)),
        }

        # Save per-sample metrics alongside aggregate results
        self._save_per_sample(per_sample)

        return results

    def _save_per_sample(self, per_sample):
        """Save per-sample metrics to outputs/per_sample_metrics.json."""
        # Derive exp output dir from checkpoint path (…/checkpoints/xxx.pth)
        ckpt_dir = os.path.dirname(self.checkpoint_path)
        exp_dir  = os.path.dirname(ckpt_dir)
        out_dir  = os.path.join(exp_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "per_sample_metrics.json")
        with open(out_path, "w") as f:
            json.dump(per_sample, f, indent=4)
        print(f"[Evaluator v1] Per-sample metrics saved: {out_path}")
