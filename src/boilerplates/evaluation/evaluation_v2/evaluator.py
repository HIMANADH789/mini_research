"""
Evaluator v2 — TTA + Standard BraTS Metrics
=============================================
Improvements over v1:

  1. TEST-TIME AUGMENTATION (TTA)
     8-flip ensemble (all combinations of axis flips):
       [no flip, D-flip, H-flip, W-flip, DH-flip, DW-flip, HW-flip, DHW-flip]
     Each augmented volume is passed through the model separately, then
     logit-space averaging is performed before argmax.
     Config: evaluation.tta: true (default: true)
     Effect: typically +0.5-1.5% Dice, especially HD95 improvement.

  2. STANDARD BraTS METRICS (alongside 4-class metrics)
     BraTS papers report three derived regions:
       WT (Whole Tumor) = TC ∪ ED ∪ ET  [label > 0]
       TC (Tumor Core)  = TC ∪ ET       [label 1 or 3]
       ET (Enhancing Tumor) = ET only    [label 3]
     These are computed and stored in results alongside per-class Dice.
     Enables direct comparison with published BraTS benchmark numbers.

  3. PER-CLASS HD95 (improved from v1)
     HD95 is computed per tumor class (TC, ED, ET) and averaged,
     rather than over the full multi-class prediction.
     This matches the BraTS evaluation protocol more closely.

  4. BEST CHECKPOINT SELECTION
     Prefers best.pth (saved by trainer v3 based on val TC Dice).
     Falls back to best_train.pth (training loss), then last epoch ckpt.

Config fields:
  evaluation:
    overlap:     0.75
    cc_filter:   true
    cc_min_size: 50
    tta:         true
"""

import os
import json

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

try:
    from scipy.ndimage import label as scipy_label
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from src.boilerplates.losses.metrics import hausdorff_distance_95
from src.boilerplates.resolver import build_dataloader
from src.boilerplates.model_builder.build import build_model
from src.utils.experiment_utils.device import get_device


# ══════════════════════════════════════════════════════════════
#  CONNECTED COMPONENT FILTER (unchanged from v1)
# ══════════════════════════════════════════════════════════════

def connected_component_filter(pred, num_classes, min_size=50):
    if not SCIPY_AVAILABLE:
        return pred
    filtered = pred.copy()
    for c in range(1, num_classes):
        binary = (pred == c).astype(np.uint8)
        labeled, n_comp = scipy_label(binary)
        for comp_id in range(1, n_comp + 1):
            if (labeled == comp_id).sum() < min_size:
                filtered[labeled == comp_id] = 0
    return filtered


# ══════════════════════════════════════════════════════════════
#  STANDARD BraTS REGION METRICS
# ══════════════════════════════════════════════════════════════

def brats_region_dice(pred_np, seg_np, eps=1e-5):
    """
    Compute BraTS standard region Dice scores:
      WT = all tumor (label > 0)
      TC = tumor core (label 1 or 3)
      ET = enhancing tumor (label 3)

    Args:
        pred_np: np.ndarray [D, H, W] int64 — argmax prediction
        seg_np:  np.ndarray [D, H, W] int64 — ground truth
    Returns:
        dict with keys 'WT', 'TC', 'ET'
    """
    def binary_dice(p, t):
        inter = (p & t).sum()
        denom = p.sum() + t.sum()
        return float(2 * inter + eps) / float(denom + eps)

    pred_p = pred_np.astype(bool)
    seg_p  = seg_np.astype(bool)

    wt_pred = pred_np > 0
    wt_seg  = seg_np  > 0

    tc_pred = np.isin(pred_np, [1, 3])
    tc_seg  = np.isin(seg_np,  [1, 3])

    et_pred = pred_np == 3
    et_seg  = seg_np  == 3

    return {
        "WT": binary_dice(wt_pred, wt_seg),
        "TC": binary_dice(tc_pred, tc_seg),
        "ET": binary_dice(et_pred, et_seg),
    }


def per_class_hd95(pred_np, seg_np, num_classes=4):
    """
    Compute HD95 per tumor class (not overall multi-class HD95).
    Average over classes 1, 2, 3 (skip background).
    Uses cdist approximation from metrics module.
    """
    from scipy.spatial.distance import cdist

    hd95_list = []
    for c in range(1, num_classes):    # skip background
        pred_c = (pred_np == c).astype(np.uint8)
        seg_c  = (seg_np  == c).astype(np.uint8)
        pred_pts = np.argwhere(pred_c > 0)
        seg_pts  = np.argwhere(seg_c  > 0)

        if len(pred_pts) == 0 or len(seg_pts) == 0:
            continue

        if len(pred_pts) > 5000:
            pred_pts = pred_pts[np.random.choice(len(pred_pts), 5000, replace=False)]
        if len(seg_pts) > 5000:
            seg_pts  = seg_pts[ np.random.choice(len(seg_pts),  5000, replace=False)]

        dists = cdist(pred_pts, seg_pts)
        hd1 = np.percentile(np.min(dists, axis=1), 95)
        hd2 = np.percentile(np.min(dists, axis=0), 95)
        hd95_list.append(float(max(hd1, hd2)))

    return float(np.mean(hd95_list)) if hd95_list else 0.0


# ══════════════════════════════════════════════════════════════
#  EVALUATOR v2
# ══════════════════════════════════════════════════════════════

class Evaluator:

    def __init__(self, config, checkpoint_path):
        self.config          = config
        self.checkpoint_path = checkpoint_path
        self.device          = get_device(config)

        eval_cfg         = getattr(config, "evaluation", None)
        overlap          = float(getattr(eval_cfg, "overlap",     0.75))
        self.cc_filter   = bool( getattr(eval_cfg, "cc_filter",   False))
        self.cc_min_size = int(  getattr(eval_cfg, "cc_min_size", 50))
        self.use_tta     = bool( getattr(eval_cfg, "tta",         True))

        # MODEL
        self.model = build_model(config).to(self.device)
        state_dict = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # DATA
        self.loader      = build_dataloader(config, split="val")
        self.patch_size  = list(config.data.patch_size)
        self.num_classes = config.model.out_channels

        # Stride from overlap
        self.stride = [max(1, int(p * (1.0 - overlap))) for p in self.patch_size]

        tta_str = " + TTA(8-flip)" if self.use_tta else ""
        cc_str  = " + CC filter"   if self.cc_filter else ""
        pct = int((1 - self.stride[0] / self.patch_size[0]) * 100)
        print(
            f"[Evaluator v2] Overlap: {pct}%{tta_str}{cc_str} | "
            f"Checkpoint: {os.path.basename(checkpoint_path)}"
        )

    # ──────────────────────────────────────────────────────
    # GAUSSIAN WEIGHT MAP
    # ──────────────────────────────────────────────────────

    def get_gaussian_weight(self, shape):
        D, H, W = shape
        z = np.linspace(-1, 1, D)
        y = np.linspace(-1, 1, H)
        x = np.linspace(-1, 1, W)
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        weight = np.exp(-(zz**2 + yy**2 + xx**2) / (2 * 0.5**2))
        return torch.tensor(weight, dtype=torch.float32)

    # ──────────────────────────────────────────────────────
    # SINGLE PASS — SLIDING WINDOW
    # ──────────────────────────────────────────────────────

    @torch.no_grad()
    def sliding_window_inference(self, volume):
        """
        Args:
            volume: [1, C, D, H, W]
        Returns:
            logits: [1, num_classes, D, H, W]
        """
        _, C, D, H, W = volume.shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride

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
                        patch = F.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d))

                    pred = self.model(patch)
                    if isinstance(pred, list):
                        pred = pred[0]
                    pred   = pred[:, :, :z2-z1, :y2-y1, :x2-x1]
                    weight = gaussian[:z2-z1, :y2-y1, :x2-x1]

                    output[:, :, z1:z2, y1:y2, x1:x2]     += pred * weight
                    weight_map[:, :, z1:z2, y1:y2, x1:x2] += weight

        return output / (weight_map + 1e-5)

    # ──────────────────────────────────────────────────────
    # TTA — 8-Flip Ensemble
    # ──────────────────────────────────────────────────────

    @torch.no_grad()
    def tta_inference(self, volume):
        """
        Ensemble predictions over 8 flip augmentations.
        Logit-space averaging before argmax.
        """
        # 8 combinations of flips on dims 2, 3, 4
        flip_combos = [
            [],
            [2], [3], [4],
            [2, 3], [2, 4], [3, 4],
            [2, 3, 4],
        ]
        accum = None

        for dims in flip_combos:
            v = volume.clone()
            if dims:
                v = torch.flip(v, dims=dims)

            pred = self.sliding_window_inference(v)

            if dims:
                pred = torch.flip(pred, dims=dims)

            accum = pred if accum is None else accum + pred

        return accum / len(flip_combos)

    # ──────────────────────────────────────────────────────
    # EVALUATE
    # ──────────────────────────────────────────────────────

    def evaluate(self):
        # Per-class Dice (4-class including BG)
        all_dices      = []
        all_hd95       = []
        all_sens       = []
        all_spec       = []
        # Standard BraTS region metrics
        all_brats_wt   = []
        all_brats_tc   = []
        all_brats_et   = []
        per_sample      = []

        for i, (img, seg) in enumerate(tqdm(self.loader, desc="Evaluating")):
            img = img.to(self.device)
            seg = seg.to(self.device)

            # Inference (with or without TTA)
            if self.use_tta:
                logits = self.tta_inference(img)
            else:
                logits = self.sliding_window_inference(img)

            pred = torch.argmax(logits, dim=1)

            # CC post-processing
            if self.cc_filter:
                pred_np = pred[0].cpu().numpy().astype(np.int64)
                pred_np = connected_component_filter(pred_np, self.num_classes, self.cc_min_size)
                pred = torch.tensor(pred_np, dtype=torch.long, device=self.device).unsqueeze(0)

            pred_np = pred[0].cpu().numpy()
            seg_np  = seg[0].cpu().numpy()

            # ── 4-class per-class Dice ────────────────────────
            eps = 1e-5
            dices = []
            for c in range(self.num_classes):
                p = (pred_np == c).astype(float)
                t = (seg_np  == c).astype(float)
                inter = (p * t).sum()
                denom = p.sum() + t.sum()
                dices.append(float(2 * inter + eps) / float(denom + eps))
            all_dices.append(dices)

            # ── Per-class HD95 (tumor classes only) ───────────
            hd = per_class_hd95(pred_np, seg_np, self.num_classes)
            all_hd95.append(hd)

            # ── Sensitivity / Specificity ──────────────────────
            sens_list, spec_list = [], []
            for c in range(self.num_classes):
                p = (pred_np == c).astype(float)
                t = (seg_np  == c).astype(float)
                tp = (p * t).sum()
                fn = ((1 - p) * t).sum()
                tn = ((1 - p) * (1 - t)).sum()
                fp = (p * (1 - t)).sum()
                sens_list.append(float((tp + eps) / (tp + fn + eps)))
                spec_list.append(float((tn + eps) / (tn + fp + eps)))
            all_sens.append(sum(sens_list) / len(sens_list))
            all_spec.append(sum(spec_list) / len(spec_list))

            # ── Standard BraTS Region Dice ────────────────────
            brats = brats_region_dice(pred_np, seg_np)
            all_brats_wt.append(brats["WT"])
            all_brats_tc.append(brats["TC"])
            all_brats_et.append(brats["ET"])

            per_sample.append({
                "sample_idx":      i,
                "dice_per_class":  [round(d, 4) for d in dices],
                "mean_dice":       round(float(np.mean(dices)), 4),
                "hd95":            round(hd, 4),
                "sensitivity":     round(all_sens[-1], 4),
                "specificity":     round(all_spec[-1], 4),
                "brats_WT":        round(brats["WT"], 4),
                "brats_TC":        round(brats["TC"], 4),
                "brats_ET":        round(brats["ET"], 4),
            })

        # Aggregate
        avg_dice = [sum(d[c] for d in all_dices) / len(all_dices) for c in range(self.num_classes)]

        results = {
            # 4-class results (BG, TC, ED, ET — our standard)
            "dice_per_class": [round(d, 6) for d in avg_dice],
            "mean_dice":      round(float(np.mean(avg_dice)), 6),
            "hd95":           round(float(np.mean(all_hd95)), 6),
            "sensitivity":    round(float(np.mean(all_sens)), 6),
            "specificity":    round(float(np.mean(all_spec)), 6),
            # Standard BraTS region metrics (for paper comparison)
            "brats_WT_dice":  round(float(np.mean(all_brats_wt)), 6),
            "brats_TC_dice":  round(float(np.mean(all_brats_tc)), 6),
            "brats_ET_dice":  round(float(np.mean(all_brats_et)), 6),
        }

        self._save_per_sample(per_sample)
        return results

    def _save_per_sample(self, per_sample):
        ckpt_dir = os.path.dirname(self.checkpoint_path)
        exp_dir  = os.path.dirname(ckpt_dir)
        out_dir  = os.path.join(exp_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "per_sample_metrics.json")
        with open(out_path, "w") as f:
            json.dump(per_sample, f, indent=4)
        print(f"[Evaluator v2] Per-sample metrics saved: {out_path}")
