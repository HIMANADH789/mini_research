"""
Data Pipeline v1
================
Improvements over v0:
  - Percentile clipping (0.5, 99.5) before Z-score normalization
    Prevents MRI intensity outliers from distorting normalization.
  - Data augmentation during training (applied jointly to image + mask):
      * RandomFlip:           each axis independently, p=0.5
      * RandomIntensityScale: per modality factor ~ U(0.9, 1.1)
      * RandomIntensityShift: per modality shift  ~ U(-0.1, 0.1)
      * RandomGaussianNoise:  zero-mean, std=0.01
  - TC-biased patch sampling:
      When a tumor patch is selected (p=tumor_ratio), there is an additional
      bias (p=0.5) to center the patch on a TC voxel specifically, not just
      any tumor voxel. This directly combats TC under-representation.

Data format expected (BraTS 2024 GLI):
  <train_root>/
    <subject_id>/
      <subject_id>-t1c.nii.gz   (T1 contrast-enhanced)
      <subject_id>-t1n.nii.gz   (T1 native)
      <subject_id>-t2f.nii.gz   (T2 FLAIR)
      <subject_id>-t2w.nii.gz   (T2 weighted)
      <subject_id>-seg.nii.gz   (segmentation: labels 0,1,2,3)

BraTS 2020 compatibility:
  Labels 0,1,2,4 are auto-detected and label 4 is remapped to 3.
"""

import os
import glob
import random

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────
# NORMALIZATION
# ──────────────────────────────────────────────────────────

def normalize_volume(volume):
    """
    Percentile-clipped Z-score normalization per modality.

    Args:
        volume: np.ndarray [C, D, H, W]  (float32)

    Returns:
        np.ndarray [C, D, H, W]  normalized
    """
    C = volume.shape[0]
    out = np.zeros_like(volume, dtype=np.float32)

    for c in range(C):
        mod = volume[c]
        brain_mask = mod > 0

        if brain_mask.sum() == 0:
            out[c] = mod
            continue

        vals = mod[brain_mask]

        # Percentile clipping to remove outlier intensities
        p_low  = np.percentile(vals, 0.5)
        p_high = np.percentile(vals, 99.5)
        mod_clipped = np.clip(mod, p_low, p_high)

        # Z-score over foreground voxels only
        brain_vals = mod_clipped[brain_mask]
        mu  = brain_vals.mean()
        std = brain_vals.std() + 1e-8

        out[c] = (mod_clipped - mu) / std

    return out


# ──────────────────────────────────────────────────────────
# AUGMENTATION
# ──────────────────────────────────────────────────────────

def augment(image, seg):
    """
    Random augmentation applied jointly to image and segmentation.
    Augmentation is only applied during training (caller's responsibility).

    Args:
        image: np.ndarray [C, D, H, W]
        seg:   np.ndarray [D, H, W]

    Returns:
        (image, seg) augmented
    """
    # ── Random Flip (each axis independently, p=0.5) ──────
    for axis in range(3):
        if random.random() < 0.5:
            image = np.flip(image, axis=axis + 1).copy()  # +1 for channel dim
            seg   = np.flip(seg,   axis=axis).copy()

    # ── Random Intensity Scale (per modality) ─────────────
    # Multiplicative factor: U(0.9, 1.1) — simulates scanner gain variation
    C = image.shape[0]
    for c in range(C):
        scale = random.uniform(0.9, 1.1)
        image[c] = image[c] * scale

    # ── Random Intensity Shift (per modality) ─────────────
    # Additive shift: U(-0.1, 0.1) — simulates scanner bias field
    for c in range(C):
        shift = random.uniform(-0.1, 0.1)
        image[c] = image[c] + shift

    # ── Random Gaussian Noise ─────────────────────────────
    # Low-variance noise: simulates acquisition noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
        image = image + noise

    return image, seg


# ──────────────────────────────────────────────────────────
# PATCH SAMPLING
# ──────────────────────────────────────────────────────────

def sample_patch(image, seg, patch_size, tumor_ratio):
    """
    Sample a 3D patch from the volume.

    Sampling strategy (v1):
      With probability `tumor_ratio`:
        - With probability 0.5: center on a TC (label=1) voxel specifically
        - Else: center on any tumor voxel (label 1,2,3)
      Otherwise: random location in the volume.

    Args:
        image:      np.ndarray [C, D, H, W]
        seg:        np.ndarray [D, H, W]
        patch_size: list [pd, ph, pw]
        tumor_ratio: float

    Returns:
        (img_patch [C, pd, ph, pw], seg_patch [pd, ph, pw])
    """
    C, D, H, W = image.shape
    pd, ph, pw = patch_size

    # Maximum valid starting coordinates
    max_d = max(D - pd, 0)
    max_h = max(H - ph, 0)
    max_w = max(W - pw, 0)

    center = None

    if random.random() < tumor_ratio:
        # Try TC-focused sampling first (p=0.5)
        if random.random() < 0.5:
            tc_voxels = np.argwhere(seg == 1)  # label 1 = TC
            if len(tc_voxels) > 0:
                center = tc_voxels[random.randint(0, len(tc_voxels) - 1)]

        # Fall back to any tumor voxel if TC not found or not chosen
        if center is None:
            tumor_voxels = np.argwhere(seg > 0)
            if len(tumor_voxels) > 0:
                center = tumor_voxels[random.randint(0, len(tumor_voxels) - 1)]

    if center is not None:
        # Place patch centered on the sampled voxel (with clamping)
        d0 = int(np.clip(center[0] - pd // 2, 0, max_d))
        h0 = int(np.clip(center[1] - ph // 2, 0, max_h))
        w0 = int(np.clip(center[2] - pw // 2, 0, max_w))
    else:
        # Random sampling
        d0 = random.randint(0, max_d)
        h0 = random.randint(0, max_h)
        w0 = random.randint(0, max_w)

    img_patch = image[:, d0:d0+pd, h0:h0+ph, w0:w0+pw]
    seg_patch = seg[d0:d0+pd, h0:h0+ph, w0:w0+pw]

    # Pad if patch extends beyond volume (edge cases)
    if img_patch.shape[1:] != (pd, ph, pw):
        pad_d = pd - img_patch.shape[1]
        pad_h = ph - img_patch.shape[2]
        pad_w = pw - img_patch.shape[3]
        img_patch = np.pad(img_patch, ((0,0),(0,pad_d),(0,pad_h),(0,pad_w)))
        seg_patch = np.pad(seg_patch, ((0,pad_d),(0,pad_h),(0,pad_w)))

    return img_patch, seg_patch


# ──────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────

class BraTSDataset(Dataset):
    """
    BraTS 3D patch dataset with augmentation (v1).

    Each __getitem__ loads a full volume, normalizes it, applies
    augmentation (if training), and samples one patch.
    """

    # Modality file suffixes in order: T1c, T1n, T2f (FLAIR), T2w
    MODALITY_SUFFIXES = ["-t1c.nii.gz", "-t1n.nii.gz", "-t2f.nii.gz", "-t2w.nii.gz"]

    def __init__(self, subject_dirs, patch_size, tumor_ratio, augment_data=False):
        """
        Args:
            subject_dirs:  list of subject directory paths
            patch_size:    [pd, ph, pw]
            tumor_ratio:   float, probability of tumor-centered patch
            augment_data:  bool, apply augmentation (True for training)
        """
        self.subject_dirs = subject_dirs
        self.patch_size   = patch_size
        self.tumor_ratio  = tumor_ratio
        self.augment_data = augment_data

    def __len__(self):
        return len(self.subject_dirs)

    def _load_volume(self, subject_dir):
        """Load 4 modalities and segmentation from a subject folder."""
        subject_id = os.path.basename(subject_dir)

        modalities = []
        for suffix in self.MODALITY_SUFFIXES:
            path = os.path.join(subject_dir, subject_id + suffix)
            if not os.path.exists(path):
                # Fallback: glob for any file with this suffix pattern
                candidates = glob.glob(os.path.join(subject_dir, f"*{suffix}"))
                if not candidates:
                    raise FileNotFoundError(
                        f"Modality file not found: {path}"
                    )
                path = candidates[0]
            modalities.append(nib.load(path).get_fdata(dtype=np.float32))

        seg_path = os.path.join(subject_dir, subject_id + "-seg.nii.gz")
        if not os.path.exists(seg_path):
            candidates = glob.glob(os.path.join(subject_dir, "*-seg.nii.gz"))
            if not candidates:
                raise FileNotFoundError(f"Segmentation not found in: {subject_dir}")
            seg_path = candidates[0]

        seg = nib.load(seg_path).get_fdata(dtype=np.float32).astype(np.int64)

        # BraTS 2020 compatibility: remap label 4 → 3
        seg[seg == 4] = 3

        image = np.stack(modalities, axis=0)  # [4, D, H, W]
        return image, seg

    def __getitem__(self, idx):
        subject_dir = self.subject_dirs[idx]

        image, seg = self._load_volume(subject_dir)

        # Normalize
        image = normalize_volume(image)

        # Augment (training only)
        if self.augment_data:
            image, seg = augment(image, seg)

        # Sample patch
        img_patch, seg_patch = sample_patch(
            image, seg, self.patch_size, self.tumor_ratio
        )

        img_tensor = torch.tensor(img_patch, dtype=torch.float32)
        seg_tensor = torch.tensor(seg_patch, dtype=torch.long)

        return img_tensor, seg_tensor


# ──────────────────────────────────────────────────────────
# FULL-VOLUME DATASET (validation / evaluation)
# ──────────────────────────────────────────────────────────

class BraTSFullVolumeDataset(Dataset):
    """
    Returns full normalized volumes for sliding window inference.
    No augmentation, no patch sampling.
    """

    MODALITY_SUFFIXES = ["-t1c.nii.gz", "-t1n.nii.gz", "-t2f.nii.gz", "-t2w.nii.gz"]

    def __init__(self, subject_dirs):
        self.subject_dirs = subject_dirs

    def __len__(self):
        return len(self.subject_dirs)

    def _load_volume(self, subject_dir):
        subject_id = os.path.basename(subject_dir)

        modalities = []
        for suffix in self.MODALITY_SUFFIXES:
            path = os.path.join(subject_dir, subject_id + suffix)
            if not os.path.exists(path):
                candidates = glob.glob(os.path.join(subject_dir, f"*{suffix}"))
                if not candidates:
                    raise FileNotFoundError(f"Modality file not found: {path}")
                path = candidates[0]
            modalities.append(nib.load(path).get_fdata(dtype=np.float32))

        seg_path = os.path.join(subject_dir, subject_id + "-seg.nii.gz")
        if not os.path.exists(seg_path):
            candidates = glob.glob(os.path.join(subject_dir, "*-seg.nii.gz"))
            if not candidates:
                raise FileNotFoundError(f"Segmentation not found in: {subject_dir}")
            seg_path = candidates[0]

        seg = nib.load(seg_path).get_fdata(dtype=np.float32).astype(np.int64)
        seg[seg == 4] = 3

        image = np.stack(modalities, axis=0)  # [4, D, H, W]
        return image, seg

    def __getitem__(self, idx):
        subject_dir = self.subject_dirs[idx]
        image, seg = self._load_volume(subject_dir)
        image = normalize_volume(image)
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(seg, dtype=torch.long),
        )


# ──────────────────────────────────────────────────────────
# BUILD DATALOADER (resolver entry point)
# ──────────────────────────────────────────────────────────

def build_dataloader(config, split="train"):
    """
    Args:
        config: Config object
        split:  "train" or "val"

    Returns:
        DataLoader yielding (img [B,4,D,H,W], seg [B,D,H,W])
    """
    train_root  = config.data.train_root
    val_ratio   = float(config.data.val_ratio)
    patch_size  = list(config.data.patch_size)
    tumor_ratio = float(config.data.tumor_ratio)
    num_workers = int(config.data.num_workers)
    max_samples = getattr(config.data, "max_samples", None)
    batch_size  = int(config.training.batch_size)

    # Discover subject directories
    all_dirs = sorted([
        os.path.join(train_root, d)
        for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
    ])

    if not all_dirs:
        raise RuntimeError(f"No subject directories found in: {train_root}")

    if max_samples:
        all_dirs = all_dirs[:max_samples]

    # Deterministic split (same as v0: last val_ratio% = validation)
    n_val   = max(1, int(len(all_dirs) * val_ratio))
    n_train = len(all_dirs) - n_val
    train_dirs = all_dirs[:n_train]
    val_dirs   = all_dirs[n_train:]

    if split == "train":
        dataset = BraTSDataset(
            subject_dirs=train_dirs,
            patch_size=patch_size,
            tumor_ratio=tumor_ratio,
            augment_data=True,          # augmentation ON for training
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    else:  # val
        dataset = BraTSFullVolumeDataset(subject_dirs=val_dirs)
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
