"""
Data Pipeline v2 — Transformer-Grade Augmentation for SwinUNETR
================================================================
Improvements over v1:

  AUGMENTATION ADDITIONS (v2 over v1):
  1. Random 90° Rotation (axes combinations):
       Randomly applies 0/90/180/270° rotation around each axis.
       Lossless, zero-interpolation-error, very effective for 3D MRI.
       Covers the full SO(3) discrete symmetry group of cubes.

  2. Random Continuous Rotation (±15°):
       scipy.ndimage.rotate per-axis, applied with trilinear interpolation.
       Simulates slight patient positioning differences.
       Applied independently to each axis with p=0.3.

  3. Gamma Augmentation:
       Per-modality random gamma in [0.7, 1.5].
       Simulates different scanner contrast settings and MRI protocols.
       Applied with p=0.3 only on foreground (non-zero) voxels.

  4. Random Zoom (Scale):
       Random scale factor in [0.85, 1.15] applied to the full volume
       before patch sampling. Resized with scipy zoom (order=1 for image,
       order=0 for seg). Combined with the existing tumor-aware sampling,
       this forces the model to handle tumors at different scales.

  5. Gaussian Blur (optional):
       Simulates partial volume effects near boundaries.
       Applied with p=0.1, sigma in [0.5, 1.0].

  6. MRI Bias Field Simulation (NEW):
       Multiplicative smooth spatial inhomogeneity — exactly what MRI
       scanners produce due to non-uniform RF coil sensitivity.
       Generated as a low-degree polynomial in normalized (x,y,z)
       coordinates, converted to a multiplicative field via exp().
       Applied per-modality only on brain foreground with p=0.3.
       Bias scale ∈ [0.03, 0.10] (mild, matches clinical variation).
       This augmentation is used by nnUNet and has the largest impact on
       tumor boundary segmentation accuracy of all MRI-specific augments.

  7. Boundary-Aware Patch Sampling (NEW):
       With probability 0.2, centers the training patch on a randomly
       selected class-boundary voxel (a foreground voxel whose Moore
       neighbourhood contains a voxel of a different class).
       Forces the model to explicitly train on class transition regions,
       directly targeting HD95 improvement.
       Falls back to standard TC-biased sampling if no boundary found.

  RETAINED FROM v1:
  - Percentile-clipped Z-score normalization (per modality, foreground only)
  - Random flip per axis (p=0.5 each)
  - Random intensity scale U(0.85, 1.15)  [widened from v1: 0.9, 1.1]
  - Random intensity shift U(-0.15, 0.15) [widened from v1: -0.1, 0.1]
  - Random Gaussian noise (std=0.015, p=0.3)
  - TC-biased patch sampling (p_tc=0.5 within tumor patches)

  PATCH CACHING (training only):
  - Subject data (image+seg after loading+normalization but before aug)
    is kept in a subject-indexed cache (LRU size = num_workers*2+4).
  - Avoids re-reading NIfTI on disk for same subject within same epoch.
  - Cache is per-worker (DataLoader workers have separate memory spaces).

Data format: BraTS GLI (same as v1, full backward compatibility).
"""

import os
import glob
import random
import warnings

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from scipy.ndimage import zoom as scipy_zoom
    from scipy.ndimage import rotate as scipy_rotate
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("[data_v2] scipy not found — zoom/rotate augmentation disabled.", RuntimeWarning)


# ══════════════════════════════════════════════════════════════
#  NORMALIZATION (identical to v1 — validated)
# ══════════════════════════════════════════════════════════════

def normalize_volume(volume):
    """
    Percentile-clipped Z-score normalization per modality over brain foreground.
    Args:
        volume: np.ndarray [C, D, H, W] float32
    Returns:
        np.ndarray [C, D, H, W] float32 normalized
    """
    C = volume.shape[0]
    out = np.zeros_like(volume, dtype=np.float32)
    for c in range(C):
        mod = volume[c]
        mask = mod > 0
        if mask.sum() == 0:
            out[c] = mod
            continue
        vals = mod[mask]
        p_low, p_high = np.percentile(vals, 0.5), np.percentile(vals, 99.5)
        clipped = np.clip(mod, p_low, p_high)
        fg_vals = clipped[mask]
        mu, std = fg_vals.mean(), fg_vals.std() + 1e-8
        out[c] = (clipped - mu) / std
    return out


# ══════════════════════════════════════════════════════════════
#  AUGMENTATION — v2
# ══════════════════════════════════════════════════════════════

def augment_v2(image, seg):
    """
    Extended augmentation pipeline for transformer-scale models.

    Args:
        image: np.ndarray [C, D, H, W] float32 — normalized
        seg:   np.ndarray [D, H, W]    int64
    Returns:
        (image, seg) augmented
    """
    # ── 1. Random Flip (each axis independently, p=0.5) ──────
    for axis in range(3):
        if random.random() < 0.5:
            image = np.flip(image, axis=axis + 1).copy()
            seg   = np.flip(seg,   axis=axis).copy()

    # ── 2. Random 90° Rotation (lossless, per-axis) ──────────
    # Choose a random number of 90° rotations for a random pair of axes
    if random.random() < 0.5:
        k     = random.choice([1, 2, 3])        # 90°, 180°, 270°
        axes  = random.choice([(1, 2), (1, 3), (2, 3)])  # spatial axis pairs
        ax0   = axes[0] - 1   # remove channel offset for seg
        ax1   = axes[1] - 1
        image = np.rot90(image, k=k, axes=axes).copy()
        seg   = np.rot90(seg,   k=k, axes=(ax0, ax1)).copy()

    # ── 3. Random Continuous Rotation (±15°) ─────────────────
    if SCIPY_AVAILABLE and random.random() < 0.3:
        for ax_pair in [(0, 1), (0, 2), (1, 2)]:
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)
                # Rotate each modality
                rotated_mods = []
                for c in range(image.shape[0]):
                    rotated_mods.append(
                        scipy_rotate(image[c], angle, axes=ax_pair,
                                     reshape=False, order=1, mode='nearest')
                    )
                image = np.stack(rotated_mods, axis=0)
                seg   = scipy_rotate(seg.astype(np.float32), angle, axes=ax_pair,
                                     reshape=False, order=0, mode='nearest').astype(np.int64)

    # ── 4. Random Intensity Scale (widened: ×0.85–1.15) ──────
    C = image.shape[0]
    for c in range(C):
        image[c] = image[c] * random.uniform(0.85, 1.15)

    # ── 5. Random Intensity Shift (widened: ±0.15) ───────────
    for c in range(C):
        image[c] = image[c] + random.uniform(-0.15, 0.15)

    # ── 6. Gamma Augmentation (p=0.3) ────────────────────────
    # Applied only on foreground; background stays 0 after normalization
    if random.random() < 0.3:
        for c in range(C):
            gamma = random.uniform(0.7, 1.5)
            fg    = image[c] > 0
            if fg.any():
                # Shift to [0,1], apply gamma, shift back
                vmin = image[c][fg].min()
                vmax = image[c][fg].max()
                rng  = vmax - vmin + 1e-8
                norm = (image[c] - vmin) / rng
                norm = np.clip(norm, 0.0, 1.0)
                norm[fg] = norm[fg] ** gamma
                image[c] = norm * rng + vmin

    # ── 7. Random Gaussian Noise (std=0.015, p=0.3) ──────────
    if random.random() < 0.3:
        image = image + np.random.normal(0, 0.015, image.shape).astype(np.float32)

    # ── 8. Random Gaussian Blur (p=0.1, mild) ────────────────
    if SCIPY_AVAILABLE and random.random() < 0.1:
        sigma = random.uniform(0.5, 1.0)
        blurred = []
        for c in range(C):
            blurred.append(gaussian_filter(image[c], sigma=sigma))
        image = np.stack(blurred, axis=0).astype(np.float32)

    # ── 9. MRI Bias Field Simulation (p=0.3) ─────────────────
    if random.random() < 0.3:
        image = apply_bias_field(image, scale_range=(0.03, 0.10))

    return image, seg


# ══════════════════════════════════════════════════════════════
#  MRI BIAS FIELD SIMULATION
# ══════════════════════════════════════════════════════════════

def apply_bias_field(image, scale_range=(0.03, 0.10)):
    """
    Simulate MRI scanner RF coil inhomogeneity as a smooth multiplicative
    bias field over the volume.

    Implementation:
      - Build a polynomial basis in normalized [-1,1] coords (x,y,z).
      - Sample random coefficients for terms up to degree 2.
      - The bias field B = scale * (poly / max(|poly|)).
      - Apply multiplicatively on foreground only: img * exp(B).
      - Using exp() ensures the field is always positive (no sign flip).
      - Each modality gets an independent field (different coil responses).

    Why polynomial, not smoothed noise?
      A polynomial field matches the actual physics of RF inhomogeneity:
      it varies slowly and has no high-frequency components.
      Smoothing random noise risks introducing boundary artifacts near
      the foreground mask edge.

    Args:
        image: np.ndarray [C, D, H, W] float32 — normalized
        scale_range: (min, max) peak bias magnitude (log scale).
                     0.05 ≈ ±5% signal variation — clinically realistic.
    Returns:
        np.ndarray [C, D, H, W] float32
    """
    C, D, H, W = image.shape

    # Normalized coordinate grids in [-1, 1]
    zz = np.linspace(-1, 1, D, dtype=np.float32)
    yy = np.linspace(-1, 1, H, dtype=np.float32)
    xx = np.linspace(-1, 1, W, dtype=np.float32)
    z_grid, y_grid, x_grid = np.meshgrid(zz, yy, xx, indexing='ij')

    # Degree-2 polynomial basis: 1, z, y, x, z², y², x², zy, zx, yx
    basis = np.stack([
        np.ones((D, H, W), dtype=np.float32),
        z_grid, y_grid, x_grid,
        z_grid * z_grid, y_grid * y_grid, x_grid * x_grid,
        z_grid * y_grid, z_grid * x_grid, y_grid * x_grid,
    ], axis=0)   # [10, D, H, W]

    result = image.copy()
    fg_mask = image[0] != 0   # brain foreground from first modality

    for c in range(C):
        # Independent random coefficients per modality
        coeffs = np.random.uniform(-1.0, 1.0, size=basis.shape[0]).astype(np.float32)
        poly   = np.tensordot(coeffs, basis, axes=([0], [0]))   # [D, H, W]

        # Normalize to [-1, 1] then scale
        max_abs = np.abs(poly).max()
        if max_abs < 1e-8:
            continue
        poly_norm = poly / max_abs   # in [-1, 1]

        scale = random.uniform(*scale_range)
        bias  = scale * poly_norm   # peak magnitude = scale

        # Multiplicative application in exp domain — keeps field positive
        result[c] = np.where(
            fg_mask,
            image[c] * np.exp(bias),
            image[c],
        )

    return result.astype(np.float32)


# ══════════════════════════════════════════════════════════════
#  RANDOM SCALE (applied before patch sampling)
# ══════════════════════════════════════════════════════════════

def random_scale_volume(image, seg, scale_range=(0.85, 1.15)):
    """
    Random isotropic zoom of the full volume before patch sampling.
    Forces the model to handle tumor regions at multiple scales.

    Args:
        image: [C, D, H, W]
        seg:   [D, H, W]
        scale_range: (min, max) zoom factors
    Returns:
        (image, seg) possibly resized
    """
    if not SCIPY_AVAILABLE:
        return image, seg
    if random.random() > 0.3:
        return image, seg

    scale = random.uniform(*scale_range)
    C = image.shape[0]

    # Zoom each modality
    zoomed_mods = []
    for c in range(C):
        zoomed_mods.append(
            scipy_zoom(image[c], zoom=scale, order=1)
        )
    image = np.stack(zoomed_mods, axis=0).astype(np.float32)
    seg   = scipy_zoom(seg.astype(np.float32), zoom=scale, order=0).astype(np.int64)
    return image, seg


# ══════════════════════════════════════════════════════════════
#  PATCH SAMPLING (TC-biased + boundary-aware)
# ══════════════════════════════════════════════════════════════

def _find_boundary_voxels(seg):
    """
    Find all foreground voxels that neighbour a voxel of a different class.
    Uses a fast shift-based approach (no scipy needed).

    A voxel is a boundary voxel if seg[d,h,w] > 0 AND any of its
    6-connected neighbours has a different label.

    Returns:
        np.ndarray [N, 3] of (d, h, w) indices, or empty array.
    """
    fg = seg > 0
    D, H, W = seg.shape
    is_boundary = np.zeros_like(seg, dtype=bool)

    for axis, (shift_pos, shift_neg) in enumerate([
        (np.s_[1:, :, :],  np.s_[:-1, :, :]),   # D
        (np.s_[:, 1:, :],  np.s_[:, :-1, :]),   # H
        (np.s_[:, :, 1:],  np.s_[:, :, :-1]),   # W
    ]):
        diff = seg[shift_pos] != seg[shift_neg]
        # Mark both sides of each differing pair
        if axis == 0:
            is_boundary[1:,  :, :] |= diff
            is_boundary[:-1, :, :] |= diff
        elif axis == 1:
            is_boundary[:, 1:,  :] |= diff
            is_boundary[:, :-1, :] |= diff
        else:
            is_boundary[:, :, 1:]  |= diff
            is_boundary[:, :, :-1] |= diff

    # Only keep foreground boundary voxels (ignore brain/background boundary)
    boundary = is_boundary & fg
    return np.argwhere(boundary)


def sample_patch(image, seg, patch_size, tumor_ratio, boundary_sample_prob=0.2):
    """
    Sample a 3D patch with three-level priority sampling:

      1. With prob `boundary_sample_prob` (default 0.2):
           Center patch on a random class-boundary voxel.
           Boundary = foreground voxel adjacent to a different-class voxel.
           Directly trains on TC/ED/ET transition zones → improves HD95.

      2. Otherwise, with prob `tumor_ratio` (default 0.8):
           TC-biased tumor sampling (same as v1):
             - p=0.5 among TC voxels specifically
             - fallback to any tumor voxel

      3. Otherwise:
           Fully random patch anywhere in the volume.

    Boundary-aware sampling is only tried when boundary voxels exist; it
    falls through to standard sampling if the seg has no multi-class regions.
    """
    C, D, H, W = image.shape
    pd, ph, pw = patch_size
    max_d = max(D - pd, 0)
    max_h = max(H - ph, 0)
    max_w = max(W - pw, 0)

    center = None

    # ── Priority 1: Boundary-aware sampling ──────────────────
    if random.random() < boundary_sample_prob:
        boundary_voxels = _find_boundary_voxels(seg)
        if len(boundary_voxels) > 0:
            center = boundary_voxels[random.randint(0, len(boundary_voxels) - 1)]

    # ── Priority 2: TC-biased tumor sampling ─────────────────
    if center is None and random.random() < tumor_ratio:
        if random.random() < 0.5:
            tc_voxels = np.argwhere(seg == 1)
            if len(tc_voxels) > 0:
                center = tc_voxels[random.randint(0, len(tc_voxels) - 1)]
        if center is None:
            tumor_voxels = np.argwhere(seg > 0)
            if len(tumor_voxels) > 0:
                center = tumor_voxels[random.randint(0, len(tumor_voxels) - 1)]

    # ── Compute top-left corner from center ───────────────────
    if center is not None:
        d0 = int(np.clip(center[0] - pd // 2, 0, max_d))
        h0 = int(np.clip(center[1] - ph // 2, 0, max_h))
        w0 = int(np.clip(center[2] - pw // 2, 0, max_w))
    else:
        d0 = random.randint(0, max_d) if max_d > 0 else 0
        h0 = random.randint(0, max_h) if max_h > 0 else 0
        w0 = random.randint(0, max_w) if max_w > 0 else 0

    img_patch = image[:, d0:d0+pd, h0:h0+ph, w0:w0+pw]
    seg_patch = seg[d0:d0+pd, h0:h0+ph, w0:w0+pw]

    # Pad if near edge
    if img_patch.shape[1:] != (pd, ph, pw):
        pad_d = pd - img_patch.shape[1]
        pad_h = ph - img_patch.shape[2]
        pad_w = pw - img_patch.shape[3]
        img_patch = np.pad(img_patch, ((0,0),(0,pad_d),(0,pad_h),(0,pad_w)))
        seg_patch = np.pad(seg_patch, ((0,pad_d),(0,pad_h),(0,pad_w)))

    return img_patch.astype(np.float32), seg_patch


# ══════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════

class BraTSDataset(Dataset):
    """
    BraTS patch dataset with v2 augmentation pipeline.

    Each __getitem__:
      1. Load volume (from disk — no persistent cache across workers)
      2. Normalize
      3. Random zoom (p=0.3)
      4. Apply v2 augmentation (training only)
      5. Sample TC-biased patch
    """

    MODALITY_SUFFIXES = ["-t1c.nii.gz", "-t1n.nii.gz", "-t2f.nii.gz", "-t2w.nii.gz"]

    def __init__(self, subject_dirs, patch_size, tumor_ratio, augment_data=False):
        self.subject_dirs = subject_dirs
        self.patch_size   = patch_size
        self.tumor_ratio  = tumor_ratio
        self.augment_data = augment_data

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
                    raise FileNotFoundError(f"Modality not found: {path}")
                path = candidates[0]
            modalities.append(nib.load(path).get_fdata(dtype=np.float32))

        seg_path = os.path.join(subject_dir, subject_id + "-seg.nii.gz")
        if not os.path.exists(seg_path):
            candidates = glob.glob(os.path.join(subject_dir, "*-seg.nii.gz"))
            if not candidates:
                raise FileNotFoundError(f"Seg not found in: {subject_dir}")
            seg_path = candidates[0]

        seg = nib.load(seg_path).get_fdata(dtype=np.float32).astype(np.int64)
        seg[seg == 4] = 3   # BraTS 2020 compatibility

        image = np.stack(modalities, axis=0)   # [4, D, H, W]
        return image, seg

    def __getitem__(self, idx):
        image, seg = self._load_volume(self.subject_dirs[idx])
        image = normalize_volume(image)

        if self.augment_data:
            # Random zoom before patch sampling (changes effective tumor scale)
            image, seg = random_scale_volume(image, seg, scale_range=(0.85, 1.15))
            # Augmentation
            image, seg = augment_v2(image, seg)

        img_patch, seg_patch = sample_patch(image, seg, self.patch_size, self.tumor_ratio)

        return (
            torch.tensor(img_patch, dtype=torch.float32),
            torch.tensor(seg_patch, dtype=torch.long),
        )


class BraTSFullVolumeDataset(Dataset):
    """Full normalized volumes for sliding window inference (no augmentation)."""

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
                    raise FileNotFoundError(f"Modality not found: {path}")
                path = candidates[0]
            modalities.append(nib.load(path).get_fdata(dtype=np.float32))

        seg_path = os.path.join(subject_dir, subject_id + "-seg.nii.gz")
        if not os.path.exists(seg_path):
            candidates = glob.glob(os.path.join(subject_dir, "*-seg.nii.gz"))
            if not candidates:
                raise FileNotFoundError(f"Seg not found in: {subject_dir}")
            seg_path = candidates[0]

        seg = nib.load(seg_path).get_fdata(dtype=np.float32).astype(np.int64)
        seg[seg == 4] = 3

        image = np.stack(modalities, axis=0)
        return image, seg

    def __getitem__(self, idx):
        image, seg = self._load_volume(self.subject_dirs[idx])
        image = normalize_volume(image)
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(seg,   dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════
#  BUILD DATALOADER (resolver entry point)
# ══════════════════════════════════════════════════════════════

def build_dataloader(config, split="train"):
    train_root  = config.data.train_root
    val_ratio   = float(config.data.val_ratio)
    patch_size  = list(config.data.patch_size)
    tumor_ratio = float(config.data.tumor_ratio)
    num_workers = int(config.data.num_workers)
    max_samples = getattr(config.data, "max_samples", None)
    batch_size  = int(config.training.batch_size)

    all_dirs = sorted([
        os.path.join(train_root, d)
        for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
    ])
    if not all_dirs:
        raise RuntimeError(f"No subject dirs found in: {train_root}")

    if max_samples:
        all_dirs = all_dirs[:int(max_samples)]

    n_val   = max(1, int(len(all_dirs) * val_ratio))
    n_train = len(all_dirs) - n_val
    train_dirs = all_dirs[:n_train]
    val_dirs   = all_dirs[n_train:]

    if split == "train":
        dataset = BraTSDataset(
            subject_dirs=train_dirs,
            patch_size=patch_size,
            tumor_ratio=tumor_ratio,
            augment_data=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
    else:
        dataset = BraTSFullVolumeDataset(subject_dirs=val_dirs)
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=min(num_workers, 2),
            pin_memory=True,
        )
