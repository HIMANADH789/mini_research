"""
SwinUNETR — 3D Swin Transformer U-Net for Brain Tumor Segmentation
====================================================================
Architecture (Hatamizadeh et al., CVPR 2022 — from-scratch implementation):

  Encoder: 3D Swin Transformer with 4 stages + hierarchical feature maps
  Decoder: UNet-style with residual skip connections and deep supervision

  Swin Transformer key properties:
    - Window-based self-attention: each token attends only within a local ws³ window
    - Shifted windows (SW-MSA): alternate layers shift windows by ws//2 for cross-window
      information flow without extra compute
    - Relative position bias: learnable 3D position encoding within each window
    - Stochastic depth (drop_path): regularizes deep transformer stacks
    - Patch merging: 2x spatial downsampling between stages

  Architecture dimensions (feature_size=48):
    PatchEmbed  →  64³ × 48
    Stage 0     →  64³ × 48    (2 blocks, 3 heads)   [skip s0]
    PatchMerge  →  32³ × 96
    Stage 1     →  32³ × 96    (2 blocks, 6 heads)   [skip s1]
    PatchMerge  →  16³ × 192
    Stage 2     →  16³ × 192   (2 blocks, 12 heads)  [skip s2]
    PatchMerge  →   8³ × 384
    Stage 3     →   8³ × 384   (2 blocks, 24 heads)  [bottleneck]

    enc0  →  128³ × 48   (raw input through shallow conv, for finest skip)
    dec4  →   16³ × 192  (up(bottleneck) + s2)
    dec3  →   32³ × 96   (up(dec4) + s1)
    dec2  →   64³ × 48   (up(dec3) + s0)
    dec1  →  128³ × 32   (up(dec2) + enc0)
    out   →  128³ × C    (1x1 conv)

  Deep supervision:
    ds4 head from dec4 → 4 classes (weight 0.5)
    ds3 head from dec3 → 4 classes (weight 0.25)
    Main head from dec1 → 4 classes (weight 1.0)
    Returned as list [main, ds4, ds3] during training, main only during eval.

Config fields:
  model.in_channels:    4
  model.out_channels:   4
  model.feature_size:   48   (embed_dim)
  model.window_size:    7
  model.depths:         [2, 2, 2, 2]
  model.num_heads:      [3, 6, 12, 24]
  model.drop_path_rate: 0.2
  model.mlp_ratio:      4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np


# ══════════════════════════════════════════════════════════════
#  UTILITY — Drop Path (Stochastic Depth)
# ══════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    """Per-sample stochastic depth regularization."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


# ══════════════════════════════════════════════════════════════
#  UTILITY — Window Partition / Reverse
# ══════════════════════════════════════════════════════════════

def window_partition(x, window_size):
    """
    Partition 3D feature map into non-overlapping windows.
    Args:
        x:           [B, D, H, W, C]
        window_size: int
    Returns:
        windows:     [num_windows*B, ws, ws, ws, C]
    """
    B, D, H, W, C = x.shape
    ws = window_size
    x = x.view(B, D // ws, ws, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, ws, ws, ws, C)


def window_reverse(windows, window_size, D, H, W):
    """
    Reverse window partition back to 3D feature map.
    Args:
        windows:     [num_windows*B, ws, ws, ws, C]
    Returns:
        x:           [B, D, H, W, C]
    """
    ws = window_size
    B = int(windows.shape[0] / (D * H * W / ws ** 3))
    x = windows.view(B, D // ws, H // ws, W // ws, ws, ws, ws, -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)


def pad_to_window(x, window_size):
    """
    Pad spatial dims of x:[B,D,H,W,C] to be divisible by window_size.
    Returns padded x and (pad_d, pad_h, pad_w) for unpadding.
    """
    _, D, H, W, _ = x.shape
    ws = window_size
    pad_d = (ws - D % ws) % ws
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    return x, (pad_d, pad_h, pad_w)


def compute_attn_mask(D, H, W, window_size, shift_size, device):
    """
    Compute attention mask for SW-MSA (shifted window attention).
    Tokens from different cyclic-shift regions are masked with -100.
    Returns: [nW, ws³, ws³]
    """
    ws, ss = window_size, shift_size
    img_mask = torch.zeros((1, D, H, W, 1), device=device)

    d_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
    h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
    w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))

    cnt = 0
    for d in d_slices:
        for h in h_slices:
            for w in w_slices:
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, ws)       # [nW, ws, ws, ws, 1]
    mask_windows = mask_windows.view(-1, ws ** 3)       # [nW, ws³]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, ws³, ws³]
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


# ══════════════════════════════════════════════════════════════
#  WINDOW ATTENTION
# ══════════════════════════════════════════════════════════════

class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-head Self-Attention with relative position bias.
    Operates on flattened window tokens: [num_windows*B, ws³, C]
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim        = dim
        self.ws         = window_size
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = head_dim ** -0.5

        # Relative position bias table: [(2*ws-1)³, num_heads]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 3, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position index [ws³, ws³]
        coords_d = torch.arange(window_size)
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))  # [3, ws, ws, ws]
        coords_flat = torch.flatten(grid, 1)                                              # [3, ws³]
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]                          # [3, ws³, ws³]
        rel = rel.permute(1, 2, 0).contiguous()                                          # [ws³, ws³, 3]
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 2] += window_size - 1
        rel[:, :, 0] *= (2 * window_size - 1) ** 2
        rel[:, :, 1] *= (2 * window_size - 1)
        self.register_buffer("relative_position_index", rel.sum(-1))   # [ws³, ws³]

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x:    [nW*B, ws³, C]
            mask: [nW, ws³, ws³] or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # [3, B_, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)      # [B_, heads, N, N]

        # Add relative position bias
        ws3 = self.ws ** 3
        rpe = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rpe = rpe.view(ws3, ws3, -1).permute(2, 0, 1).contiguous()  # [heads, ws³, ws³]
        attn = attn + rpe.unsqueeze(0)

        # Apply SW-MSA mask
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ══════════════════════════════════════════════════════════════
#  SWIN TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════

class SwinTransformerBlock3D(nn.Module):
    """
    Single 3D Swin Transformer Block.
    Even index → W-MSA (no shift), Odd index → SW-MSA (shift by ws//2).

    Structure: LN → (W/SW)-MSA → drop_path + residual → LN → MLP → drop_path + residual
    """

    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim        = dim
        self.ws         = window_size
        self.ss         = shift_size
        self.num_heads  = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, attn_mask):
        """
        Args:
            x:         [B, D, H, W, C]
            attn_mask: precomputed SW-MSA mask or None
        """
        B, D, H, W, C = x.shape
        ws, ss = self.ws, self.ss

        # Pad to window multiple
        x, (pad_d, pad_h, pad_w) = pad_to_window(x, ws)
        _, Dp, Hp, Wp, _ = x.shape

        # ── Attention branch ─────────────────────────────────
        shortcut = x
        x = self.norm1(x)

        # Cyclic shift for SW-MSA
        if ss > 0:
            x_shifted = torch.roll(x, shifts=(-ss, -ss, -ss), dims=(1, 2, 3))
        else:
            x_shifted = x

        # Window partition
        x_windows = window_partition(x_shifted, ws)         # [nW*B, ws, ws, ws, C]
        x_windows = x_windows.view(-1, ws ** 3, C)          # [nW*B, ws³, C]

        # Window attention
        attn_out = self.attn(x_windows, mask=attn_mask)

        # Reverse windows
        attn_out = attn_out.view(-1, ws, ws, ws, C)
        attn_out = window_reverse(attn_out, ws, Dp, Hp, Wp)  # [B, Dp, Hp, Wp, C]

        # Reverse cyclic shift
        if ss > 0:
            attn_out = torch.roll(attn_out, shifts=(ss, ss, ss), dims=(1, 2, 3))

        # Remove padding
        if pad_d or pad_h or pad_w:
            attn_out = attn_out[:, :D, :H, :W, :].contiguous()

        x = shortcut[:, :D, :H, :W, :] + self.drop_path(attn_out)

        # ── MLP branch ───────────────────────────────────────
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════
#  BASIC LAYER (One Encoder Stage)
# ══════════════════════════════════════════════════════════════

class BasicLayer3D(nn.Module):
    """
    A sequence of SwinTransformerBlock3D forming one encoder stage.
    Alternates W-MSA and SW-MSA blocks.
    Handles attention mask caching per resolution.
    """

    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio,
                 qkv_bias, drop, attn_drop, drop_path):
        super().__init__()
        self.ws = window_size

        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])

        # Cache masks per (D, H, W)
        self._mask_cache = {}

    def _get_mask(self, D, H, W, shift_size, device):
        if shift_size == 0:
            return None
        key = (D, H, W)
        if key not in self._mask_cache:
            # Pad to window multiple for mask computation
            ws = self.ws
            Dp = int(np.ceil(D / ws)) * ws
            Hp = int(np.ceil(H / ws)) * ws
            Wp = int(np.ceil(W / ws)) * ws
            mask = compute_attn_mask(Dp, Hp, Wp, ws, shift_size, device)
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(self, x):
        """x: [B, D, H, W, C]"""
        B, D, H, W, C = x.shape
        for blk in self.blocks:
            mask = self._get_mask(D, H, W, blk.ss, x.device)
            x = blk(x, mask)
        return x


# ══════════════════════════════════════════════════════════════
#  PATCH MERGING (Spatial Downsampling 2×)
# ══════════════════════════════════════════════════════════════

class PatchMerging3D(nn.Module):
    """
    Downsample by 2× via 2×2×2 neighbor concatenation + linear projection.
    [B, D, H, W, C] → [B, D/2, H/2, W/2, 2C]
    """

    def __init__(self, dim):
        super().__init__()
        self.norm       = nn.LayerNorm(8 * dim)
        self.reduction  = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x):
        B, D, H, W, C = x.shape
        # Pad to even spatial dims
        if D % 2 != 0: x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        if H % 2 != 0: x = F.pad(x, (0, 0, 0, 0, 0, 1))
        if W % 2 != 0: x = F.pad(x, (0, 0, 0, 1))

        # Collect 2×2×2 neighbors
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)  # [B, D/2, H/2, W/2, 8C]
        return self.reduction(self.norm(x))


# ══════════════════════════════════════════════════════════════
#  PATCH EMBED
# ══════════════════════════════════════════════════════════════

class PatchEmbed3D(nn.Module):
    """
    Convert [B, C_in, D, H, W] to [B, D/ps, H/ps, W/ps, embed_dim]
    via a single Conv3d with kernel=patch_size, stride=patch_size.
    """

    def __init__(self, in_channels, embed_dim, patch_size=2):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                          # [B, embed_dim, D/ps, H/ps, W/ps]
        x = x.permute(0, 2, 3, 4, 1).contiguous() # [B, D/ps, H/ps, W/ps, embed_dim]
        return self.norm(x)


# ══════════════════════════════════════════════════════════════
#  ENCODER PROJECTION BLOCK (skip connection projectors)
# ══════════════════════════════════════════════════════════════

class EncoderBlock(nn.Module):
    """
    Two-layer conv block to project encoder features for skip connections.
    Converts [B, C, D, H, W] feature maps with residual shortcut.
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.shortcut = nn.Conv3d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


# ══════════════════════════════════════════════════════════════
#  DECODER BLOCK
# ══════════════════════════════════════════════════════════════

class DecoderBlock(nn.Module):
    """
    Upsample (2×) + skip concat + two-conv residual block.
    """

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_c, in_c // 2, kernel_size=2, stride=2)
        merged    = in_c // 2 + skip_c
        self.conv = nn.Sequential(
            nn.Conv3d(merged, out_c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.shortcut = nn.Conv3d(merged, out_c, 1, bias=False)

    def forward(self, x, skip):
        x    = self.up(x)
        D, H, W = x.shape[2:]
        skip = skip[:, :, :D, :H, :W]            # align spatial dims
        x    = torch.cat([x, skip], dim=1)
        return self.conv(x) + self.shortcut(x)


# ══════════════════════════════════════════════════════════════
#  SWINUNETR — Main Model
# ══════════════════════════════════════════════════════════════

class SwinUNETR(nn.Module):
    """
    3D SwinUNETR for multi-class brain tumor segmentation.

    Config fields consumed:
      config.model.in_channels:    4    (MRI modalities)
      config.model.out_channels:   4    (BG, TC, ED, ET)
      config.model.feature_size:   48   (encoder base dim)
      config.model.window_size:    7
      config.model.depths:         [2, 2, 2, 2]
      config.model.num_heads:      [3, 6, 12, 24]
      config.model.drop_path_rate: 0.2
      config.model.mlp_ratio:      4.0

    During training: returns [main, ds1, ds2]
      main: [B, out_c, D, H, W]  — full-resolution output
      ds1:  [B, out_c, D/8, H/8, W/8]  — deep supervision from bottleneck up
      ds2:  [B, out_c, D/4, H/4, W/4]  — deep supervision from next decoder level
    During eval: returns main only.
    """

    def __init__(self, config):
        super().__init__()

        in_c  = config.model.in_channels
        out_c = config.model.out_channels
        fs    = getattr(config.model, "feature_size",   48)
        ws    = getattr(config.model, "window_size",    7)
        depths = list(getattr(config.model, "depths",  [2, 2, 2, 2]))
        heads  = list(getattr(config.model, "num_heads",[3, 6, 12, 24]))
        dpr    = float(getattr(config.model, "drop_path_rate", 0.2))
        mlp_r  = float(getattr(config.model, "mlp_ratio",      4.0))

        # Total depth for stochastic depth schedule
        total_depth = sum(depths)
        dpr_list    = [x.item() for x in torch.linspace(0, dpr, total_depth)]

        # ── Raw input skip projection (128³) ──────────────────
        self.enc0 = EncoderBlock(in_c, fs)           # 128³ × fs

        # ── Patch Embedding (stride 2) ────────────────────────
        # 128³ → 64³ × fs
        self.patch_embed = PatchEmbed3D(in_c, fs, patch_size=2)

        # ── Encoder Stages ────────────────────────────────────
        dp = 0
        self.stage0 = BasicLayer3D(
            dim=fs, depth=depths[0], num_heads=heads[0],
            window_size=ws, mlp_ratio=mlp_r, qkv_bias=True,
            drop=0.0, attn_drop=0.0,
            drop_path=dpr_list[dp: dp + depths[0]],
        )
        dp += depths[0]

        self.merge0 = PatchMerging3D(fs)             # 64³ → 32³, fs→2fs

        self.stage1 = BasicLayer3D(
            dim=2*fs, depth=depths[1], num_heads=heads[1],
            window_size=ws, mlp_ratio=mlp_r, qkv_bias=True,
            drop=0.0, attn_drop=0.0,
            drop_path=dpr_list[dp: dp + depths[1]],
        )
        dp += depths[1]

        self.merge1 = PatchMerging3D(2*fs)           # 32³ → 16³, 2fs→4fs

        self.stage2 = BasicLayer3D(
            dim=4*fs, depth=depths[2], num_heads=heads[2],
            window_size=ws, mlp_ratio=mlp_r, qkv_bias=True,
            drop=0.0, attn_drop=0.0,
            drop_path=dpr_list[dp: dp + depths[2]],
        )
        dp += depths[2]

        self.merge2 = PatchMerging3D(4*fs)           # 16³ → 8³, 4fs→8fs

        self.stage3 = BasicLayer3D(
            dim=8*fs, depth=depths[3], num_heads=heads[3],
            window_size=ws, mlp_ratio=mlp_r, qkv_bias=True,
            drop=0.0, attn_drop=0.0,
            drop_path=dpr_list[dp: dp + depths[3]],
        )

        # ── Skip Connection Projectors ─────────────────────────
        # Project transformer BDHWC outputs → standard BCDHW for decoder
        self.proj_s0 = nn.Sequential(
            nn.Linear(fs, fs),
            nn.LayerNorm(fs),
        )
        self.proj_s1 = nn.Sequential(
            nn.Linear(2*fs, 2*fs),
            nn.LayerNorm(2*fs),
        )
        self.proj_s2 = nn.Sequential(
            nn.Linear(4*fs, 4*fs),
            nn.LayerNorm(4*fs),
        )
        self.proj_bot = nn.Sequential(
            nn.Linear(8*fs, 8*fs),
            nn.LayerNorm(8*fs),
        )

        # ── Decoder ───────────────────────────────────────────
        # dec4: 8³×8fs up → 16³, concat 4fs skip → 4fs
        self.dec4 = DecoderBlock(8*fs,  4*fs,  4*fs)
        # dec3: 16³×4fs up → 32³, concat 2fs skip → 2fs
        self.dec3 = DecoderBlock(4*fs,  2*fs,  2*fs)
        # dec2: 32³×2fs up → 64³, concat fs skip → fs
        self.dec2 = DecoderBlock(2*fs,  fs,    fs)
        # dec1: 64³×fs up → 128³, concat enc0 fs skip → fs//2
        self.dec1 = DecoderBlock(fs,    fs,    fs // 2)

        # ── Output & Deep Supervision Heads ───────────────────
        self.out_head   = nn.Conv3d(fs // 2, out_c, kernel_size=1)
        self.ds_head4   = nn.Conv3d(4*fs,    out_c, kernel_size=1)  # from dec4 (16³)
        self.ds_head3   = nn.Conv3d(2*fs,    out_c, kernel_size=1)  # from dec3 (32³)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm,)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _to_channels_first(x):
        """[B, D, H, W, C] → [B, C, D, H, W]"""
        return x.permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x):
        # ── Raw skip (full resolution) ─────────────────────────
        skip_enc0 = self.enc0(x)                                # [B, fs, 128, 128, 128]

        # ── Patch Embed ──────────────────────────────────────
        x_embed = self.patch_embed(x)                           # [B, 64, 64, 64, fs]

        # ── Encoder ──────────────────────────────────────────
        s0 = self.stage0(x_embed)                               # [B, 64, 64, 64, fs]
        s1 = self.stage1(self.merge0(s0))                       # [B, 32, 32, 32, 2fs]
        s2 = self.stage2(self.merge1(s1))                       # [B, 16, 16, 16, 4fs]
        bot = self.stage3(self.merge2(s2))                      # [B, 8, 8, 8, 8fs]

        # ── Project to [B, C, D, H, W] for decoder ───────────
        skip_s0  = self._to_channels_first(self.proj_s0(s0))   # [B, fs,   64, 64, 64]
        skip_s1  = self._to_channels_first(self.proj_s1(s1))   # [B, 2fs,  32, 32, 32]
        skip_s2  = self._to_channels_first(self.proj_s2(s2))   # [B, 4fs,  16, 16, 16]
        bottleneck = self._to_channels_first(self.proj_bot(bot)) # [B, 8fs,   8,  8,  8]

        # ── Decoder ──────────────────────────────────────────
        d4 = self.dec4(bottleneck, skip_s2)                     # [B, 4fs,  16, 16, 16]
        d3 = self.dec3(d4, skip_s1)                             # [B, 2fs,  32, 32, 32]
        d2 = self.dec2(d3, skip_s0)                             # [B, fs,   64, 64, 64]
        d1 = self.dec1(d2, skip_enc0)                           # [B, fs/2, 128, 128, 128]

        # ── Output ───────────────────────────────────────────
        main = self.out_head(d1)                                 # [B, out_c, 128, 128, 128]

        if self.training:
            ds1 = self.ds_head4(d4)                              # [B, out_c, 16, 16, 16]
            ds2 = self.ds_head3(d3)                              # [B, out_c, 32, 32, 32]
            return [main, ds1, ds2]

        return main
