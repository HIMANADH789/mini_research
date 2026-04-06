"""
Residual UNet (ResUNet) for 3D Brain Tumor Segmentation
========================================================
Architecture:
  - 5-level encoder-decoder (deeper than UNet3D's 4 levels)
  - Residual blocks in both encoder and decoder paths
    Residual connection = identity (or 1x1 conv if channels change)
    prevents vanishing gradients, enables stable deeper training
  - ConvTranspose3d for learned upsampling
  - InstanceNorm3d + LeakyReLU throughout
  - base_channels=32 → [32, 64, 128, 256, 512] across 5 levels

Channel progression:
  Encoder:    in_c → 32 → 64 → 128 → 256
  Bottleneck: 256 → 512
  Decoder:    512+256 → 256 → 256+128 → 128 → 128+64 → 64 → 64+32 → 32
  Output:     32 → out_c

Why residual connections:
  Gradients flow directly through the identity shortcut, avoiding
  the vanishing gradient problem in deeper 3D networks. This allows
  the 5-level architecture to be trained stably without degradation.
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────
# RESIDUAL BLOCK
# ──────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    3D Residual block: two 3x3x3 convolutions with a skip connection.

    If in_c != out_c, the skip connection uses a 1x1x1 projection conv
    to match dimensions. Otherwise identity shortcut.
    """

    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_c, affine=True)

        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_c, affine=True)

        self.act = nn.LeakyReLU(0.01, inplace=True)

        # Projection shortcut if channel dimensions differ
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, bias=False),
                nn.InstanceNorm3d(out_c, affine=True),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act(out + residual)

        return out


# ──────────────────────────────────────────────────────────
# DECODER BLOCK
# ──────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Upsample via ConvTranspose3d then concatenate skip connection,
    followed by a ResBlock.
    """

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up  = nn.ConvTranspose3d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.res = ResBlock(in_c // 2 + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        # Crop skip to match x spatial dims (handles odd sizes)
        _, _, D, H, W = x.shape
        skip = skip[:, :, :D, :H, :W]
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


# ──────────────────────────────────────────────────────────
# RESUNET
# ──────────────────────────────────────────────────────────

class ResUNet(nn.Module):
    """
    5-level 3D Residual UNet for multi-class segmentation.

    Config fields used:
      config.model.in_channels   (int) — number of input modalities (4 for BraTS)
      config.model.out_channels  (int) — number of output classes (4 for BraTS)
      config.model.base_channels (int) — base feature width (default 32)
    """

    def __init__(self, config):
        super().__init__()

        in_c  = config.model.in_channels
        out_c = config.model.out_channels
        b     = config.model.base_channels   # 32

        self.pool = nn.MaxPool3d(2)

        # ── Encoder ──────────────────────────────────────────
        self.enc1 = ResBlock(in_c,   b)       # [B, 32,  D,    H,    W   ]
        self.enc2 = ResBlock(b,      b*2)     # [B, 64,  D/2,  H/2,  W/2 ]
        self.enc3 = ResBlock(b*2,    b*4)     # [B, 128, D/4,  H/4,  W/4 ]
        self.enc4 = ResBlock(b*4,    b*8)     # [B, 256, D/8,  H/8,  W/8 ]

        # ── Bottleneck ───────────────────────────────────────
        self.bottleneck = ResBlock(b*8, b*16) # [B, 512, D/16, H/16, W/16]

        # ── Decoder ──────────────────────────────────────────
        # in_c=512, up→256, cat skip(256) → ResBlock(512→256)
        self.dec4 = DecoderBlock(b*16, b*8,  b*8)
        # in_c=256, up→128, cat skip(128) → ResBlock(256→128)
        self.dec3 = DecoderBlock(b*8,  b*4,  b*4)
        # in_c=128, up→64,  cat skip(64)  → ResBlock(128→64)
        self.dec2 = DecoderBlock(b*4,  b*2,  b*2)
        # in_c=64,  up→32,  cat skip(32)  → ResBlock(64→32)
        self.dec1 = DecoderBlock(b*2,  b,    b)

        # ── Output ───────────────────────────────────────────
        self.final = nn.Conv3d(b, out_c, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))

        # Decoder (each block handles upsample + skip cat + ResBlock)
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.final(d1)
