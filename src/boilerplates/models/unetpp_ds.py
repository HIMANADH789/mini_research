"""
UNet++ with Deep Supervision (4-level) for 3D Brain Tumor Segmentation
=======================================================================
Architecture:
  Full 4-level UNet++ with dense nested skip connections and
  deep supervision on intermediate output nodes.

  Node notation:  x_{level}_{column}
    level 0 = finest resolution (full size)
    level 3 = coarsest resolution (most downsampled)

  Grid:
    x0,0  x0,1  x0,2  x0,3   ← output row (all produce predictions)
    x1,0  x1,1  x1,2
    x2,0  x2,1
    x3,0                      ← bottleneck

  Dense skip connections:
    x_{i,j} receives: x_{i,0..j-1}  (all previous nodes at same level)
                     + up(x_{i+1,j-1})  (upsampled from level below)

Deep Supervision:
  During training: forward() returns [x0,3, x0,2, x0,1]
    (main output first, then auxiliary outputs in coarser order)
  During inference: forward() returns x0,3 only

  Loss weights applied in trainer: [1.0, 0.5, 0.25] for [main, aux1, aux2]

Why deep supervision is mandatory for UNet++:
  Without auxiliary outputs at intermediate nodes, only the final loss
  gradient reaches x0,1 and x0,2 through long paths. Deep supervision
  provides direct gradient to each node, preventing the dense paths
  from becoming degenerate feature aggregators.

Channel table (base=32):
  x0,*: 32  |  x1,*: 64  |  x2,*: 128  |  x3,0: 256 (bottleneck)

  x0,1 input: x0,0(32) + up(x1,0)(64)           = 96
  x1,1 input: x1,0(64) + up(x2,0)(128)           = 192
  x2,1 input: x2,0(128) + up(x3,0)(256)          = 384
  x0,2 input: x0,0(32) + x0,1(32) + up(x1,1)(64) = 128
  x1,2 input: x1,0(64) + x1,1(64) + up(x2,1)(128) = 256
  x0,3 input: x0,0(32)+x0,1(32)+x0,2(32)+up(x1,2)(64) = 160
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────
# CONV BLOCK
# ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two 3x3x3 convolutions with InstanceNorm3d and LeakyReLU."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def crop_to(src, ref):
    """Crop src spatial dims to match ref."""
    _, _, D, H, W = ref.shape
    return src[:, :, :D, :H, :W]


# ──────────────────────────────────────────────────────────
# UNET++ WITH DEEP SUPERVISION
# ──────────────────────────────────────────────────────────

class UNetPPDS(nn.Module):
    """
    4-level 3D UNet++ with Deep Supervision.

    Config fields used:
      config.model.in_channels   (int)
      config.model.out_channels  (int)
      config.model.base_channels (int, default 32)

    Returns during training (model.training=True):
      list [main_out, aux1_out, aux2_out]
        main_out: logits from x0,3  [B, out_c, D, H, W]
        aux1_out: logits from x0,2  [B, out_c, D, H, W]
        aux2_out: logits from x0,1  [B, out_c, D, H, W]

    Returns during eval (model.training=False):
      main_out only  [B, out_c, D, H, W]
    """

    def __init__(self, config):
        super().__init__()

        in_c  = config.model.in_channels
        out_c = config.model.out_channels
        b     = config.model.base_channels   # 32

        self.pool = nn.MaxPool3d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        # ── Encoder nodes (column 0) ─────────────────────────
        self.x0_0 = ConvBlock(in_c, b)        # [B, 32,  D,    H,    W   ]
        self.x1_0 = ConvBlock(b,    b*2)      # [B, 64,  D/2,  H/2,  W/2 ]
        self.x2_0 = ConvBlock(b*2,  b*4)      # [B, 128, D/4,  H/4,  W/4 ]
        self.x3_0 = ConvBlock(b*4,  b*8)      # [B, 256, D/8,  H/8,  W/8 ] bottleneck

        # ── Dense nodes (columns 1, 2, 3) ────────────────────
        # Column 1
        self.x0_1 = ConvBlock(b   + b*2,  b)     # 32+64  = 96  → 32
        self.x1_1 = ConvBlock(b*2 + b*4,  b*2)   # 64+128 = 192 → 64
        self.x2_1 = ConvBlock(b*4 + b*8,  b*4)   # 128+256 = 384 → 128

        # Column 2
        self.x0_2 = ConvBlock(b + b + b*2,      b)   # 32+32+64  = 128 → 32
        self.x1_2 = ConvBlock(b*2 + b*2 + b*4,  b*2) # 64+64+128 = 256 → 64

        # Column 3
        self.x0_3 = ConvBlock(b + b + b + b*2,  b)   # 32+32+32+64 = 160 → 32

        # ── Output heads ─────────────────────────────────────
        self.out3 = nn.Conv3d(b, out_c, 1)   # main output (x0,3)
        self.out2 = nn.Conv3d(b, out_c, 1)   # aux  output (x0,2)
        self.out1 = nn.Conv3d(b, out_c, 1)   # aux  output (x0,1)

    def forward(self, x):
        # ── Encode ───────────────────────────────────────────
        f0_0 = self.x0_0(x)
        f1_0 = self.x1_0(self.pool(f0_0))
        f2_0 = self.x2_0(self.pool(f1_0))
        f3_0 = self.x3_0(self.pool(f2_0))

        # ── Column 1 ─────────────────────────────────────────
        # x0,1: [x0,0  +  up(x1,0)]
        up_x1_0 = crop_to(self.up(f1_0), f0_0)
        f0_1 = self.x0_1(torch.cat([f0_0, up_x1_0], dim=1))

        # x1,1: [x1,0  +  up(x2,0)]
        up_x2_0 = crop_to(self.up(f2_0), f1_0)
        f1_1 = self.x1_1(torch.cat([f1_0, up_x2_0], dim=1))

        # x2,1: [x2,0  +  up(x3,0)]
        up_x3_0 = crop_to(self.up(f3_0), f2_0)
        f2_1 = self.x2_1(torch.cat([f2_0, up_x3_0], dim=1))

        # ── Column 2 ─────────────────────────────────────────
        # x0,2: [x0,0  +  x0,1  +  up(x1,1)]
        up_x1_1 = crop_to(self.up(f1_1), f0_0)
        f0_2 = self.x0_2(torch.cat([f0_0, f0_1, up_x1_1], dim=1))

        # x1,2: [x1,0  +  x1,1  +  up(x2,1)]
        up_x2_1 = crop_to(self.up(f2_1), f1_0)
        f1_2 = self.x1_2(torch.cat([f1_0, f1_1, up_x2_1], dim=1))

        # ── Column 3 (final output row) ───────────────────────
        # x0,3: [x0,0  +  x0,1  +  x0,2  +  up(x1,2)]
        up_x1_2 = crop_to(self.up(f1_2), f0_0)
        f0_3 = self.x0_3(torch.cat([f0_0, f0_1, f0_2, up_x1_2], dim=1))

        # ── Output heads ─────────────────────────────────────
        main = self.out3(f0_3)

        if self.training:
            # Return all supervision outputs for deep supervision loss
            aux1 = self.out2(f0_2)
            aux2 = self.out1(f0_1)
            return [main, aux1, aux2]

        return main
