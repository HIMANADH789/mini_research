import torch
import torch.nn as nn


def crop(enc, dec):
    _, _, D, H, W = dec.shape
    return enc[:, :, :D, :H, :W]


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetPP(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_c = config.model.in_channels
        out_c = config.model.out_channels
        base = config.model.base_channels

        self.pool = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.conv0_0 = ConvBlock(in_c, base)
        self.conv1_0 = ConvBlock(base, base*2)
        self.conv2_0 = ConvBlock(base*2, base*4)

        self.conv0_1 = ConvBlock(base + base*2, base)
        self.conv1_1 = ConvBlock(base*2 + base*4, base*2)

        self.conv0_2 = ConvBlock(base*2 + base, base)

        self.final = nn.Conv3d(base, out_c, 1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))

        x1_1 = self.conv1_1(torch.cat([
            x1_0,
            crop(self.up(x2_0), x1_0)
        ], dim=1))

        x0_1 = self.conv0_1(torch.cat([
            x0_0,
            crop(self.up(x1_0), x0_0)
        ], dim=1))

        x0_2 = self.conv0_2(torch.cat([
            x0_1,
            crop(self.up(x1_1), x0_1)
        ], dim=1))

        return self.final(x0_2)