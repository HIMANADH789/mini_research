import torch
import torch.nn as nn


def crop(enc, dec):
    _, _, D, H, W = dec.shape
    return enc[:, :, :D, :H, :W]


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, 1),
            nn.InstanceNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, 1),
            nn.InstanceNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, 1),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_c = config.model.in_channels
        out_c = config.model.out_channels
        base = config.model.base_channels

        self.pool = nn.MaxPool3d(2)

        self.enc1 = DoubleConv(in_c, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)

        self.bottleneck = DoubleConv(base*4, base*8)

        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, 2)
        self.att3 = AttentionBlock(base*4, base*4, base*2)
        self.dec3 = DoubleConv(base*8, base*4)

        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, 2)
        self.att2 = AttentionBlock(base*2, base*2, base)
        self.dec2 = DoubleConv(base*4, base*2)

        self.up1 = nn.ConvTranspose3d(base*2, base, 2, 2)
        self.att1 = AttentionBlock(base, base, base//2)
        self.dec1 = DoubleConv(base*2, base)

        self.final = nn.Conv3d(base, out_c, 1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        e3 = crop(e3, d3)
        e3 = self.att3(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2 = crop(e2, d2)
        e2 = self.att2(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1 = crop(e1, d1)
        e1 = self.att1(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)