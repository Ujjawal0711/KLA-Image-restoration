"""U-Net baseline for joint denoise + 2x super-resolution.

Design decisions, in order of how much they matter:

1. **All work happens at LR; upsample once at the very end via PixelShuffle.**
   Restoring at HR would cost 4x the compute and activation memory at every layer.
   With 8 GB of VRAM that is the difference between training and not training.
   PixelShuffle is preferred over transposed convolution because it cannot produce
   checkerboard artifacts — it is a pure reshape of channels into space.

2. **Global residual from a bilinear upsample of the input.** Speckle is unbiased
   (E[degraded] = clean), so a plain upsample is already a decent estimate of the
   target. Predicting only the correction means the network never has to learn the
   identity, which measurably speeds up early training.

3. **Fully convolutional, no fixed sizes.** Required: the same weights must handle
   128->256 and 256->512. No linear layers, no global pooling, no hard-coded shapes.

4. **No BatchNorm.** Restoration networks are consistently better without it — it
   couples samples in a batch and injects train/eval discrepancy for a task where
   absolute intensity carries signal. NAFNet and most modern SR nets drop it too.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


def _stack(ch: int, n: int) -> nn.Sequential:
    return nn.Sequential(*[ResBlock(ch) for _ in range(n)])


class UNetSR(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 48, levels: int = 4,
                 blocks: int = 2, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.levels = levels

        chs = [base * (2 ** i) for i in range(levels)]  # e.g. 48, 96, 192, 384
        self.head = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(levels - 1):
            self.encoders.append(_stack(chs[i], blocks))
            self.downs.append(nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1))

        self.middle = _stack(chs[-1], blocks + 1)

        self.ups = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(levels - 1)):
            self.ups.append(nn.Conv2d(chs[i + 1], chs[i] * 4, 1))  # 1x1 then shuffle
            self.fuse.append(nn.Conv2d(chs[i] * 2, chs[i], 1))
            self.decoders.append(_stack(chs[i], blocks))
        self.pix = nn.PixelShuffle(2)

        # Upsampling head: LR features -> HR image, one PixelShuffle at the end.
        self.tail = nn.Sequential(
            nn.Conv2d(chs[0], chs[0], 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(chs[0], in_ch * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )
        nn.init.zeros_(self.tail[-2].weight)
        nn.init.zeros_(self.tail[-2].bias)  # start as an exact bilinear upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad so every downsampling level divides evenly; crop back after.
        m = 2 ** (self.levels - 1)
        h, w = x.shape[-2:]
        ph, pw = (m - h % m) % m, (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")

        base = F.interpolate(x, scale_factor=self.scale, mode="bilinear",
                             align_corners=False)

        f = self.head(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            f = enc(f)
            skips.append(f)
            f = down(f)
        f = self.middle(f)
        for up, fuse, dec, skip in zip(self.ups, self.fuse, self.decoders, reversed(skips)):
            f = self.pix(up(f))
            f = fuse(torch.cat([f, skip], dim=1))
            f = dec(f)

        out = self.tail(f) + base
        if ph or pw:
            out = out[..., : h * self.scale, : w * self.scale]
        return out


def build(**kw) -> UNetSR:
    return UNetSR(**kw)
