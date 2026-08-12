"""NAFNet — Nonlinear Activation Free Network, adapted for 2x super-resolution.

Reference: "Simple Baselines for Image Restoration" (Chen et al., ECCV 2022).

Why this one, given an 8 GB training budget and a wall-clock-scored inference:

  - **SimpleGate replaces every activation.** Split the channels in half and multiply
    the halves elementwise. That is the entire nonlinearity — no GELU, no ReLU. It is
    cheaper than GELU and empirically works better here.
  - **Simplified Channel Attention** is global-average-pool -> 1x1 conv -> multiply.
    No softmax, no spatial attention, no quadratic cost in pixel count. This is what
    keeps it affordable relative to Restormer/SwinIR.
  - **No normalisation inside blocks** beyond LayerNorm over channels, which does not
    couple samples across the batch.

Adaptation for this task: the original is a same-resolution restoration net. Here the
whole body runs at LR and a single PixelShuffle at the tail produces the 2x output —
same rationale as the U-Net baseline (4x less work than restoring at HR), plus the
global bilinear residual so the net only learns the correction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an NCHW tensor."""

    def __init__(self, ch: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Channel-split multiplicative gate. Halves the channel count."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, ch: int, dw_expand: int = 2, ffn_expand: int = 2,
                 drop_path: float = 0.0):
        super().__init__()
        dw = ch * dw_expand
        self.norm1 = LayerNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)  # depthwise
        self.sg = SimpleGate()
        # Simplified channel attention: pooled context -> per-channel gain.
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw // 2, dw // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw // 2, ch, 1)

        ffn = ch * ffn_expand
        self.norm2 = LayerNorm2d(ch)
        self.conv4 = nn.Conv2d(ch, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, ch, 1)

        # Learnable residual scales, init 0 -> block starts as identity. Makes deep
        # stacks trainable without warmup tricks.
        self.beta = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)
        return x + y * self.gamma


class NAFNetSR(nn.Module):
    def __init__(self, in_ch: int = 1, width: int = 48,
                 enc_blocks: tuple[int, ...] = (2, 2, 4),
                 middle_blocks: int = 8,
                 dec_blocks: tuple[int, ...] = (2, 2, 2),
                 scale: int = 2):
        super().__init__()
        self.scale = scale
        self.levels = len(enc_blocks)

        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2

        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            # 1x1 to 2x channels then PixelShuffle(2) halves channels and doubles HW.
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1, bias=False),
                                          nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

        self.tail = nn.Sequential(
            nn.Conv2d(ch, in_ch * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )
        nn.init.zeros_(self.tail[0].weight)
        nn.init.zeros_(self.tail[0].bias)  # start as an exact bilinear upsample

    def forward(self, x):
        m = 2 ** self.levels
        h, w = x.shape[-2:]
        ph, pw = (m - h % m) % m, (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")

        base = F.interpolate(x, scale_factor=self.scale, mode="bilinear",
                             align_corners=False)

        f = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            f = enc(f)
            skips.append(f)
            f = down(f)
        f = self.middle(f)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            f = up(f)
            f = f + skip           # additive skip, as in the reference implementation
            f = dec(f)

        out = self.tail(f) + base
        if ph or pw:
            out = out[..., : h * self.scale, : w * self.scale]
        return out


def build(**kw) -> NAFNetSR:
    # `base` is the shared arg name in train.py; map it to NAFNet's `width`.
    if "base" in kw:
        kw["width"] = kw.pop("base")
    return NAFNetSR(**kw)
