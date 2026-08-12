"""Losses: Charbonnier + SSIM + (optional) LPIPS.

Charbonnier over L2 because speckle produces genuine outliers — a pixel corrupted by
the tail of the Gamma distribution is not a mistake to be chased. L2 would let those
few pixels dominate the gradient; Charbonnier is a smooth L1 that grows linearly.

SSIM and LPIPS are in the loss because they are two of the three scored metrics.
Optimising pixel error alone reliably produces the over-smoothed result that scores
well on pSNR and badly on the other two.

Weights start at 1.0 / 0.2 / 0.1 per the plan and are tuned on validation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


def _gaussian_window(size: int, sigma: float, device, dtype):
    c = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(1, 1, size, size).contiguous()


class SSIMLoss(nn.Module):
    """1 - SSIM, computed in-graph so it can be optimised directly.

    Uses a Gaussian window (as in the original SSIM paper and skimage's
    gaussian_weights=True path) rather than a uniform box, so the training objective
    matches the evaluation metric as closely as practical.
    """

    def __init__(self, window: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        self.window = window
        self.sigma = sigma
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2
        self._w = None

    def forward(self, pred, target):
        if self._w is None or self._w.device != pred.device or self._w.dtype != pred.dtype:
            self._w = _gaussian_window(self.window, self.sigma, pred.device, pred.dtype)
        c = pred.shape[1]
        w = self._w.expand(c, 1, self.window, self.window)
        pad = self.window // 2

        mu1 = F.conv2d(pred, w, padding=pad, groups=c)
        mu2 = F.conv2d(target, w, padding=pad, groups=c)
        mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s1 = F.conv2d(pred * pred, w, padding=pad, groups=c) - mu1s
        s2 = F.conv2d(target * target, w, padding=pad, groups=c) - mu2s
        s12 = F.conv2d(pred * target, w, padding=pad, groups=c) - mu12

        ssim = ((2 * mu12 + self.C1) * (2 * s12 + self.C2)) / \
               ((mu1s + mu2s + self.C1) * (s1 + s2 + self.C2))
        return 1.0 - ssim.mean()


class LPIPSLoss(nn.Module):
    """Wraps the lpips package. Greyscale is repeated to 3 channels and mapped to
    [-1,1], matching how src/metrics.py scores it."""

    def __init__(self, net: str = "alex"):
        super().__init__()
        import lpips
        self.model = lpips.LPIPS(net=net)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def forward(self, pred, target):
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        return self.model(pred.clamp(0, 1) * 2 - 1, target * 2 - 1).mean()


class CombinedLoss(nn.Module):
    def __init__(self, w_char: float = 1.0, w_ssim: float = 0.2, w_lpips: float = 0.1,
                 lpips_net: str = "alex"):
        super().__init__()
        self.w = {"char": w_char, "ssim": w_ssim, "lpips": w_lpips}
        self.char = CharbonnierLoss()
        self.ssim = SSIMLoss() if w_ssim > 0 else None
        self.lpips = LPIPSLoss(lpips_net) if w_lpips > 0 else None

    def forward(self, pred, target) -> tuple[torch.Tensor, dict]:
        parts = {"char": self.char(pred, target)}
        if self.ssim is not None:
            parts["ssim"] = self.ssim(pred, target)
        if self.lpips is not None:
            # LPIPS backbone is fp32-only in practice; run it outside autocast.
            with torch.autocast("cuda", enabled=False):
                parts["lpips"] = self.lpips(pred.float(), target.float())
        total = sum(self.w[k] * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}
