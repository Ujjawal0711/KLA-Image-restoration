"""Forward degradation model — the hypothesis from NOTES.md §0.3.

    clean_lr = downsample(gt, factor=2, kernel)
    degraded = clean_lr * g + n,   g ~ Gamma(L, 1/L),   n ~ N(0, sigma^2)

Used in two places:
  - Step 4 dataset, to synthesise fresh degradations on the fly (the OOD defence)
  - scripts/validate_analysis.py, to check the Step 2 estimator recovers known params

No clipping anywhere. The deck states four times that degraded values may exceed the
GT range, and the multiplicative term is exactly why. Clipping here would train the
model on inputs the test set will not contain.

PARAMETER RANGES BELOW ARE PROVISIONAL — anchored on two figure titles from the deck,
not on data. Step 2 (`scripts/analyze_degradation.py`) must replace them.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# cv2 interpolation codes, kept as names so configs stay readable.
KERNELS = {
    "area": cv2.INTER_AREA,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


@dataclass
class DegradationConfig:
    """Sampling ranges for synthetic degradation.

    `L_range` is centred on the two observed values (16.86, 18.13) but widened
    substantially in both directions: the test set contains out-of-distribution
    samples, and a generator that only reproduces the training noise level gives
    no robustness. `sigma_range` is sampled log-uniformly because the two observed
    values differ by 8x, which reads like a log-scale draw.
    """

    L_range: tuple[float, float] = (8.0, 40.0)
    sigma_range: tuple[float, float] = (5e-4, 2e-2)
    sigma_log_uniform: bool = True
    kernels: tuple[str, ...] = ("area", "bilinear", "bicubic", "lanczos")
    kernel_weights: tuple[float, ...] | None = None
    scale: int = 2

    # Set True only if Step 2 finds evidence of blur (slide 9's ambiguous wording).
    blur_prob: float = 0.0
    blur_sigma_range: tuple[float, float] = (0.3, 0.8)

    def sample(self, rng: np.random.Generator) -> dict:
        L = float(rng.uniform(*self.L_range))
        if self.sigma_log_uniform:
            lo, hi = np.log(self.sigma_range[0]), np.log(self.sigma_range[1])
            sigma = float(np.exp(rng.uniform(lo, hi)))
        else:
            sigma = float(rng.uniform(*self.sigma_range))
        p = None
        if self.kernel_weights is not None:
            w = np.asarray(self.kernel_weights, dtype=np.float64)
            p = w / w.sum()
        kernel = str(rng.choice(self.kernels, p=p))
        blur = 0.0
        if self.blur_prob > 0 and rng.random() < self.blur_prob:
            blur = float(rng.uniform(*self.blur_sigma_range))
        return {"L": L, "sigma": sigma, "kernel": kernel, "blur": blur}


def downsample(gt: np.ndarray, scale: int = 2, kernel: str = "area") -> np.ndarray:
    h, w = gt.shape[:2]
    return cv2.resize(gt, (w // scale, h // scale), interpolation=KERNELS[kernel])


def apply_noise(clean_lr: np.ndarray, L: float, sigma: float,
                rng: np.random.Generator) -> np.ndarray:
    """Multiplicative Gamma speckle then additive Gaussian.

    Gamma(shape=L, scale=1/L) has mean 1 and variance 1/L, so the speckle is
    unbiased and its relative std is 1/sqrt(L).
    """
    g = rng.gamma(shape=L, scale=1.0 / L, size=clean_lr.shape).astype(np.float32)
    out = clean_lr * g
    if sigma > 0:
        out = out + rng.normal(0.0, sigma, clean_lr.shape).astype(np.float32)
    return out.astype(np.float32)


def degrade(gt: np.ndarray, params: dict, rng: np.random.Generator) -> np.ndarray:
    """gt: float32 (H,W) in [0,1] -> degraded float32 (H/2,W/2), unclipped."""
    x = gt
    if params.get("blur", 0.0) > 0:
        x = cv2.GaussianBlur(x, (0, 0), params["blur"])
    clean_lr = downsample(x, params.get("scale", 2), params["kernel"])
    return apply_noise(clean_lr, params["L"], params["sigma"], rng)


def degrade_random(gt: np.ndarray, cfg: DegradationConfig,
                   rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    p = cfg.sample(rng)
    p["scale"] = cfg.scale
    return degrade(gt, p, rng), p
