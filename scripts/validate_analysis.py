"""Validate the Step 2 estimators against synthetic data with KNOWN parameters.

The real dataset is not released yet, but the analysis code can still be trusted
or falsified today: generate pairs from the NOTES.md §0.3 model with parameters we
chose, then check `analyze_degradation.py` recovers them. If it cannot recover
parameters from data that provably follows the model, any number it reports on the
real data is meaningless.

This does NOT validate the model itself — only the estimator. Confirming the model
requires the real pairs.

    python scripts/validate_analysis.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.degradation import degrade, downsample  # noqa: E402
from analyze_degradation import (  # noqa: E402
    analyse_noise, check_blur, check_residual_whiteness, equivalent_kernels,
    identify_kernel, per_pair_params,
)

TRUE_KERNEL = "area"


def synth_gt(rng: np.random.Generator, size: int = 512) -> np.ndarray:
    """A 512x512 field with the character of the deck samples: mostly dark, heavy-tailed,
    with fine high-frequency structure. Kernel identification is only meaningful when
    the image actually has energy near Nyquist, so the texture term matters."""
    import cv2

    base = rng.random((size, size)).astype(np.float32)
    smooth = cv2.GaussianBlur(base, (0, 0), 6.0)
    smooth = (smooth - smooth.min()) / max(float(np.ptp(smooth)), 1e-9)

    fine = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 1.0)
    fine = (fine - fine.mean()) / max(fine.std(), 1e-9)

    img = smooth ** 3.0 + 0.12 * fine          # cube -> skewed dark, like the samples
    # sparse bright filaments, as in the dendrite sample
    for _ in range(14):
        y, x = rng.integers(20, size - 20, 2)
        ang = rng.random() * np.pi
        length = int(rng.integers(40, 160))
        pts = np.stack([
            y + np.arange(length) * np.sin(ang),
            x + np.arange(length) * np.cos(ang),
        ]).astype(int)
        ok = (pts[0] >= 0) & (pts[0] < size) & (pts[1] >= 0) & (pts[1] < size)
        img[pts[0][ok], pts[1][ok]] = 1.0
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    return np.clip(img, 0, 1).astype(np.float32)


def main() -> int:
    rng = np.random.default_rng(1234)
    n = 24
    true_L = rng.uniform(14.0, 20.0, n)
    true_sigma = np.exp(rng.uniform(np.log(8e-4), np.log(1.2e-2), n))

    print(f"generating {n} synthetic pairs "
          f"(kernel={TRUE_KERNEL}, L in [{true_L.min():.2f},{true_L.max():.2f}], "
          f"sigma in [{true_sigma.min():.5f},{true_sigma.max():.5f}])\n")

    pairs = []
    for i in range(n):
        gt = synth_gt(rng)
        d = degrade(gt, {"L": true_L[i], "sigma": true_sigma[i],
                         "kernel": TRUE_KERNEL, "scale": 2}, rng)
        pairs.append((d, gt))

    kernel = identify_kernel(pairs)
    L_hat, sigma_hat = analyse_noise(pairs, kernel)
    check_residual_whiteness(pairs, kernel)
    per_pair_params(pairs, kernel)
    check_blur(pairs, kernel)

    # Pooled fit recovers a harmonic-style average of per-sample L, so compare against
    # the mean of 1/L rather than the mean of L.
    L_expected = 1.0 / np.mean(1.0 / true_L)
    sigma_expected = np.sqrt(np.mean(true_sigma ** 2))

    print("\n" + "=" * 64)
    print("RECOVERY CHECK")
    print("=" * 64)
    # Accept any kernel in the true kernel's equivalence class: at exactly 2x,
    # area and bilinear are the same operation, so demanding one specific name
    # makes this assertion a coin flip decided by float rounding.
    eq = equivalent_kernels(TRUE_KERNEL)
    kernel_ok = kernel in eq
    print(f"  kernel   true={TRUE_KERNEL:>8}   recovered={kernel:>8}   "
          f"{'OK' if kernel_ok else 'FAIL'}   (class: {{{', '.join(eq)}}})")
    print(f"  L        true={L_expected:>8.3f}   recovered={L_hat:>8.3f}   "
          f"err={abs(L_hat - L_expected) / L_expected * 100:>5.1f}%")
    print(f"  sigma    true={sigma_expected:>8.6f}   recovered={sigma_hat:>8.6f}   "
          f"err={abs(sigma_hat - sigma_expected) / sigma_expected * 100:>5.1f}%")

    # Tolerances: L is strongly identified, so hold it to 15%. sigma is intrinsically
    # harder (it is ~2% of the total variance at mid intensities); 25% is the tightest
    # bound the dark-pixel estimator meets reliably, and is ample for choosing an
    # augmentation range.
    ok = (kernel_ok
          and abs(L_hat - L_expected) / L_expected < 0.15
          and abs(sigma_hat - sigma_expected) / sigma_expected < 0.25)
    print("\n" + ("PASS — estimator is trustworthy on model-conforming data."
                  if ok else
                  "FAIL — fix the estimator before running it on real data."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
