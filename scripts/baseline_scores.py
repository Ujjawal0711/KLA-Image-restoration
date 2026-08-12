"""Step 3 — score the non-learned floors on the validation split.

Any trained model that does not beat these is broken, and knowing the floor is the
only way to tell "the model learned something" from "upsampling is just easy on this
data". Reported floors:

  1. bicubic upsample only               - pure SR, no denoising
  2. bicubic + median 3x3                - cheap speckle suppression
  3. bicubic + bilateral                 - edge-preserving denoise
  4. median 3x3 at LR, then bicubic      - denoise BEFORE upsampling

(4) is included because order matters for multiplicative noise: filtering at LR
removes speckle while it is still white, whereas upsampling first spreads each noisy
pixel across four output pixels and correlates it, making it harder to remove.

Uses exactly the same val split, seed and degradation config as src/train.py, so the
numbers are directly comparable to the training log.

    python scripts/baseline_scores.py [--n 200]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import metrics as M  # noqa: E402
from src.dataset import build_datasets  # noqa: E402
from src.degradation import DegradationConfig  # noqa: E402


def up_bicubic(lr: np.ndarray, scale: int = 2) -> np.ndarray:
    h, w = lr.shape[:2]
    return cv2.resize(lr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


BASELINES = {
    "bicubic": lambda lr: up_bicubic(lr),
    "bicubic+median3": lambda lr: cv2.medianBlur(
        np.clip(up_bicubic(lr), 0, 1).astype(np.float32), 3),
    "bicubic+bilateral": lambda lr: cv2.bilateralFilter(
        np.clip(up_bicubic(lr), 0, 1).astype(np.float32), 5, 0.1, 5),
    "median3@LR+bicubic": lambda lr: up_bicubic(
        cv2.medianBlur(np.clip(lr, 0, 1).astype(np.float32), 3)),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="data/raw/gt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--hr-size", type=int, default=256)
    args = ap.parse_args()

    _, val = build_datasets(args.gt_dir, hr_size=args.hr_size,
                            cfg=DegradationConfig(), seed=0)
    n = min(args.n, len(val))
    print(f"scoring {n} validation samples from {len(val.paths)} held-out images\n")

    try:
        lpips_fn = M.LPIPS()
    except Exception as e:  # noqa: BLE001
        print(f"LPIPS unavailable ({e}) — pSNR/SSIM only\n")
        lpips_fn = None

    results = {k: [] for k in BASELINES}
    for i in range(n):
        lr_t, hr_t = val[i]
        lr = lr_t.numpy()[0]
        hr = hr_t.numpy()[0]
        for name, fn in BASELINES.items():
            pred = np.clip(fn(lr), 0, 1).astype(np.float32)
            row = {"psnr": M.psnr(pred, hr), "ssim": M.ssim(pred, hr)}
            if lpips_fn is not None:
                row["lpips"] = lpips_fn(pred, hr)
            results[name].append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    print(f"\n{'baseline':>20}  {'pSNR':>8}  {'SSIM':>8}  {'LPIPS':>8}")
    print("  " + "-" * 50)
    agg = {}
    for name, rows in results.items():
        a = M.aggregate(rows)
        agg[name] = a
        lp = f"{a.get('lpips', float('nan')):8.4f}"
        print(f"{name:>20}  {a['psnr']:8.3f}  {a['ssim']:8.4f}  {lp}")

    best = max(agg, key=lambda k: agg[k]["ssim"])
    print(f"\nFLOOR TO BEAT (best SSIM): {best} -> SSIM {agg[best]['ssim']:.4f}, "
          f"pSNR {agg[best]['psnr']:.3f}")
    print("Record these in NOTES.md. A trained model below this line is broken.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
