"""Measure test-time augmentation: is the quality gain worth the inference cost?

Geometric self-ensemble: run the image through the network several times under
flips/rotations, undo each transform, and average. Standard practice in
super-resolution and usually worth a few tenths of a decibel.

But it multiplies inference cost by the number of transforms, and half the competition
score is wall-clock time. So this measures BOTH sides and reports the trade, rather
than assuming the quality gain settles it.

    python scripts/tta_eval.py [--n 120]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics as M  # noqa: E402
from src.dataset import build_datasets  # noqa: E402
from src.degradation import DegradationConfig  # noqa: E402

# (flip_h, flip_v, transpose). Identity first so MODES[:k] is always a sensible subset.
TRANSFORMS = [
    (False, False, False), (True, False, False),
    (False, True, False), (True, True, False),
    (False, False, True), (True, False, True),
    (False, True, True), (True, True, True),
]


def apply(x, h, v, t):
    if h:
        x = torch.flip(x, [-1])
    if v:
        x = torch.flip(x, [-2])
    if t:
        x = x.transpose(-2, -1)
    return x


def unapply(x, h, v, t):
    # Inverse in reverse order.
    if t:
        x = x.transpose(-2, -1)
    if v:
        x = torch.flip(x, [-2])
    if h:
        x = torch.flip(x, [-1])
    return x


@torch.inference_mode()
def predict(model, x, k: int):
    """k = number of transforms to ensemble over (1 = no TTA)."""
    acc = None
    for (h, v, t) in TRANSFORMS[:k]:
        xt = apply(x, h, v, t)
        with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            yt = model(xt)
        y = unapply(yt.float(), h, v, t)
        acc = y if acc is None else acc + y
    return (acc / k).clamp(0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "checkpoints" / "model.pt"))
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    arch = ck.get("arch", {"name": "unet", "base": 48})
    from src.models import build_from_arch
    model = build_from_arch(arch)
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    model = model.to(dev).eval()

    _, val = build_datasets(ROOT / "data" / "raw" / "gt", hr_size=256,
                            cfg=DegradationConfig(), seed=0)
    n = min(args.n, len(val))
    pairs = [val[i] for i in range(n)]
    print(f"model {arch} on {n} identical validation samples\n")

    lp = None
    try:
        lp = M.LPIPS()
    except Exception:
        pass

    print(f"{'mode':<12}{'pSNR':>9}{'SSIM':>9}{'LPIPS':>9}{'ms/img':>10}{'vs 1x':>9}")
    print("-" * 58)
    base = None
    for k, label in [(1, "none (1x)"), (2, "h-flip (2x)"), (4, "flips (4x)"),
                     (8, "full (8x)")]:
        rows = []
        # warm up so the first mode is not charged for kernel selection
        predict(model, pairs[0][0][None].to(dev), k)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        t0 = time.perf_counter()
        preds = []
        for lr_t, _ in pairs:
            preds.append(predict(model, lr_t[None].to(dev), k).cpu().numpy()[0, 0])
        torch.cuda.synchronize() if dev.type == "cuda" else None
        ms = (time.perf_counter() - t0) / n * 1000

        for pred, (_, hr_t) in zip(preds, pairs):
            hr = hr_t.numpy()[0]
            r = {"psnr": M.psnr(pred, hr), "ssim": M.ssim(pred, hr)}
            if lp:
                r["lpips"] = lp(pred, hr)
            rows.append(r)
        a = M.aggregate(rows)
        if base is None:
            base = (a, ms)
            delta = "-"
        else:
            delta = f"{ms / base[1]:.2f}x"
        lpv = f"{a.get('lpips', float('nan')):9.4f}"
        print(f"{label:<12}{a['psnr']:9.3f}{a['ssim']:9.4f}{lpv}{ms:10.1f}{delta:>9}")
        if base is not None and k > 1:
            print(f"{'':12}{a['psnr']-base[0]['psnr']:+9.3f}"
                  f"{a['ssim']-base[0]['ssim']:+9.4f}"
                  f"{a.get('lpips',0)-base[0].get('lpips',0):+9.4f}"
                  f"{'  (delta)':>19}")

    print("\nTrade-off: the competition rewards faster pipelines when quality is")
    print("comparable. Adopt TTA only if the quality gain is large enough to outweigh")
    print("the multiplied inference time on the full test set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
