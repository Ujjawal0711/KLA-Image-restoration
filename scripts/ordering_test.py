"""Does the assumed noise ORDER matter? (NOTES.md open question #2)

We assume speckle is applied first and additive Gaussian second:

    degraded = clean * g + n

The reverse is equally plausible and no test distinguishes them from the deck:

    degraded = (clean + n) * g

Retraining under the other ordering and comparing would prove nothing, because both
would be scored on their own synthetic validation set -- there is no ground truth to
appeal to. The useful question is different and much cheaper:

    does the shipped model actually CARE which ordering it is given?

If it scores the same either way, the open question is moot and no work is needed. If
it drops sharply on the untrained ordering, the generator should randomise the order
so the model covers both.

    python scripts/ordering_test.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics as M  # noqa: E402
from src.dataset import list_images, load_gray01  # noqa: E402
from src.degradation import KERNELS, downsample  # noqa: E402
from src.models import build_from_arch  # noqa: E402


def degrade_speckle_first(clean_lr, L, sigma, rng):
    """clean * g + n  -- what we trained on."""
    g = rng.gamma(L, 1.0 / L, clean_lr.shape).astype(np.float32)
    return (clean_lr * g + rng.normal(0, sigma, clean_lr.shape)).astype(np.float32)


def degrade_additive_first(clean_lr, L, sigma, rng):
    """(clean + n) * g  -- the untested alternative."""
    g = rng.gamma(L, 1.0 / L, clean_lr.shape).astype(np.float32)
    return ((clean_lr + rng.normal(0, sigma, clean_lr.shape)) * g).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "checkpoints" / "model.pt"))
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = build_from_arch(ck.get("arch"))
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    model = model.to(dev).eval()

    paths = list_images(ROOT / "data" / "raw" / "gt")
    rng0 = np.random.default_rng(0)
    hrs = []
    for i in rng0.choice(len(paths), size=min(args.n * 3, len(paths)), replace=False):
        img = load_gray01(paths[int(i)])
        if img is None or min(img.shape[:2]) < 256:
            continue
        h, w = img.shape[:2]
        y, x = (h - 256) // 2, (w - 256) // 2
        hrs.append(np.clip(img[y:y + 256, x:x + 256], 0, 1).astype(np.float32))
        if len(hrs) >= args.n:
            break

    lp = None
    try:
        lp = M.LPIPS()
    except Exception:
        pass

    print(f"model {ck.get('arch')} on {len(hrs)} images\n")
    print(f"{'ordering':<22}{'pSNR':>9}{'SSIM':>9}{'LPIPS':>9}")
    print("-" * 49)

    out = {}
    # A meaningful sigma: at the deck's ~0.0086 the additive term is so small that any
    # ordering difference would be invisible. 0.02 is the top of the trained range.
    for label, fn, sigma in [("speckle then noise", degrade_speckle_first, 0.02),
                             ("noise then speckle", degrade_additive_first, 0.02),
                             ("speckle then noise (s=0.05)", degrade_speckle_first, 0.05),
                             ("noise then speckle (s=0.05)", degrade_additive_first, 0.05)]:
        rows = []
        rng = np.random.default_rng(1234)
        for hr in hrs:
            clr = downsample(hr, 2, "area")
            lr = fn(clr, 17.0, sigma, rng)
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=dev.type == "cuda"):
                    y = model(torch.from_numpy(lr[None, None]).to(dev))
                y = y.float().clamp_(0, 1)
            pred = y.cpu().numpy()[0, 0]
            r = {"psnr": M.psnr(pred, hr), "ssim": M.ssim(pred, hr)}
            if lp:
                r["lpips"] = lp(pred, hr)
            rows.append(r)
        a = M.aggregate(rows)
        out[label] = a
        lpv = f"{a.get('lpips', float('nan')):9.4f}"
        print(f"{label:<22}{a['psnr']:9.3f}{a['ssim']:9.4f}{lpv}")

    print()
    for s in ("", " (s=0.05)"):
        k1, k2 = f"speckle then noise{s}", f"noise then speckle{s}"
        if k1 in out and k2 in out:
            d = out[k2]["ssim"] - out[k1]["ssim"]
            print(f"  sigma{s or ' = 0.02'}: SSIM difference {d:+.5f}")

    diffs = [abs(out[f"noise then speckle{s}"]["ssim"]
                 - out[f"speckle then noise{s}"]["ssim"]) for s in ("", " (s=0.05)")]
    if max(diffs) < 0.005:
        print("\n  The model is INDIFFERENT to the ordering. Open question #2 is moot:")
        print("  whichever order the organisers used, performance is unchanged.")
    else:
        print("\n  ! The ordering matters. Randomise it in DegradationConfig so the")
        print("    model covers both possibilities.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
