"""Phase 2 — measure how the model degrades when the true noise differs from training.

This is the one experiment that directly probes the project's central risk. The
degradation model was inferred from two figure captions and can never be verified
against real data, so the question that matters is not "how good is the model at the
assumed noise level" but "how fast does it fall apart if that assumption is wrong".

Sweeps speckle strength L and additive sigma across and BEYOND the training ranges,
scoring each cell. A flat surface means the wide-range training strategy worked; a
sharp cliff inside the plausible parameter space is a reason to retrain wider.

    python scripts/sensitivity_sweep.py --weights checkpoints/model.pt
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import metrics as M  # noqa: E402
from src.dataset import list_images, load_gray01  # noqa: E402
from src.degradation import degrade  # noqa: E402

# Training used L in [8,40] and sigma in [5e-4,2e-2]. The grid deliberately extends
# past both ends -- the interesting question is what happens outside what we trained on.
L_GRID = [4, 6, 8, 12, 17, 25, 40, 60, 100]
SIGMA_GRID = [0.0, 5e-4, 2e-3, 8.6e-3, 2e-2, 5e-2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="checkpoints/model.pt")
    ap.add_argument("--gt-dir", default="data/raw/gt")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--kernel", default="area")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    arch = ck.get("arch", {"name": "unet", "base": 48})
    from src.models import build_from_arch
    model = build_from_arch(arch)
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    model = model.to(dev).eval()

    paths = list_images(pathlib.Path(args.gt_dir))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(paths), size=min(args.n * 3, len(paths)), replace=False)
    hrs = []
    for i in sel:
        img = load_gray01(paths[int(i)])
        if img is None or min(img.shape[:2]) < 256:
            continue
        h, w = img.shape[:2]
        y, x = (h - 256) // 2, (w - 256) // 2
        hrs.append(np.clip(img[y:y + 256, x:x + 256], 0, 1).astype(np.float32))
        if len(hrs) >= args.n:
            break
    print(f"model {arch} on {len(hrs)} images, kernel={args.kernel}, device={dev}\n")

    print("SSIM surface (rows = L, cols = sigma)")
    header = "      L | " + "  ".join(f"{s:>7.4f}" for s in SIGMA_GRID)
    print(header)
    print("  " + "-" * (len(header) - 2))

    table = {}
    for L in L_GRID:
        row = []
        for sg in SIGMA_GRID:
            preds, gts = [], []
            r2 = np.random.default_rng(1234)
            for hr in hrs:
                lr = degrade(hr, {"L": float(L), "sigma": float(sg),
                                  "kernel": args.kernel, "scale": 2}, r2)
                x = torch.from_numpy(lr[None, None]).to(dev)
                with torch.inference_mode():
                    with torch.autocast("cuda", dtype=torch.float16,
                                        enabled=dev.type == "cuda"):
                        y = model(x)
                    y = y.float().clamp_(0, 1)
                preds.append(y.cpu().numpy()[0, 0])
                gts.append(hr)
            s = float(np.mean([M.ssim(p, g) for p, g in zip(preds, gts)]))
            row.append(s)
            table[(L, sg)] = s
        marks = []
        for sg, s in zip(SIGMA_GRID, row):
            inside = 8 <= L <= 40 and 5e-4 <= sg <= 2e-2
            marks.append(f"{s:>7.4f}{'*' if inside else ' '}")
        print(f"  {L:>6} | " + " ".join(marks))

    inside = [v for (L, s), v in table.items()
              if 8 <= L <= 40 and 5e-4 <= s <= 2e-2]
    outside = [v for (L, s), v in table.items()
               if not (8 <= L <= 40 and 5e-4 <= s <= 2e-2)]
    print("\n  * = inside the training range")
    print(f"\n  inside  training range: mean SSIM {np.mean(inside):.4f}  "
          f"min {np.min(inside):.4f}")
    print(f"  outside training range: mean SSIM {np.mean(outside):.4f}  "
          f"min {np.min(outside):.4f}")
    drop = (np.mean(inside) - np.min(outside)) / max(np.mean(inside), 1e-9)
    print(f"  worst-case drop vs in-range mean: {drop*100:.1f}%")
    if drop > 0.25:
        print("\n  ! Sharp degradation outside the trained range. Consider widening")
        print("    DegradationConfig and retraining before submission.")
    else:
        print("\n  Degrades gracefully outside the trained range — the wide-range")
        print("  strategy is doing its job.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
