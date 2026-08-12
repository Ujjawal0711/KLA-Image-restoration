"""Score a directory of restored images against ground truth.

Pre-submission sanity check: run evaluate.py, then score its actual on-disk output.
This catches things the training loop cannot — wrong dtype on write, filename
mismatches, silent clipping, a stale checkpoint — because it reads back exactly what
the grader will read.

    python scripts/score_outputs.py <pred_dir> <gt_dir> [--baseline <input_dir>]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import metrics as M  # noqa: E402
from src.dataset import load_gray01  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir", type=pathlib.Path)
    ap.add_argument("gt_dir", type=pathlib.Path)
    ap.add_argument("--baseline", type=pathlib.Path, default=None,
                    help="directory of degraded inputs; also scores bicubic upsample "
                         "of these as a reference floor")
    args = ap.parse_args()

    gts = {p.name: p for p in args.gt_dir.iterdir() if p.is_file()}
    preds = {p.name: p for p in args.pred_dir.iterdir() if p.is_file()}
    common = sorted(set(gts) & set(preds))
    if not common:
        print("ERROR: no filenames in common between pred and gt dirs")
        print(f"  pred sample: {sorted(preds)[:3]}")
        print(f"  gt   sample: {sorted(gts)[:3]}")
        return 1
    missing = sorted(set(gts) - set(preds))
    if missing:
        print(f"WARNING: {len(missing)} GT files have no prediction "
              f"(e.g. {missing[:3]}) — the grader would score these as failures")

    try:
        lpips_fn = M.LPIPS()
    except Exception as e:  # noqa: BLE001
        print(f"LPIPS unavailable ({e})")
        lpips_fn = None

    rows, base_rows, shape_mismatch = [], [], []
    for name in common:
        gt = load_gray01(gts[name])
        pr = load_gray01(preds[name])
        if gt is None or pr is None:
            continue
        if pr.shape != gt.shape:
            shape_mismatch.append((name, pr.shape, gt.shape))
            continue
        r = {"psnr": M.psnr(pr, gt), "ssim": M.ssim(pr, gt)}
        if lpips_fn:
            r["lpips"] = lpips_fn(pr, gt)
        rows.append(r)

        if args.baseline:
            src = args.baseline / name
            if src.exists():
                lr = load_gray01(src)
                up = cv2.resize(lr, (gt.shape[1], gt.shape[0]),
                                interpolation=cv2.INTER_CUBIC)
                b = {"psnr": M.psnr(up, gt), "ssim": M.ssim(up, gt)}
                if lpips_fn:
                    b["lpips"] = lpips_fn(up, gt)
                base_rows.append(b)

    if shape_mismatch:
        print(f"\nERROR: {len(shape_mismatch)} predictions have the wrong shape:")
        for n, ps, gs in shape_mismatch[:5]:
            print(f"  {n}: predicted {ps}, expected {gs}")

    a = M.aggregate(rows)
    print(f"\nscored {len(rows)} images")
    print(f"  pSNR  {a['psnr']:8.3f}")
    print(f"  SSIM  {a['ssim']:8.4f}")
    if "lpips" in a:
        print(f"  LPIPS {a['lpips']:8.4f}  (lower is better)")

    if base_rows:
        b = M.aggregate(base_rows)
        print(f"\nbicubic floor on the same {len(base_rows)} images")
        print(f"  pSNR  {b['psnr']:8.3f}   delta {a['psnr'] - b['psnr']:+.3f}")
        print(f"  SSIM  {b['ssim']:8.4f}   delta {a['ssim'] - b['ssim']:+.4f}")
        if "lpips" in b:
            print(f"  LPIPS {b['lpips']:8.4f}   delta {a['lpips'] - b['lpips']:+.4f}"
                  f"  (negative is better)")
        if a["ssim"] <= b["ssim"]:
            print("\n  !! MODEL DOES NOT BEAT BICUBIC ON SSIM — do not submit this.")

    return 0 if not shape_mismatch else 1


if __name__ == "__main__":
    sys.exit(main())
