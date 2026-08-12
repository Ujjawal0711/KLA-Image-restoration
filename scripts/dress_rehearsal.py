"""Full dress rehearsal: build a realistic test set, run the PACKAGED submission on it,
score the results.

This is the last check before submitting. It deliberately differs from every earlier
measurement in three ways:

  1. It runs `submission/evaluate.py`, not the development copy, with the working
     directory inside the package and PYTHONPATH cleared -- the same conditions the
     graders will use.
  2. The test images come from held-out sources the model never trained on, INCLUDING
     the two out-of-distribution corpora, mirroring the deck's statement that the real
     test set mixes in-distribution and out-of-distribution samples.
  3. It mixes both scale regimes (128->256 and 256->512) in one directory, because the
     real test set will not be sorted for us.

Everything is scored from files on disk, so it measures what the graders measure --
including any loss from quantising to 8-bit PNG on the way out.

    python scripts/dress_rehearsal.py [--n 200]
"""

from __future__ import annotations

import argparse
import collections
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics as M  # noqa: E402
from src.dataset import list_images, load_gray01, split_by_source  # noqa: E402
from src.degradation import DegradationConfig, degrade  # noqa: E402


def build_testset(dst_in: pathlib.Path, dst_gt: pathlib.Path, n: int,
                  seed: int = 20260816):
    """Held-out + OOD sources, both regimes, realistic degradation."""
    rng = np.random.default_rng(seed)
    cfg = DegradationConfig()

    # Held-out validation images (never trained on) ...
    _, val = split_by_source(list_images(ROOT / "data" / "raw" / "gt"), 0.05, 0)
    # ... plus genuinely unseen corpora, to mimic the deck's OOD samples.
    ood = []
    for name in ("bright", "fluor"):
        d = ROOT / "data" / "raw" / "ood" / name
        if d.exists():
            ood.extend(list_images(d))

    pool = [(p, "held-out") for p in val] + [(p, "OOD") for p in ood]
    rng.shuffle(pool)

    manifest = []
    made = 0
    for path, origin in pool:
        img = load_gray01(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        # Alternate regimes; 512 crops need a big enough source.
        size = 512 if (made % 2 == 0 and min(h, w) >= 512) else 256
        if min(h, w) < size:
            continue
        y, x = (h - size) // 2, (w - size) // 2
        hr = np.clip(img[y:y + size, x:x + size], 0, 1).astype(np.float32)
        p = cfg.sample(rng)
        p["scale"] = 2
        lr = degrade(hr, p, rng)

        name = f"test_{made:04d}.png"
        # Inputs are written the way a provider would: 8-bit PNG, clipped.
        cv2.imwrite(str(dst_in / name), (np.clip(lr, 0, 1) * 255).round().astype(np.uint8))
        cv2.imwrite(str(dst_gt / name), (hr * 255).round().astype(np.uint8))
        manifest.append({"name": name, "origin": origin, "regime": f"{size//2}->{size}",
                         "corpus": path.stem.split("_")[0], "L": p["L"],
                         "sigma": p["sigma"], "kernel": p["kernel"]})
        made += 1
        if made >= n:
            break
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--package", type=pathlib.Path, default=ROOT / "submission")
    ap.add_argument("--keep", action="store_true", help="keep the test set on disk")
    args = ap.parse_args()

    if not (args.package / "evaluate.py").exists():
        print(f"ERROR: no evaluate.py in {args.package}. Run package_submission.py first.")
        return 1

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="kla_rehearsal_"))
    d_in, d_gt, d_out = tmp / "input", tmp / "gt", tmp / "output"
    for d in (d_in, d_gt):
        d.mkdir(parents=True)

    print("=" * 70)
    print("1/4  Building test set")
    print("=" * 70)
    man = build_testset(d_in, d_gt, args.n)
    reg = collections.Counter(m["regime"] for m in man)
    org = collections.Counter(m["origin"] for m in man)
    cor = collections.Counter(m["corpus"] for m in man)
    print(f"  {len(man)} images written to {d_in}")
    print(f"  regimes : {dict(reg)}")
    print(f"  origin  : {dict(org)}")
    print(f"  corpora : {dict(cor)}")
    Ls = [m["L"] for m in man]
    sg = [m["sigma"] for m in man]
    print(f"  L       : {min(Ls):.1f} to {max(Ls):.1f}")
    print(f"  sigma   : {min(sg):.5f} to {max(sg):.5f}")

    print("\n" + "=" * 70)
    print("2/4  Running the PACKAGED evaluate.py (graders' conditions)")
    print("=" * 70)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "evaluate.py", str(d_in), str(d_out)],
                       cwd=str(args.package), env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    print((r.stdout or "").rstrip())
    if r.returncode != 0:
        print("STDERR:", (r.stderr or "")[-2000:])
        print(f"\nFAILED: exit {r.returncode}")
        return 1
    print(f"  measured wall clock (process start to exit): {wall:.2f}s "
          f"({len(man)/wall:.1f} img/s)")

    print("\n" + "=" * 70)
    print("3/4  Verifying outputs")
    print("=" * 70)
    problems = []
    outs = sorted(d_out.rglob("*"))
    outs = [p for p in outs if p.is_file()]
    print(f"  files in  : {len(man)}")
    print(f"  files out : {len(outs)}")
    if len(outs) != len(man):
        problems.append(f"count mismatch: {len(outs)} outputs for {len(man)} inputs")
    dtypes = collections.Counter()
    for m in man:
        q = d_out / m["name"]
        if not q.exists():
            problems.append(f"missing output {m['name']}")
            continue
        a = cv2.imread(str(d_in / m["name"]), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(str(q), cv2.IMREAD_UNCHANGED)
        dtypes[str(b.dtype)] += 1
        want = (a.shape[0] * 2, a.shape[1] * 2)
        if b.shape[:2] != want:
            problems.append(f"{m['name']}: {b.shape[:2]} != expected {want}")
    print(f"  dtypes    : {dict(dtypes)}")
    print(f"  dimensions: {'all exactly 2x' if not problems else 'MISMATCH'}")

    print("\n" + "=" * 70)
    print("4/4  Scoring on-disk outputs against ground truth")
    print("=" * 70)
    lp = None
    try:
        lp = M.LPIPS()
    except Exception:
        pass

    rows, by = [], collections.defaultdict(list)
    for m in man:
        q = d_out / m["name"]
        if not q.exists():
            continue
        pred = cv2.imread(str(q), cv2.IMREAD_UNCHANGED).astype(np.float32)
        pred = pred / (65535.0 if pred.dtype == np.uint16 else 255.0)
        gt = cv2.imread(str(d_gt / m["name"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        lr = cv2.imread(str(d_in / m["name"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        bic = np.clip(cv2.resize(lr, (gt.shape[1], gt.shape[0]),
                                 interpolation=cv2.INTER_CUBIC), 0, 1)
        row = {"psnr": M.psnr(pred, gt), "ssim": M.ssim(pred, gt),
               "b_psnr": M.psnr(bic, gt), "b_ssim": M.ssim(bic, gt)}
        if lp:
            row["lpips"] = lp(pred, gt)
            row["b_lpips"] = lp(bic, gt)
        rows.append(row)
        by[m["regime"]].append(row)
        by[m["origin"]].append(row)

    a = M.aggregate(rows)
    print(f"  {'group':<16}{'n':>5}{'pSNR':>9}{'SSIM':>9}{'LPIPS':>9}")
    print("  " + "-" * 46)
    print(f"  {'ALL':<16}{len(rows):>5}{a['psnr']:9.3f}{a['ssim']:9.4f}"
          f"{a.get('lpips', float('nan')):9.4f}")
    for k in sorted(by):
        g = M.aggregate(by[k])
        print(f"  {k:<16}{len(by[k]):>5}{g['psnr']:9.3f}{g['ssim']:9.4f}"
              f"{g.get('lpips', float('nan')):9.4f}")

    print(f"\n  bicubic reference on the same files:")
    print(f"  {'':<16}{len(rows):>5}{a['b_psnr']:9.3f}{a['b_ssim']:9.4f}"
          f"{a.get('b_lpips', float('nan')):9.4f}")
    print(f"  gain: pSNR {a['psnr']-a['b_psnr']:+.3f}   "
          f"SSIM {a['ssim']-a['b_ssim']:+.4f}   "
          f"LPIPS {a.get('lpips',0)-a.get('b_lpips',0):+.4f}")

    if args.keep:
        keep = ROOT / "rehearsal"
        if keep.exists():
            shutil.rmtree(keep)
        shutil.copytree(tmp, keep)
        print(f"\n  test set kept at {keep}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    if problems:
        print("PROBLEMS FOUND:")
        for p in problems[:20]:
            print(f"  - {p}")
        return 1
    print("DRESS REHEARSAL PASSED — the packaged submission works end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
