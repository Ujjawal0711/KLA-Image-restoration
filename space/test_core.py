"""Exercise every demo path without Gradio installed.

Gradio is only wiring; these are the parts that can actually be wrong — the model
loading, the degradation, the restoration, the metrics and the histogram. Run from
inside the space/ directory:

    python test_core.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

import core

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


print(f"model: {core.ARCH}  {core.N_PARAMS/1e6:.1f}M params")
print(f"samples: {len(core.SAMPLES)}")
for lbl in core.SAMPLES:
    print(f"    {lbl}")

print("\n=== every sample, degrade + restore ===")
for label in core.SAMPLES:
    gt_d, lr_d, fig, info, st = core.make_degraded(label, None, 17, 0.0086, "area", 0)
    gt, lr = st
    t0 = time.perf_counter()
    bic_d, out_d, md = core.restore(st)
    dt = time.perf_counter() - t0

    ok_shape = (out_d.shape[0] == lr.shape[0] * 2 and out_d.shape[1] == lr.shape[1] * 2)
    ok_gt = out_d.shape[:2] == gt.shape[:2]
    check(f"{label[:34]:<34} {lr.shape[0]}->{out_d.shape[0]}  {dt:.2f}s",
          ok_shape and ok_gt and out_d.dtype == np.uint8)
    # The demo's whole premise: degraded values must exceed the GT range.
    if lr.max() <= 1.0:
        check(f"    overflow present for {label[:22]}", False,
              f"max={lr.max():.3f}")

print("\n=== overflow scales with speckle strength ===")
prev = None
for L in (100, 40, 17, 8, 4):
    _, _, _, _, st = core.make_degraded(next(iter(core.SAMPLES)), None,
                                        L, 0.0086, "area", 0)
    _, lr = st
    over = float((lr > 1.0).mean()) * 100
    print(f"  L={L:>3}   max={lr.max():.3f}   {over:5.2f}% of pixels above 1.0")
    if prev is not None:
        check(f"  L={L} noisier than L={prev}", lr.max() >= prev_max - 1e-6)
    prev, prev_max = L, lr.max()

print("\n=== quality improves as noise falls ===")
last = -1.0
for L in (4, 17, 100):
    _, _, _, _, st = core.make_degraded(next(iter(core.SAMPLES)), None,
                                        L, 0.0086, "area", 0)
    _, _, md = core.restore(st)
    ssim = float(md.split("| SSIM |")[1].split("|")[1].replace("*", "").strip())
    print(f"  L={L:>3}   model SSIM {ssim:.4f}")
    check(f"  SSIM rises from L={L}", ssim > last)
    last = ssim

print("\n=== uploaded image path ===")
rng = np.random.default_rng(0)
for shape, note in [((300, 400), "non-square"), ((257, 257), "odd -> trimmed"),
                    ((900, 900), "large -> cropped to 512")]:
    up = (rng.random(shape) * 255).astype(np.uint8)
    gt_d, lr_d, fig, info, st = core.make_degraded(None, up, 17, 0.0086, "area", 0)
    gt, lr = st
    _, out_d, _ = core.restore(st)
    check(f"  upload {shape} ({note}) -> {gt.shape} -> out {out_d.shape[:2]}",
          gt.shape[0] % 2 == 0 and gt.shape[1] % 2 == 0
          and out_d.shape[:2] == gt.shape[:2] and max(gt.shape) <= 512)

print("\n=== all four kernels ===")
from src.degradation import KERNELS
for k in KERNELS:
    _, _, _, _, st = core.make_degraded(next(iter(core.SAMPLES)), None, 17, 0.0086, k, 0)
    _, _, md = core.restore(st)
    check(f"  kernel {k:<9}", "SSIM" in md)

print("\n=== determinism (same seed -> identical image) ===")
_, _, _, _, a = core.make_degraded(next(iter(core.SAMPLES)), None, 17, 0.0086, "area", 3)
_, _, _, _, b = core.make_degraded(next(iter(core.SAMPLES)), None, 17, 0.0086, "area", 3)
_, _, _, _, c = core.make_degraded(next(iter(core.SAMPLES)), None, 17, 0.0086, "area", 4)
check("  same seed reproduces", np.array_equal(a[1], b[1]))
check("  different seed differs", not np.array_equal(a[1], c[1]))

print("\n=== empty state is handled ===")
r = core.restore(None)
check("  restore(None) returns a message, no crash", r[0] is None and isinstance(r[2], str))

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
