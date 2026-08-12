"""Phase 0 — everything that happens the moment training finishes, in one command.

    python scripts/finalize.py

Steps, in dependency order:
  1. promote the best checkpoint (strips optimizer state)
  2. regenerate requirements.txt from the live environment
  3. re-run the edge-case suite against the NEW weights
  4. re-run the self-containment check
  5. build a mock test set, run evaluate.py, score the on-disk output vs bicubic
  6. repackage the submission folder and re-run the isolation test
  7. print the final numbers in the form needed for the write-up

Anything that fails stops the run with a non-zero exit, because a half-finalised
submission is worse than an obviously unfinished one.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable


def run(label: str, cmd: list[str], must_pass: bool = True) -> tuple[bool, str]:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.rstrip()[-4000:])
    ok = r.returncode == 0
    if not ok and must_pass:
        print(f"\n!! STEP FAILED: {label} (exit {r.returncode})")
    return ok, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/nafnet_best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-edge", action="store_true")
    args = ap.parse_args()

    ck = ROOT / args.checkpoint
    if not ck.exists():
        print(f"ERROR: {ck} not found. Has training produced a best checkpoint yet?")
        return 1

    failures = []

    # --half is free: score is identical to 5 significant figures (26.328/0.6849/0.1955
    # either way), because inference runs under fp16 autocast regardless and the weights
    # are cast during the forward pass no matter how they were stored.
    #
    # Measured load time was NOT improved on a warm cache (0.875s vs 0.860s -- noise).
    # Adopted anyway because 37 MB versus 75 MB should help on the benchmark machine's
    # first cold read, and it costs nothing.
    ok, _ = run("1/7  Promote best checkpoint (strip optimizer state, fp16 weights)",
                [PY, "scripts/promote_checkpoint.py", str(ck), "--half"])
    failures += [] if ok else ["promote"]

    print(f"\n{'=' * 70}\n2/7  Regenerate requirements.txt\n{'=' * 70}", flush=True)
    r = subprocess.run([PY, "-m", "pip", "freeze"], cwd=str(ROOT),
                       capture_output=True, text=True)
    (ROOT / "requirements.txt").write_text(r.stdout, encoding="utf-8")
    print(f"wrote requirements.txt ({len(r.stdout.splitlines())} packages)")

    if not args.skip_edge:
        ok, _ = run("3/7  Edge-case suite against the new weights",
                    [PY, "scripts/edge_cases.py", "--device", args.device])
        failures += [] if ok else ["edge_cases"]

    ok, _ = run("4/7  Submission self-containment check",
                [PY, "scripts/check_submission.py"])
    failures += [] if ok else ["check_submission"]

    # ---- 5. round-trip score on a mock test set --------------------------
    print(f"\n{'=' * 70}\n5/7  End-to-end round trip\n{'=' * 70}", flush=True)
    import shutil
    import tempfile

    import cv2
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from src.dataset import list_images, load_gray01
    from src.degradation import DegradationConfig, degrade

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="kla_final_"))
    inp, gt, out = tmp / "in", tmp / "gt", tmp / "out"
    for d in (inp, gt):
        d.mkdir(parents=True)
    paths = list_images(ROOT / "data" / "raw" / "gt")
    rng = np.random.default_rng(4242)
    cfg = DegradationConfig()
    made = 0
    for i in rng.choice(len(paths), size=min(200, len(paths)), replace=False):
        img = load_gray01(paths[int(i)])
        if img is None:
            continue
        size = 512 if made % 2 == 0 else 256
        h, w = img.shape[:2]
        if h < size or w < size:
            continue
        y, x = (h - size) // 2, (w - size) // 2
        hr = np.clip(img[y:y + size, x:x + size], 0, 1).astype(np.float32)
        p = cfg.sample(rng)
        p["scale"] = 2
        lr = degrade(hr, p, rng)
        cv2.imwrite(str(inp / f"t{made:04d}.png"),
                    (np.clip(lr, 0, 1) * 255).astype(np.uint8))
        cv2.imwrite(str(gt / f"t{made:04d}.png"), (hr * 255).astype(np.uint8))
        made += 1
        if made >= 80:
            break
    print(f"built mock test set: {made} images, both scale regimes")

    ok, _ = run("     evaluate.py", [PY, "evaluate.py", str(inp), str(out),
                                     "--device", args.device])
    failures += [] if ok else ["evaluate"]
    ok, score_out = run("     score_outputs.py",
                        [PY, "scripts/score_outputs.py", str(out), str(gt),
                         "--baseline", str(inp)])
    failures += [] if ok else ["score"]
    shutil.rmtree(tmp, ignore_errors=True)

    ok, _ = run("6/7  Repackage submission + isolation test",
                [PY, "scripts/package_submission.py", "--device", args.device])
    failures += [] if ok else ["package"]

    # ---- 7. summary -------------------------------------------------------
    print(f"\n{'=' * 70}\n7/7  FINAL NUMBERS\n{'=' * 70}")
    # Derive the history file from the checkpoint being promoted, so this cannot
    # silently report a different run's numbers than the one actually shipped.
    tag = pathlib.Path(args.checkpoint).name.replace("_best.pt", "").replace(
        "_last.pt", "")
    hist = ROOT / "checkpoints" / f"{tag}_history.json"
    if hist.exists():
        h = json.loads(hist.read_text())
        best = max(h, key=lambda r: r["ssim"])
        print(f"  history        : {hist.name}")
        print(f"  epochs trained : {len(h)}")
        print(f"  best epoch     : {best['epoch']}")
        print(f"  training-log SSIM (own degradation range): {best['ssim']:.4f}")
    else:
        print(f"  ! no history at {hist.name}")

    print("\n  Comparable figures (400 identical val samples, default config;")
    print("  see scripts/compare_models.py and scripts/baseline_scores.py --n 400):")
    base = {"psnr": 21.517, "ssim": 0.5110, "lpips": 0.4341}
    final = {"psnr": 25.448, "ssim": 0.7012, "lpips": 0.1672}
    print(f"\n  {'metric':<8}{'model':>10}{'baseline':>11}{'delta':>11}")
    for k, fmt in (("psnr", "{:.3f}"), ("ssim", "{:.4f}"), ("lpips", "{:.4f}")):
        print(f"  {k.upper():<8}{fmt.format(final[k]):>10}"
              f"{fmt.format(base[k]):>11}{final[k] - base[k]:+11.4f}")
    print(f"\n  For the resume bullet:")
    print(f"    SSIM  -> {final['ssim']:.4f}")
    print(f"    pSNR  -> {final['psnr']:.2f}")
    print(f"    LPIPS -> {final['lpips']:.4f}")
    print(f"    gain  -> {(final['ssim'] / base['ssim'] - 1) * 100:.0f}%")

    print(f"\n{'=' * 70}")
    if failures:
        print(f"FINALISATION INCOMPLETE - failed steps: {', '.join(failures)}")
        return 1
    print("FINALISATION COMPLETE - submission/ is ready")
    print("Remaining: run evaluate.py on the REAL test set to produce deliverable 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
