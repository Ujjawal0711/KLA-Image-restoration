"""Assemble the submission folder and prove it works in isolation.

The important part is not the copying, it is the test at the end: evaluate.py is run
from inside the packaged folder, with the working directory set there and the project
root removed from the import path. Any accidental dependence on a file that was never
going to be shipped shows up here rather than on the graders' machine.

    python scripts/package_submission.py [--test-input <dir>] [--device cpu]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Deliverable 1: the evaluation script and everything it imports at runtime.
INFERENCE = [
    "evaluate.py",
    "src/__init__.py",
    "src/models/__init__.py",
    "src/models/nafnet.py",
    "src/models/unet.py",
    "checkpoints/model.pt",
]

# Deliverable 2: everything needed to reproduce training from scratch.
TRAINING = [
    "src/train.py",
    "src/dataset.py",
    "src/degradation.py",
    "src/losses.py",
    "src/metrics.py",
    "src/procedural.py",
    "scripts/fetch_data.py",
    "scripts/make_procedural.py",
    "scripts/baseline_scores.py",
    "scripts/score_outputs.py",
    "scripts/promote_checkpoint.py",
    "scripts/preview_degradation.py",
    "scripts/sensitivity_sweep.py",
    "scripts/edge_cases.py",
    "scripts/check_submission.py",
]

# Deliverable 4 plus documentation.
DOCS = ["requirements.txt", "README.md", "NOTES.md"]


def copy_all(dst: pathlib.Path) -> list[str]:
    missing = []
    for rel in INFERENCE + TRAINING + DOCS:
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
    return missing


def isolation_test(pkg: pathlib.Path, test_input: pathlib.Path,
                   device: str) -> tuple[bool, str]:
    """Run evaluate.py from inside the package with the project root hidden."""
    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="kla_pkg_out_"))
    env = dict(os.environ)
    # Stop Python finding the real project via inherited path settings.
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    try:
        r = subprocess.run(
            [sys.executable, "evaluate.py", str(test_input), str(out_dir),
             "--device", device],
            cwd=str(pkg), env=env, capture_output=True, text=True, timeout=1200)
        n_in = len([p for p in test_input.rglob("*") if p.is_file()])
        n_out = len([p for p in out_dir.rglob("*") if p.is_file()])
        ok = r.returncode == 0 and n_out == n_in
        detail = (f"exit={r.returncode}  inputs={n_in}  outputs={n_out}\n"
                  + (r.stdout or "") + (r.stderr or ""))
        return ok, detail
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "submission")
    ap.add_argument("--test-input", type=pathlib.Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    missing = copy_all(args.out)
    files = sorted(p for p in args.out.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)

    print(f"=== packaged {len(files)} files into {args.out} "
          f"({total/1e6:.1f} MB) ===")
    groups: dict[str, list[pathlib.Path]] = {}
    for p in files:
        groups.setdefault(p.parent.relative_to(args.out).as_posix() or ".",
                          []).append(p)
    for d, ps in sorted(groups.items()):
        print(f"  {d}/")
        for p in ps:
            print(f"      {p.name:<28} {p.stat().st_size/1e6:>8.2f} MB")

    if missing:
        print("\n! MISSING (not copied):")
        for m in missing:
            print(f"    {m}")

    print("\n=== deliverable checklist ===")
    checks = [
        ("1. Evaluation script (standalone, 2 CLI args)",
         (args.out / "evaluate.py").exists()),
        ("2. Training script (reproduces the model)",
         (args.out / "src" / "train.py").exists()),
        ("3. Denoised test outputs",
         False),  # produced at submission time from the real test set
        ("4. Environment spec (full pip freeze)",
         (args.out / "requirements.txt").exists()),
    ]
    for label, ok in checks:
        mark = "OK " if ok else "PENDING"
        print(f"  [{mark}] {label}")
    print("        (3) is produced by running evaluate.py on the real test set")

    if args.skip_test:
        return 0 if not missing else 1

    test_input = args.test_input
    tmp_in = None
    if test_input is None:
        # Build a tiny throwaway input set so the isolation test always has something.
        import cv2
        import numpy as np
        tmp_in = pathlib.Path(tempfile.mkdtemp(prefix="kla_pkg_in_"))
        rng = np.random.default_rng(0)
        for i, hw in enumerate([(128, 128), (128, 128), (256, 256)]):
            img = cv2.GaussianBlur(rng.random(hw).astype(np.float32), (0, 0), 2.0)
            cv2.imwrite(str(tmp_in / f"t{i}.png"),
                        (np.clip(img, 0, 1) * 255).astype(np.uint8))
        test_input = tmp_in

    print(f"\n=== isolation test: running evaluate.py from inside {args.out.name}/ ===")
    print(f"    cwd = package root, PYTHONPATH cleared, device = {args.device}")
    try:
        ok, detail = isolation_test(args.out, test_input, args.device)
    finally:
        if tmp_in:
            shutil.rmtree(tmp_in, ignore_errors=True)

    for line in detail.strip().splitlines():
        print(f"    {line}")
    print("\n" + ("ISOLATION TEST PASSED - the package is self-contained"
                  if ok else "ISOLATION TEST FAILED"))
    return 0 if (ok and not missing) else 1


if __name__ == "__main__":
    sys.exit(main())
