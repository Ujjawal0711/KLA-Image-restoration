"""Phase 1 hardening — exercise evaluate.py against inputs that could crash it.

These are the failures that score ZERO rather than "slightly worse": a crash partway
through the test set loses every remaining image. Runs on CPU by default so it does
not contend with a training run.

    python scripts/edge_cases.py [--device cpu]
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build_cases(root: pathlib.Path) -> dict[str, int]:
    """Each case is a directory; value is how many images evaluate.py should emit."""
    rng = np.random.default_rng(0)
    cases: dict[str, int] = {}

    def img(h, w):
        return np.clip(cv2.GaussianBlur(
            rng.random((h, w)).astype(np.float32), (0, 0), 2.0), 0, 1)

    # 1. single image
    d = root / "single"; d.mkdir(parents=True)
    cv2.imwrite(str(d / "a.png"), (img(128, 128) * 255).astype(np.uint8))
    cases["single"] = 1

    # 2. non-square
    d = root / "nonsquare"; d.mkdir(parents=True)
    cv2.imwrite(str(d / "a.png"), (img(120, 200) * 255).astype(np.uint8))
    cases["nonsquare"] = 1

    # 3. odd dimensions (not divisible by the network's downsampling factor)
    d = root / "odd"; d.mkdir(parents=True)
    for i, (h, w) in enumerate([(97, 131), (63, 63), (129, 129)]):
        cv2.imwrite(str(d / f"o{i}.png"), (img(h, w) * 255).astype(np.uint8))
    cases["odd"] = 3

    # 4. RGB input (must be handled, not rejected)
    d = root / "rgb"; d.mkdir(parents=True)
    c = (np.stack([img(128, 128)] * 3, -1) * 255).astype(np.uint8)
    cv2.imwrite(str(d / "c.png"), c)
    cases["rgb"] = 1

    # 5. mixed dtypes and extensions together
    d = root / "mixed"; d.mkdir(parents=True)
    cv2.imwrite(str(d / "u8.png"), (img(128, 128) * 255).astype(np.uint8))
    cv2.imwrite(str(d / "u16.png"), (img(128, 128) * 65535).astype(np.uint16))
    cv2.imwrite(str(d / "t.tif"), img(128, 128).astype(np.float32))
    cv2.imwrite(str(d / "j.jpg"), (img(128, 128) * 255).astype(np.uint8))
    cases["mixed"] = 4

    # 6. a corrupt file among good ones -- must skip it, not abort the run
    d = root / "corrupt"; d.mkdir(parents=True)
    cv2.imwrite(str(d / "good1.png"), (img(128, 128) * 255).astype(np.uint8))
    (d / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 20)
    cv2.imwrite(str(d / "good2.png"), (img(128, 128) * 255).astype(np.uint8))
    cases["corrupt"] = 2

    # 7. nested subdirectories
    d = root / "nested" / "sub"; d.mkdir(parents=True)
    cv2.imwrite(str(d / "n.png"), (img(128, 128) * 255).astype(np.uint8))
    cases["nested"] = 1

    # 8. empty directory -- should exit non-zero with a clear message, not traceback
    (root / "empty").mkdir(parents=True)
    cases["empty"] = 0

    # 9. large single image (well past the stated 256 max)
    d = root / "large"; d.mkdir(parents=True)
    cv2.imwrite(str(d / "L.png"), (img(512, 512) * 255).astype(np.uint8))
    cases["large"] = 1

    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default=str(ROOT / "checkpoints" / "model.pt"))
    args = ap.parse_args()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="kla_edge_"))
    try:
        cases = build_cases(tmp / "in")
        print(f"{'case':>12}  {'exit':>4}  {'out':>4}  {'want':>4}  verdict")
        print("  " + "-" * 52)
        failures = 0
        for name, want in cases.items():
            src = tmp / "in" / name
            dst = tmp / "out" / name
            r = subprocess.run(
                [sys.executable, str(ROOT / "evaluate.py"), str(src), str(dst),
                 "--weights", args.weights, "--device", args.device, "--quiet"],
                capture_output=True, text=True, timeout=600)
            got = len(list(dst.rglob("*"))) if dst.exists() else 0

            if name == "empty":
                ok = r.returncode != 0 and "Traceback" not in r.stderr
                verdict = "clean error" if ok else "FAIL (should exit non-zero cleanly)"
            else:
                ok = r.returncode == 0 and got == want
                verdict = "ok" if ok else "FAIL"
                if r.returncode != 0:
                    verdict = f"FAIL rc={r.returncode}"
            failures += 0 if ok else 1
            print(f"{name:>12}  {r.returncode:>4}  {got:>4}  {want:>4}  {verdict}")
            if not ok:
                err = (r.stderr or "").strip().splitlines()
                for line in err[-4:]:
                    print(f"                {line[:100]}")

        # shape correctness on the odd case: output must be exactly 2x input
        print("\nshape check (odd dimensions):")
        for p in sorted((tmp / "in" / "odd").glob("*.png")):
            q = tmp / "out" / "odd" / p.name
            if not q.exists():
                print(f"  {p.name}: MISSING")
                failures += 1
                continue
            a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            b = cv2.imread(str(q), cv2.IMREAD_UNCHANGED)
            want = (a.shape[0] * 2, a.shape[1] * 2)
            ok = b.shape[:2] == want
            failures += 0 if ok else 1
            print(f"  {p.name}: {a.shape[:2]} -> {b.shape[:2]} "
                  f"(want {want}) {'ok' if ok else 'FAIL'}")

        print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
        return 0 if failures == 0 else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
