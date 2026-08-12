"""Write procedurally generated GT images alongside the downloaded corpora.

    python scripts/make_procedural.py --n 1500 --out data/raw/gt
    python scripts/make_procedural.py --contact-sheet   # visual check, writes a PNG grid
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.procedural import GENERATORS, random_image  # noqa: E402


def contact_sheet(out: pathlib.Path, size: int = 256, cols: int = 5, rows: int = 4):
    """One row per generator so the output can actually be eyeballed."""
    rng = np.random.default_rng(0)
    kinds = list(GENERATORS)
    tiles = []
    for k in kinds:
        row = [GENERATORS[k](size, rng) for _ in range(cols)]
        tiles.append(np.hstack(row))
    sheet = np.vstack(tiles)
    cv2.imwrite(str(out), (np.clip(sheet, 0, 1) * 255).astype(np.uint8))
    print(f"wrote {out}  ({len(kinds)} rows x {cols} cols, one row per generator)")
    for i, k in enumerate(kinds):
        print(f"  row {i}: {k}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/raw/gt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contact-sheet", action="store_true")
    args = ap.parse_args()

    if args.contact_sheet:
        contact_sheet(pathlib.Path("procedural_samples.png"))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    counts: dict[str, int] = {}
    for i in range(args.n):
        img, kind = random_image(args.size, rng)
        counts[kind] = counts.get(kind, 0) + 1
        cv2.imwrite(str(args.out / f"proc_{i:06d}.png"),
                    (np.clip(img, 0, 1) * 65535).round().astype(np.uint16))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{args.n}", flush=True)

    print(f"\nwrote {args.n} procedural images to {args.out.resolve()}")
    for k, v in sorted(counts.items()):
        print(f"  {k:>9}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
