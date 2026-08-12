"""Step 1 — data inspection.

The dataset layout is unknown until it is released, so this script discovers it
rather than assuming it: it walks the tree, groups files by directory and by
image shape, then tries several pairing strategies and reports which one fit.
Run it, read the output, and only then hard-code anything.

    python scripts/inspect_data.py [data_dir]
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

import numpy as np

IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".npy", ".npz"}

# Tokens that commonly mark the two sides of a pair, degraded first.
DEGRADED_TOKENS = ["noisylr", "noisy", "degraded", "lr", "low", "input", "in"]
CLEAN_TOKENS = ["gt", "groundtruth", "clean", "hr", "high", "target", "out"]
ALL_TOKENS = set(DEGRADED_TOKENS) | set(CLEAN_TOKENS)

_SPLIT = re.compile(r"[_\-.\s]+")


def load(path: pathlib.Path) -> np.ndarray | None:
    """Load without normalising — dtype and raw range are exactly what we want to see."""
    try:
        if path.suffix == ".npy":
            return np.load(path)
        if path.suffix == ".npz":
            with np.load(path) as z:
                return z[list(z.keys())[0]]
        import cv2

        # IMREAD_UNCHANGED preserves uint16 and any float TIFF; anything else silently
        # converts to 8-bit BGR and would hide the intensity range we are here to measure.
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return a
    except Exception as e:  # noqa: BLE001
        print(f"    ! failed to load {path.name}: {type(e).__name__}: {e}")
        return None


def dtype_scale(a: np.ndarray) -> float:
    """Divisor mapping a dtype's full range onto [0,1]. Applied per array, never
    per pair: degraded and GT may be stored at different bit depths, and using one
    array's scale for the other silently destroys the overflow measurement."""
    if a.dtype == np.uint8:
        return 255.0
    if a.dtype == np.uint16:
        return 65535.0
    return 1.0


def strip_tokens(stem: str) -> tuple[str, str | None]:
    """Return (key, matched_token) with a degraded/clean marker removed from the stem.

    Matches only *whole* delimiter-bounded segments. Plain substring matching is
    unusable here: "lr"/"in"/"out" occur inside ordinary words, so `train_0042`
    matches "in" and every file in the set gets labelled degraded — silently
    yielding zero pairs while looking like it worked.
    """
    low = stem.lower()
    segs = [s for s in _SPLIT.split(low) if s]

    # Two-segment tokens first ("ground_truth"), then single segments.
    for i in range(len(segs) - 1):
        if segs[i] + segs[i + 1] in ALL_TOKENS:
            return "_".join(segs[:i] + segs[i + 2:]), segs[i] + segs[i + 1]
    for i, s in enumerate(segs):
        if s in ALL_TOKENS:
            return "_".join(segs[:i] + segs[i + 1:]), s

    # Fallback: token glued to a number with no delimiter, e.g. "gt0001" / "0001lr".
    # Anchored to the full stem so ordinary words ("brain", "train") cannot match.
    for tok in sorted(ALL_TOKENS, key=len, reverse=True):
        m = re.fullmatch(rf"(\d*){re.escape(tok)}(\d*)", low)
        if m and (m.group(1) or m.group(2)):
            return m.group(1) + m.group(2), tok
    return low, None


def side_of(token: str | None, path: pathlib.Path) -> str | None:
    """Classify a file as 'degraded' or 'clean' from its filename token or its parent dir."""
    if token:
        return "degraded" if token in DEGRADED_TOKENS else "clean"
    parts = [p.lower() for p in path.parts]
    for p in reversed(parts[:-1]):
        if p in DEGRADED_TOKENS:
            return "degraded"
        if p in CLEAN_TOKENS:
            return "clean"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data/raw", type=pathlib.Path)
    ap.add_argument("--sample", type=int, default=200, help="files to open for stats")
    args = ap.parse_args()

    root = args.data_dir
    if not root.exists():
        print(f"ERROR: {root.resolve()} does not exist.")
        return 1

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"=== LAYOUT ===\nroot: {root.resolve()}\nimage files: {len(files)}")
    if not files:
        print("No image files found. Check the path / extensions.")
        return 1

    by_dir = collections.Counter(p.parent.relative_to(root).as_posix() or "." for p in files)
    print(f"\ndirectories ({len(by_dir)}):")
    for d, n in by_dir.most_common(30):
        print(f"  {n:>7}  {d}")
    if len(by_dir) > 30:
        print(f"  ... and {len(by_dir) - 30} more")

    print("\nextensions:", dict(collections.Counter(p.suffix.lower() for p in files)))
    print("\nexample filenames:")
    for p in files[:8]:
        print(f"  {p.relative_to(root).as_posix()}")

    # ---- shapes and dtypes over a sample -------------------------------------
    step = max(1, len(files) // args.sample)
    sample = files[::step][: args.sample]
    print(f"\n=== SHAPES / DTYPES (sampling {len(sample)} of {len(files)}) ===")

    shapes, dtypes, ranges = collections.Counter(), collections.Counter(), {}
    for p in sample:
        a = load(p)
        if a is None:
            continue
        shapes[a.shape] += 1
        dtypes[str(a.dtype)] += 1
        # Key on dtype as well as shape: mixing a float32 and a uint8 array of the
        # same resolution into one row reports "min=-0.01 max=255", which is
        # meaningless.
        ranges.setdefault((a.shape[:2], str(a.dtype)), []).append(
            (float(a.min()), float(a.max())))

    for s, n in shapes.most_common():
        print(f"  {n:>5}  {s}")
    print("dtypes:", dict(dtypes))

    print("\nintensity range by (resolution, dtype):")
    for (hw, dt), rs in sorted(ranges.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        lo = min(r[0] for r in rs)
        hi = max(r[1] for r in rs)
        print(f"  {str(hw):>12} {dt:>8}: min={lo:<12.6g} max={hi:<12.6g} (n={len(rs)})")

    # ---- pairing --------------------------------------------------------------
    print("\n=== PAIRING ===")
    groups: dict[str, dict[str, pathlib.Path]] = collections.defaultdict(dict)
    unclassified = []
    for p in files:
        key, tok = strip_tokens(p.stem)
        side = side_of(tok, p)
        if side is None:
            unclassified.append(p)
            continue
        groups[key][side] = p

    complete = {k: v for k, v in groups.items() if len(v) == 2}
    print(f"complete pairs: {len(complete)}")
    print(f"half pairs:     {len(groups) - len(complete)}")
    print(f"unclassified:   {len(unclassified)}")
    if unclassified:
        print("  e.g.", ", ".join(p.name for p in unclassified[:5]))

    if not complete:
        print("\n! Pairing heuristics failed. Read the layout above and pair explicitly.")
        return 0

    # ---- scale regimes and the range-overflow claim ---------------------------
    regimes = collections.Counter()
    overflow_checked = overflow_seen = dtype_mismatch = 0
    overflow_by_dtype: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    deg_max_over_gt: list[float] = []

    for key in list(complete)[: args.sample]:
        d, c = load(complete[key]["degraded"]), load(complete[key]["clean"])
        if d is None or c is None:
            continue
        regimes[(d.shape[:2], c.shape[:2])] += 1

        dn = d.astype(np.float64) / dtype_scale(d)
        cn = c.astype(np.float64) / dtype_scale(c)
        if d.dtype != c.dtype:
            dtype_mismatch += 1
        overflow_checked += 1
        over = dn.max() > cn.max() + 1e-9
        overflow_seen += int(over)
        overflow_by_dtype[str(d.dtype)][int(over)] += 1
        deg_max_over_gt.append(dn.max() / max(cn.max(), 1e-9))

    print("\nscale regimes (degraded_hw -> clean_hw):")
    for (dhw, chw), n in regimes.most_common():
        factor = chw[0] / dhw[0] if dhw[0] else 0
        print(f"  {n:>6}  {dhw} -> {chw}   factor {factor:g}")

    if dtype_mismatch:
        print(f"\n! {dtype_mismatch} pairs store degraded and GT at different dtypes — "
              f"note this, it affects normalisation in evaluate.py")

    print(f"\nrange overflow: {overflow_seen}/{overflow_checked} pairs have "
          f"max(degraded) > max(GT)")
    # Broken down by dtype because this answers NOTES.md open question #5: an integer
    # dtype cannot represent values above its own max, so storing a speckled image as
    # uint8 either clips the overflow away or rescales it. Whether the overflow the
    # deck promises actually survives to disk decides how evaluate.py must normalise.
    for dt, (no, yes) in sorted(overflow_by_dtype.items()):
        print(f"    degraded stored as {dt:>8}: {yes} overflow / {no + yes} pairs")
        if yes == 0 and dt.startswith("uint"):
            print(f"      -> {dt} pairs show NO overflow: quantisation clipped or "
                  f"rescaled it.")
            print("         Do not assume inputs exceed 1.0 for these; confirm the "
                  "intended scale.")
    if deg_max_over_gt:
        r = np.array(deg_max_over_gt)
        print(f"  max(deg)/max(GT):  median={np.median(r):.4f}  "
              f"p95={np.percentile(r, 95):.4f}  max={r.max():.4f}")
        if overflow_seen == 0:
            print("  ! No overflow found. Either the data is clipped/quantised on disk,")
            print("    or the pairing is wrong. Both invalidate NOTES.md §0.3 — investigate.")

    # ---- histograms -----------------------------------------------------------
    print("\n=== HISTOGRAM (pooled over sampled pairs, normalised scale) ===")
    dh = np.zeros(60)
    ch = np.zeros(60)
    edges = np.linspace(0, 2.0, 61)
    for key in list(complete)[:50]:
        d, c = load(complete[key]["degraded"]), load(complete[key]["clean"])
        if d is None or c is None:
            continue
        dh += np.histogram(d.astype(np.float64).ravel() / dtype_scale(d), bins=edges)[0]
        ch += np.histogram(c.astype(np.float64).ravel() / dtype_scale(c), bins=edges)[0]
    dh /= max(dh.sum(), 1)
    ch /= max(ch.sum(), 1)
    print(f"{'bin':>12}  {'GT':>8}  {'degraded':>9}")
    for i in range(60):
        if ch[i] > 1e-4 or dh[i] > 1e-4:
            print(f"{edges[i]:>6.3f}-{edges[i+1]:<5.3f}  {ch[i]:>8.4f}  {dh[i]:>9.4f}")

    print("\nWrite these numbers into NOTES.md Step 1 before moving on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
