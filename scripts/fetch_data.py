"""Download and normalise clean ground-truth images.

No training pairs were released, so every GT image is sourced externally. The deck
permits this explicitly ("synthetic data generation is explicitly encouraged").

Corpora, chosen for what each contributes:
  - EPFL electron microscopy : real EM imagery, the closest public analogue to
                               semiconductor inspection. Domain match.
  - DTD (describable textures): 47 texture families. Structural diversity, which is
                               what the OOD half of the test set is probing.
  - procedural                : dendrite / grain / wafer-lattice structures matching
                               the two samples shown in the deck (see src/procedural.py)

Everything is written as single-channel 16-bit PNG. Greyscale because the deck's
samples are single-channel; 16-bit because 8-bit quantisation of a [0,1] GT throws away
precision that the speckle model then amplifies.

    python scripts/fetch_data.py [--out data/raw/gt] [--skip-download]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import urllib.request

import cv2
import numpy as np

HF = "https://huggingface.co/datasets"
SOURCES = {
    "em": [
        f"{HF}/hasangoni/Electron_microscopy_dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        f"{HF}/hasangoni/Electron_microscopy_dataset/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
    ],
    "dtd": [
        f"{HF}/cansa/Describable-Textures-Dataset-DTD/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        f"{HF}/cansa/Describable-Textures-Dataset-DTD/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet",
    ],
    # Materials-science micrographs (SEM/TEM) scraped from publication figures.
    # Needs the strict filter — the corpus also contains plots and schematics.
    "mat": [
        f"{HF}/kvriza8/microscopy_images/resolve/refs%2Fconvert%2Fparquet/default/train/000{i}.parquet"
        for i in range(3)
    ],
}

# Corpora that need the plot/diagram rejection filter.
STRICT = {"mat"}

MIN_SIDE = 256  # HR crops are 256x256, so anything smaller is unusable


def looks_like_micrograph(x: np.ndarray) -> bool:
    """Reject plots, charts and line diagrams from paper-figure corpora.

    Some microscopy datasets are scraped from publication figures, so they contain
    graphs and schematics alongside real micrographs. Those have large flat white
    regions and few distinct intensity levels — training on them would teach the model
    to reconstruct chart axes, and they would inflate every average the way the early
    procedural images did (SSIM 0.9337 against real EM's 0.6282).

    `x` is float32 in [0,1].
    """
    if float(np.mean(x > 0.95)) > 0.22:
        return False                      # mostly white page -> a figure, not an image

    # Largest CONTIGUOUS near-white blob. This is the discriminator that matters:
    # plot margins, table backgrounds and the gutters between panels of a composite
    # figure are all one big connected white region. Real micrographs essentially never
    # contain one. Visual inspection of the first pass showed the fraction-based test
    # alone let through an XRD plot, a table of numbers and several multi-panel figures.
    white = (x > 0.93).astype(np.uint8)
    if white.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
        if n > 1:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
            if largest > 0.06 * x.size:
                return False

    # Same test for a uniform dark background (dark-field plots, black-matted panels).
    dark = (x < 0.04).astype(np.uint8)
    if dark.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        if n > 1 and int(stats[1:, cv2.CC_STAT_AREA].max()) > 0.35 * x.size:
            return False

    hist = np.histogram(x, bins=64, range=(0.0, 1.0))[0].astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    if float(-(p * np.log2(p)).sum()) < 4.0:
        return False                      # too few distinct levels -> line art
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    if float(np.mean(np.hypot(gx, gy))) < 0.015:
        return False                      # essentially featureless
    return True


def _is_parquet(p: pathlib.Path) -> bool:
    """A complete parquet file starts AND ends with the magic 'PAR1'. Checking only the
    header is not enough — a truncated download keeps a valid header, which is exactly
    how a half-finished file slipped through and got cached as if it were good."""
    try:
        if p.stat().st_size < 8:
            return False
        with open(p, "rb") as f:
            head = f.read(4)
            f.seek(-4, 2)
            return head == b"PAR1" and f.read(4) == b"PAR1"
    except OSError:
        return False


def download(url: str, dest: pathlib.Path, retries: int = 12) -> pathlib.Path:
    """Resumable download.

    The HF CDN drops these multi-hundred-MB connections partway through fairly
    reliably. Restarting from byte 0 on each retry never converges — it just burns
    bandwidth and hits the same drop. Instead the partial file is kept and each attempt
    resumes with a Range request, so every attempt makes forward progress.

    Falls back to a clean restart if the server ignores Range (responds 200 not 206).
    """
    if _is_parquet(dest):
        print(f"  cached  {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    if dest.exists():
        print(f"  {dest.name} present but incomplete — refetching")
        dest.unlink()

    tmp = dest.with_suffix(".part")
    for attempt in range(1, retries + 1):
        pos = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "kla-hackathon/1.0"}
        if pos:
            headers["Range"] = f"bytes={pos}-"
        print(f"  {dest.name} attempt {attempt}/{retries}"
              f"{f' (resume at {pos/1e6:.0f} MB)' if pos else ''} ...", flush=True)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r:
                partial = r.status == 206
                if pos and not partial:
                    pos = 0  # server ignored Range; start over
                total = int(r.headers.get("Content-Length") or 0) + (pos if partial else 0)
                done = pos
                with open(tmp, "ab" if pos else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r    {done / 1e6:7.0f} / {total / 1e6:.0f} MB",
                                  end="", flush=True)
            print()
            # urllib returns b"" on a dropped connection exactly as on a clean EOF,
            # so a short read is otherwise completely silent.
            if total and done < total:
                raise OSError(f"short read: {done} of {total} bytes")
            tmp.rename(dest)
            if not _is_parquet(dest):
                dest.unlink()
                raise OSError("not a valid parquet after download")
            return dest
        except Exception as e:  # noqa: BLE001
            print(f"    failed: {type(e).__name__}: {e}")
            time.sleep(min(2 ** min(attempt, 5), 30))
    raise RuntimeError(f"could not download {url} after {retries} attempts")


def to_gray_u16(img: np.ndarray, strict: bool = False) -> np.ndarray | None:
    """Any decoded image -> single-channel uint16, contrast-normalised to full range.

    Per-image min/max normalisation is deliberate: the degradation model is defined on
    a [0,1] GT, and source corpora arrive with wildly different exposure. Stretching
    each image to the full range makes 'brightness' mean the same thing everywhere,
    which matters because speckle strength scales with brightness.
    """
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    if min(img.shape[:2]) < MIN_SIDE:
        return None
    x = img.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return None  # flat image, no signal
    x = (x - lo) / (hi - lo)
    if strict and not looks_like_micrograph(x):
        return None
    return (x * 65535.0).round().astype(np.uint16)


def extract_parquet(path: pathlib.Path, out_dir: pathlib.Path, prefix: str,
                    start_idx: int, strict: bool = False) -> int:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    col = "image" if "image" in pf.schema_arrow.names else pf.schema_arrow.names[0]
    n = start_idx
    kept = skipped = 0
    for batch in pf.iter_batches(batch_size=64, columns=[col]):
        for item in batch.column(0).to_pylist():
            raw = item.get("bytes") if isinstance(item, dict) else item
            if not raw:
                skipped += 1
                continue
            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            g = to_gray_u16(arr, strict=strict)
            if g is None:
                skipped += 1
                continue
            cv2.imwrite(str(out_dir / f"{prefix}_{n:06d}.png"), g)
            n += 1
            kept += 1
    print(f"    kept {kept}, skipped {skipped} (too small / flat / undecodable)")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/raw/gt"))
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("data/raw/_parquet"))
    ap.add_argument("--only", nargs="*", default=None, help="subset of: em dtd")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    names = args.only or list(SOURCES)
    for name in names:
        print(f"\n[{name}]")
        idx = 0
        strict = name in STRICT
        if strict:
            print("  (strict filter on: rejecting plots/diagrams/flat images)")
        for i, url in enumerate(SOURCES[name]):
            p = download(url, args.cache / f"{name}_{i}.parquet")
            idx = extract_parquet(p, args.out, name, idx, strict=strict)
        print(f"  -> {idx} images from {name}")

    total = len(list(args.out.glob("*.png")))
    print(f"\nGT images now in {args.out.resolve()}: {total}")
    print("Next: python scripts/make_procedural.py   (adds dendrite/grain/wafer structures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
