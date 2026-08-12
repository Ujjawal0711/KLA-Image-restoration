"""Procedurally generated ground-truth structures.

Downloaded corpora (textures, electron microscopy) supply realistic image statistics
but none of them are semiconductor inspection imagery. These generators fill the gap
by reproducing the structure types actually visible in the KLA deck's two sample
figures, plus two more that inspection imagery is full of:

    dendrite  - sparse bright branching filaments on near-black background (deck fig 2)
    granular  - dark base with bright granular clusters, fine grained     (deck fig 1)
    grain     - polycrystalline cell boundaries
    lattice   - periodic die / line-and-space patterns
    fibers    - long thin curved strands

Why bother when we have real photos: super-resolution has to learn what plausible
high-frequency detail looks like *for this domain*. A model trained only on natural
textures will hallucinate natural-texture detail into a wafer image. These also give
exact control over edge sharpness and sparsity, which is what the metrics react to.

All generators return float32 (H,W) in [0,1].
"""

from __future__ import annotations

import cv2
import numpy as np


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _fbm(size: int, rng: np.random.Generator, octaves=(1.0, 3.0, 8.0, 20.0),
         weights=(1.0, 0.6, 0.35, 0.2)) -> np.ndarray:
    """Fractal noise: sum of Gaussian-blurred white noise at several scales.
    Gives the 1/f-ish spectrum real imagery has, instead of flat white noise."""
    out = np.zeros((size, size), np.float32)
    for s, w in zip(octaves, weights):
        out += w * cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), s)
    return _norm(out)


def _substrate(size: int, rng: np.random.Generator) -> np.ndarray:
    """Faint textured background.

    Measured problem: the first version of these generators produced images that were
    largely pure black, and the model scored SSIM 0.9337 on them against 0.6282 on real
    electron microscopy. Flat black regions are trivially predictable, so those pixels
    contributed almost no gradient signal while inflating every average.

    Real inspection imagery always has *some* substrate structure. Adding a faint,
    spatially varying floor makes these images both more realistic and harder, without
    changing the bright structures that define each generator.
    """
    amp = float(rng.uniform(0.04, 0.16))
    fine = _fbm(size, rng, (1.0, 2.5, 6.0), (1.0, 0.6, 0.3))
    coarse = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0),
                              float(rng.uniform(14, 30)))
    return (amp * (0.65 * fine + 0.35 * _norm(coarse))).astype(np.float32)


# ---------------------------------------------------------------- dendrite
def _branch(img, y, x, ang, length, width, depth, rng):
    if depth <= 0 or length < 2.0:
        return
    y2 = y + length * np.sin(ang)
    x2 = x + length * np.cos(ang)
    cv2.line(img, (int(round(x)), int(round(y))), (int(round(x2)), int(round(y2))),
             float(rng.uniform(0.6, 1.0)), max(1, int(round(width))), cv2.LINE_AA)
    for _ in range(int(rng.integers(2, 4))):
        _branch(img, y2, x2, ang + float(rng.normal(0, 0.55)),
                length * float(rng.uniform(0.55, 0.82)), width * 0.72, depth - 1, rng)


def dendrite(size: int, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((size, size), np.float32)
    # Denser than the first version (3-8 trunks): sparse images left most of the frame
    # empty and therefore trivial to reconstruct.
    for _ in range(int(rng.integers(6, 14))):
        y = float(rng.integers(0, size))
        x = float(rng.integers(0, size))
        _branch(img, y, x, float(rng.uniform(0, 2 * np.pi)),
                size * float(rng.uniform(0.08, 0.18)),
                float(rng.uniform(1.5, 3.5)), int(rng.integers(5, 8)), rng)
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    return np.clip(img + _substrate(size, rng), 0, 1).astype(np.float32)


# ---------------------------------------------------------------- granular
def granular(size: int, rng: np.random.Generator) -> np.ndarray:
    fine = _fbm(size, rng, (0.8, 1.8, 4.0), (1.0, 0.7, 0.4))
    # Low-frequency mask decides *where* the bright grain clusters sit, matching the
    # deck's texture sample: dark field with localised bright granular patches.
    field = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0),
                             float(rng.uniform(10, 22)))
    field = _norm(field)
    mask = np.clip((field - np.percentile(field, float(rng.uniform(60, 80)))) * 6, 0, 1)
    img = fine ** float(rng.uniform(2.0, 3.2)) * 0.45 + mask * fine * float(rng.uniform(0.8, 1.4))
    return _norm(img)


# ---------------------------------------------------------------- grain
def grain(size: int, rng: np.random.Generator) -> np.ndarray:
    """Polycrystalline cells. Bright boundaries = where the two nearest seeds are
    equidistant, which is the Voronoi edge set."""
    n = int(rng.integers(18, 60))
    pts = rng.random((n, 2)).astype(np.float32) * size
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.sqrt((yy[..., None] - pts[:, 0]) ** 2 + (xx[..., None] - pts[:, 1]) ** 2)
    part = np.partition(d, 1, axis=2)
    edge = part[:, :, 1] - part[:, :, 0]
    w = float(rng.uniform(1.5, 4.0))
    img = np.exp(-edge / w).astype(np.float32)
    # faint per-cell intensity variation
    if rng.random() < 0.6:
        lab = np.argmin(d, axis=2)
        tint = rng.random(n).astype(np.float32) * 0.25
        img = np.clip(img + tint[lab], 0, 1)
    return np.clip(_norm(img) + _substrate(size, rng), 0, 1).astype(np.float32)


# ---------------------------------------------------------------- lattice
def lattice(size: int, rng: np.random.Generator) -> np.ndarray:
    """Periodic die / line-and-space patterns — the single most characteristic
    structure in wafer inspection imagery."""
    img = np.zeros((size, size), np.float32)
    pitch = int(rng.integers(14, 56))
    thick = max(1, int(pitch * float(rng.uniform(0.10, 0.30))))
    val = float(rng.uniform(0.6, 1.0))
    off = int(rng.integers(0, pitch))
    for i in range(-size, size * 2, pitch):
        cv2.line(img, (i + off, -size), (i + off, size * 2), val, thick)
    if rng.random() < 0.7:  # cross-hatch into a die grid
        pitch2 = pitch if rng.random() < 0.5 else int(rng.integers(14, 56))
        for i in range(-size, size * 2, pitch2):
            cv2.line(img, (-size, i + off), (size * 2, i + off), val, thick)
    if rng.random() < 0.5:  # scattered contacts / vias
        for _ in range(int(rng.integers(10, 60))):
            c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            cv2.circle(img, c, int(rng.integers(2, 6)), float(rng.uniform(0.5, 1.0)), -1)

    ang = float(rng.uniform(-30, 30))
    M = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
    img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.5, 1.2)))
    return np.clip(_norm(img) + _substrate(size, rng), 0, 1).astype(np.float32)


# ---------------------------------------------------------------- fibers
def fibers(size: int, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((size, size), np.float32)
    for _ in range(int(rng.integers(20, 60))):
        y, x = float(rng.integers(0, size)), float(rng.integers(0, size))
        ang = float(rng.uniform(0, 2 * np.pi))
        val = float(rng.uniform(0.4, 1.0))
        w = max(1, int(rng.integers(1, 4)))
        for _ in range(int(rng.integers(40, 200))):
            ang += float(rng.normal(0, 0.12))
            y2, x2 = y + 3 * np.sin(ang), x + 3 * np.cos(ang)
            cv2.line(img, (int(x), int(y)), (int(x2), int(y2)), val, w, cv2.LINE_AA)
            y, x = y2, x2
            if not (0 <= y < size and 0 <= x < size):
                break
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    return np.clip(img + _substrate(size, rng), 0, 1).astype(np.float32)


GENERATORS = {
    "dendrite": dendrite,
    "granular": granular,
    "grain": grain,
    "lattice": lattice,
    "fibers": fibers,
}

# Weighted toward the two structure types the deck actually showed.
WEIGHTS = {"dendrite": 0.25, "granular": 0.25, "grain": 0.2, "lattice": 0.2, "fibers": 0.1}


def random_image(size: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    kinds = list(GENERATORS)
    p = np.array([WEIGHTS[k] for k in kinds], dtype=np.float64)
    kind = str(rng.choice(kinds, p=p / p.sum()))
    return GENERATORS[kind](size, rng), kind
