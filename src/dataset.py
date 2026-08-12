"""Training dataset: clean GT images in, (degraded LR, clean HR) pairs out.

Because no paired data was released, EVERY pair is synthesised on the fly from a clean
GT image using src/degradation.py. That inverts the usual risk: instead of overfitting
to one fixed noise level, the model sees a fresh draw of (kernel, L, sigma) every time
it sees an image, and never sees the same degradation twice.

Two design points that matter more than usual here:

1. **Both scale regimes, one model.** A 256x256 HR crop with a 128x128 LR input covers
   the 256->128 regime directly, and the 512->256 regime is the identical operation at
   a different absolute size. Because the network is fully convolutional and the factor
   is always exactly 2, one set of weights serves both. Training on crops rather than
   whole images is what makes that true.

2. **No intensity augmentation.** Brightness/contrast jitter is standard elsewhere and
   is wrong here: speckle magnitude is tied to absolute intensity (Var = c^2/L), so
   rescaling intensities after degradation teaches a noise model that does not exist.
   Only geometric augmentation (flips, 90-degree rotations) is safe — it commutes with
   a per-pixel noise process.
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
from torch.utils.data import Dataset

from .degradation import DegradationConfig, degrade

IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def load_gray01(path: pathlib.Path) -> np.ndarray | None:
    """Load as float32 [0,1], single channel."""
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        return None
    if a.ndim == 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    return a.astype(np.float32)


def list_images(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def split_by_source(paths: list[pathlib.Path], val_frac: float = 0.05,
                    seed: int = 0) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """Hold out whole images, stratified by corpus prefix.

    Stratified because the corpora have very different statistics — a random split
    could hand validation an unrepresentative mix and make the number meaningless.
    """
    rng = np.random.default_rng(seed)
    groups: dict[str, list[pathlib.Path]] = {}
    for p in paths:
        groups.setdefault(p.stem.split("_")[0], []).append(p)
    train, val = [], []
    for _, items in sorted(groups.items()):
        items = sorted(items)
        idx = rng.permutation(len(items))
        n_val = max(1, int(round(len(items) * val_frac)))
        val += [items[i] for i in idx[:n_val]]
        train += [items[i] for i in idx[n_val:]]
    return sorted(train), sorted(val)


class RestorationDataset(Dataset):
    """Yields (lr, hr) float32 tensors shaped (1, hr//2, hr//2) and (1, hr, hr)."""

    def __init__(self, paths: list[pathlib.Path], hr_size: int = 256,
                 cfg: DegradationConfig | None = None, train: bool = True,
                 length: int | None = None, seed: int = 0,
                 corpus_weights: dict[str, float] | None = None):
        self.paths = list(paths)
        self.hr = hr_size
        self.cfg = cfg or DegradationConfig()
        self.train = train
        self.seed = seed
        # Decoupling epoch length from image count lets the sampler revisit images with
        # fresh degradations, which is the whole point of synthesising on the fly.
        self.length = length if length is not None else len(self.paths)
        if not self.paths:
            raise ValueError("no GT images found")

        # Sampling probability per image, so corpora can be weighted independently of
        # how many files each happens to contain.
        #
        # Why this matters: measured per-corpus SSIM is 0.9337 for `proc` against 0.6282
        # for `em`. The procedural images are largely predictable and contribute little
        # gradient, while electron microscopy is both the hardest and the closest
        # analogue to real wafer inspection. Uniform file sampling also let `dtd`
        # dominate simply by having the most files.
        self.p = None
        if train and corpus_weights:
            groups = [p.stem.split("_")[0] for p in self.paths]
            counts: dict[str, int] = {}
            for g in groups:
                counts[g] = counts.get(g, 0) + 1
            w = np.array([corpus_weights.get(g, 1.0) / max(counts[g], 1)
                          for g in groups], dtype=np.float64)
            if w.sum() > 0:
                self.p = w / w.sum()

    def __len__(self) -> int:
        return self.length

    def _rng(self, idx: int) -> np.random.Generator:
        # Validation must be identical on every epoch or the metric moves for reasons
        # that have nothing to do with the model. Training stays fully random.
        if self.train:
            return np.random.default_rng()
        return np.random.default_rng(self.seed * 1_000_003 + idx)

    def _crop(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = img.shape[:2]
        s = self.hr
        if h < s or w < s:
            # Reflect rather than zero-pad: a black border is a hard edge the model
            # would learn to reconstruct, and it does not exist in real images.
            img = cv2.copyMakeBorder(img, 0, max(0, s - h), 0, max(0, s - w),
                                     cv2.BORDER_REFLECT_101)
            h, w = img.shape[:2]
        if self.train:
            y = int(rng.integers(0, h - s + 1))
            x = int(rng.integers(0, w - s + 1))
        else:
            y, x = (h - s) // 2, (w - s) // 2
        return img[y:y + s, x:x + s]

    @staticmethod
    def _geometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < 0.5:
            img = img[:, ::-1]
        if rng.random() < 0.5:
            img = img[::-1, :]
        k = int(rng.integers(0, 4))
        if k:
            img = np.rot90(img, k)
        return np.ascontiguousarray(img)

    def __getitem__(self, idx: int):
        import torch

        rng = self._rng(idx)
        if not self.train:
            path = self.paths[idx % len(self.paths)]
        elif self.p is not None:
            path = self.paths[int(rng.choice(len(self.paths), p=self.p))]
        else:
            path = self.paths[int(rng.integers(0, len(self.paths)))]

        img = load_gray01(path)
        if img is None:  # unreadable file: fall back rather than crash a training run
            img = np.zeros((self.hr, self.hr), np.float32)

        hr = self._crop(img, rng)
        if self.train:
            hr = self._geometric(hr, rng)
        hr = np.clip(hr, 0.0, 1.0).astype(np.float32)

        params = self.cfg.sample(rng)
        params["scale"] = self.cfg.scale
        lr = degrade(hr, params, rng)

        return (torch.from_numpy(lr[None].copy()),
                torch.from_numpy(hr[None].copy()))


def build_datasets(gt_dir: str | pathlib.Path, hr_size: int = 256,
                   cfg: DegradationConfig | None = None, val_frac: float = 0.05,
                   train_len: int | None = None, seed: int = 0,
                   corpus_weights: dict[str, float] | None = None):
    paths = list_images(pathlib.Path(gt_dir))
    if not paths:
        raise FileNotFoundError(f"no images under {gt_dir}")
    tr, va = split_by_source(paths, val_frac, seed)
    train = RestorationDataset(tr, hr_size, cfg, train=True, length=train_len,
                               seed=seed, corpus_weights=corpus_weights)
    # Validation stays UNweighted so the metric keeps describing the whole corpus and
    # stays comparable with every number measured before weighting was introduced.
    val = RestorationDataset(va, hr_size, cfg, train=False, seed=seed)
    return train, val
