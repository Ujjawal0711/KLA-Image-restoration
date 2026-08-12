"""Render GT / degraded / histogram triptychs in the same layout as the deck's figures.

The whole approach rests on the §0.3 degradation model being right, and that model was
reverse-engineered from two figure titles. This produces the directly comparable
picture: if our synthetic NoisyLR does not look like the deck's NoisyLR, and the
histograms do not spread the same way, the premise is wrong and everything downstream
inherits the error.

    python scripts/preview_degradation.py --out degradation_preview.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.dataset import list_images, load_gray01  # noqa: E402
from src.degradation import degrade  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="data/raw/gt")
    ap.add_argument("--out", default="degradation_preview.png")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = list_images(pathlib.Path(args.gt_dir))
    if not paths:
        print(f"no images in {args.gt_dir}")
        return 1
    rng = np.random.default_rng(args.seed)

    # Deck-reported parameters, so this is as close to their figure as we can get.
    deck = [{"L": 16.86, "sigma": 0.008594}, {"L": 18.13, "sigma": 0.001065}]

    fig, axes = plt.subplots(args.rows, 3, figsize=(13, 3.4 * args.rows))
    picked = rng.choice(len(paths), size=args.rows * 4, replace=False)

    r = 0
    for i in picked:
        if r >= args.rows:
            break
        img = load_gray01(paths[int(i)])
        if img is None or min(img.shape[:2]) < 512:
            continue
        h, w = img.shape[:2]
        y, x = (h - 512) // 2, (w - 512) // 2
        hr = np.clip(img[y:y + 512, x:x + 512], 0, 1).astype(np.float32)

        p = dict(deck[r % len(deck)])
        p.update({"kernel": "area", "scale": 2})
        lr = degrade(hr, p, rng)

        axes[r][0].imshow(hr, cmap="gray", vmin=0, vmax=1)
        axes[r][0].set_title(f"GT (512x512)  [{paths[int(i)].stem}]", fontsize=9)
        axes[r][1].imshow(lr, cmap="gray")
        axes[r][1].set_title(f"NoisyLR (256x256)  L={p['L']}, "
                             f"$\\sigma$={p['sigma']}", fontsize=9)
        axes[r][2].hist(hr.ravel(), bins=120, alpha=0.6, density=True, label="GT")
        axes[r][2].hist(lr.ravel(), bins=120, alpha=0.6, density=True, label="NoisyLR")
        axes[r][2].set_title(f"Intensity histogram  (LR max = {lr.max():.3f})",
                             fontsize=9)
        axes[r][2].legend(fontsize=8)
        axes[r][2].set_xlabel("Intensity Value", fontsize=8)
        for c in (0, 1):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
        r += 1

    fig.suptitle("Synthetic degradation vs. the deck's model "
                 "(GT | NoisyLR | histogram)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")
    print("Compare against the deck figures: NoisyLR must show visible speckle and the")
    print("histogram must extend past the GT's maximum.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
