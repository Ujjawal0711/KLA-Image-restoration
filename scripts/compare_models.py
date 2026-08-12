"""Score several checkpoints on identical data with identical degradation.

Necessary because each training run validates against its OWN degradation config. A
model trained over a wider noise range validates on a different distribution, so its
reported SSIM is not comparable with another run's. Here every model sees byte-identical
inputs, drawn from the DEFAULT config so the numbers line up with the published
baselines too.

    python scripts/compare_models.py checkpoints/model.pt checkpoints/model_wide.pt
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import metrics as M  # noqa: E402
from src.dataset import build_datasets  # noqa: E402
from src.degradation import DegradationConfig  # noqa: E402


def load_model(path: pathlib.Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", {"name": "unet", "base": 48})
    from src.models import build_from_arch
    m = build_from_arch(arch)
    m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    n = sum(p.numel() for p in m.parameters())
    return m.to(device).eval(), arch, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=pathlib.Path)
    ap.add_argument("--gt-dir", default="data/raw/gt")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--balanced", action="store_true",
                    help="draw equally from each corpus instead of uniformly by file")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default config + fixed seed => every model sees byte-identical inputs.
    _, val = build_datasets(args.gt_dir, hr_size=256, cfg=DegradationConfig(), seed=0)

    if args.balanced:
        # Uniform-by-file sampling makes the score a popularity contest between
        # corpora: `dtd` alone is 42% of the files, so a model is largely graded on
        # generic photographic textures — the corpus LEAST like wafer inspection.
        # Equal draws per corpus make every image family count the same, which is a
        # far better proxy for a test set of unknown composition.
        import collections
        groups: dict[str, list[int]] = collections.defaultdict(list)
        for i, p in enumerate(val.paths):
            groups[p.stem.split("_")[0]].append(i)
        per = max(1, args.n // len(groups))
        idx = []
        for g in sorted(groups):
            idx.extend(groups[g][:per])
        pairs = [val[i] for i in idx]
        print(f"scoring {len(pairs)} samples, BALANCED across "
              f"{len(groups)} corpora ({per} each): {', '.join(sorted(groups))}\n")
    else:
        n = min(args.n, len(val))
        pairs = [val[i] for i in range(n)]
        print(f"scoring {n} identical validation samples "
              f"(uniform by file — note this is {len(val.paths)} files, "
              f"corpus-imbalanced)\n")

    lp = None
    try:
        lp = M.LPIPS()
    except Exception:
        pass

    rows = []

    # bicubic floor on exactly the same inputs
    b = []
    for lr_t, hr_t in pairs:
        lr, hr = lr_t.numpy()[0], hr_t.numpy()[0]
        up = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
        up = np.clip(up, 0, 1).astype(np.float32)
        r = {"psnr": M.psnr(up, hr), "ssim": M.ssim(up, hr)}
        if lp:
            r["lpips"] = lp(up, hr)
        b.append(r)
    rows.append(("bicubic floor", "-", M.aggregate(b)))

    for ckpt in args.checkpoints:
        model, arch, npar = load_model(ckpt, dev)
        acc = []
        for lr_t, hr_t in pairs:
            x = lr_t[None].to(dev)
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=dev.type == "cuda"):
                    y = model(x)
                y = y.float().clamp_(0, 1)
            pred = y.cpu().numpy()[0, 0]
            hr = hr_t.numpy()[0]
            r = {"psnr": M.psnr(pred, hr), "ssim": M.ssim(pred, hr)}
            if lp:
                r["lpips"] = lp(pred, hr)
            acc.append(r)
        rows.append((ckpt.name, f"{arch['name']} b{arch.get('base')} "
                                f"{npar/1e6:.1f}M", M.aggregate(acc)))
        del model
        torch.cuda.empty_cache()

    print(f"{'checkpoint':<22}{'arch':<22}{'pSNR':>9}{'SSIM':>9}{'LPIPS':>9}")
    print("-" * 71)
    for name, arch, a in rows:
        lpv = f"{a.get('lpips', float('nan')):9.4f}"
        print(f"{name:<22}{arch:<22}{a['psnr']:9.3f}{a['ssim']:9.4f}{lpv}")

    if len(rows) > 2:
        base = rows[1][2]
        print(f"\ndeltas vs {rows[1][0]}:")
        for name, _, a in rows[2:]:
            print(f"  {name:<22} pSNR {a['psnr']-base['psnr']:+7.3f}   "
                  f"SSIM {a['ssim']-base['ssim']:+7.4f}   "
                  f"LPIPS {a.get('lpips',0)-base.get('lpips',0):+7.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
