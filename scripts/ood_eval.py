"""Score the shipped model on corpora it has NEVER seen.

Every number produced so far is measured on the three corpora that were in training.
That says nothing about the axis the competition explicitly scores: behaviour on
imagery from unfamiliar sources. This downloads microscopy datasets of different
modalities (brightfield, fluorescence -- neither is electron microscopy) and scores
the model on them cold.

    python scripts/ood_eval.py

Interpretation: a small drop versus the in-training corpora means the model learned
restoration rather than memorising image families. A large drop means the training
corpus needs broadening before submission.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_data import download, extract_parquet  # noqa: E402
from src import metrics as M  # noqa: E402
from src.dataset import RestorationDataset, list_images  # noqa: E402
from src.degradation import DegradationConfig  # noqa: E402

HF = "https://huggingface.co/datasets"
OOD_SOURCES = {
    # brightfield microscopy -- different illumination and contrast from EM
    "bright": [f"{HF}/mario-dg/brightfield-microscopy-scc-filtered/resolve/"
               f"refs%2Fconvert%2Fparquet/default/test/0000.parquet"],
    # fluorescence single-cell -- sparse bright structures on dark field
    "fluor": [f"{HF}/Jiabang/Single-cell_Microscopy_Images_for_Cancer_Classification/"
              f"resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "checkpoints" / "model.pt"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "raw" / "ood")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    cache = ROOT / "data" / "raw" / "_ood_parquet"
    cache.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        for name, urls in OOD_SOURCES.items():
            d = args.out / name
            if d.exists() and len(list(d.glob("*.png"))) > 50:
                print(f"[{name}] already extracted ({len(list(d.glob('*.png')))} images)")
                continue
            d.mkdir(parents=True, exist_ok=True)
            print(f"\n[{name}]")
            idx = 0
            for i, url in enumerate(urls):
                p = download(url, cache / f"{name}_{i}.parquet")
                idx = extract_parquet(p, d, name, idx)
            print(f"  -> {idx} images")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    arch = ck.get("arch", {"name": "unet", "base": 48})
    from src.models import build_from_arch
    model = build_from_arch(arch)
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    model = model.to(dev).eval()
    print(f"\nmodel: {arch}  device: {dev}")

    lp = None
    try:
        lp = M.LPIPS()
    except Exception:
        pass

    import cv2

    print(f"\n{'corpus':<12}{'seen?':<8}{'n':>5}{'pSNR':>9}{'SSIM':>9}{'LPIPS':>9}"
          f"{'bicubic SSIM':>14}")
    print("-" * 66)

    results = {}
    for name, seen in [("dtd", "train"), ("em", "train"), ("mat", "train"),
                       ("proc", "train"), ("bright", "UNSEEN"), ("fluor", "UNSEEN")]:
        d = args.out / name if seen == "UNSEEN" else ROOT / "data" / "raw" / "gt"
        paths = ([p for p in list_images(d)] if seen == "UNSEEN"
                 else [p for p in list_images(d) if p.stem.startswith(name)])
        if not paths:
            print(f"{name:<12}{seen:<8}    -   (no images found)")
            continue
        # Deterministic subset; validation-style (centre crop, fixed noise seed).
        rng = np.random.default_rng(0)
        sel = sorted(rng.choice(len(paths), size=min(args.n * 2, len(paths)),
                                replace=False))
        ds = RestorationDataset([paths[i] for i in sel], 256, DegradationConfig(),
                                train=False, seed=0)
        rows, brows = [], []
        n = min(args.n, len(ds))
        for i in range(n):
            lr_t, hr_t = ds[i]
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=dev.type == "cuda"):
                    y = model(lr_t[None].to(dev))
                y = y.float().clamp_(0, 1)
            pred, hr = y.cpu().numpy()[0, 0], hr_t.numpy()[0]
            r = {"psnr": M.psnr(pred, hr), "ssim": M.ssim(pred, hr)}
            if lp:
                r["lpips"] = lp(pred, hr)
            rows.append(r)
            up = np.clip(cv2.resize(lr_t.numpy()[0], (hr.shape[1], hr.shape[0]),
                                    interpolation=cv2.INTER_CUBIC), 0, 1)
            brows.append({"ssim": M.ssim(up.astype(np.float32), hr)})
        a = M.aggregate(rows)
        b = M.aggregate(brows)
        results[name] = (a, b, seen)
        lpv = f"{a.get('lpips', float('nan')):9.4f}"
        print(f"{name:<12}{seen:<8}{n:>5}{a['psnr']:9.3f}{a['ssim']:9.4f}{lpv}"
              f"{b['ssim']:14.4f}")

    seen_ssim = [v[0]["ssim"] for k, v in results.items() if v[2] == "train"]
    ood_ssim = [v[0]["ssim"] for k, v in results.items() if v[2] == "UNSEEN"]
    if seen_ssim and ood_ssim:
        print("-" * 66)
        print(f"  mean SSIM, seen corpora   : {np.mean(seen_ssim):.4f}")
        print(f"  mean SSIM, UNSEEN corpora : {np.mean(ood_ssim):.4f}")
        drop = (np.mean(seen_ssim) - np.mean(ood_ssim)) / np.mean(seen_ssim) * 100
        print(f"  drop                      : {drop:+.1f}%")
        # The gain over bicubic ON THE SAME IMAGES is the fairer signal: some corpora
        # are intrinsically harder for everyone, which a raw SSIM drop conflates.
        print("\n  gain over bicubic on the same images (the fairer comparison):")
        for k, (a, b, s) in results.items():
            print(f"    {k:<10}{s:<8} {a['ssim'] - b['ssim']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
