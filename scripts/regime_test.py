"""Does the model handle the 256->512 regime as well as the 128->256 it trained on?

Training crops are 128 LR -> 256 HR. The competition also contains 256 LR -> 512 HR.
The network is fully convolutional so it runs at any size, but it has never SEEN an
input that large, and receptive field relative to image content differs.

If the larger regime scores materially worse, fine-tuning on larger crops is worth the
hour. If not, it is not. Measured on the same source images so content is controlled --
only the crop size differs.

    python scripts/regime_test.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics as M  # noqa: E402
from src.dataset import list_images, load_gray01, split_by_source  # noqa: E402
from src.degradation import DegradationConfig, degrade  # noqa: E402
from src.models import build_from_arch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "checkpoints" / "model.pt"))
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = build_from_arch(ck.get("arch"))
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    model = model.to(dev).eval()

    paths = list_images(ROOT / "data" / "raw" / "gt")
    _, val = split_by_source(paths, 0.05, 0)

    # Only images big enough for BOTH regimes, so the comparison uses identical sources.
    usable = []
    for p in val:
        img = load_gray01(p)
        if img is not None and min(img.shape[:2]) >= 512:
            usable.append(img)
        if len(usable) >= args.n:
            break
    print(f"{len(usable)} source images large enough for both regimes\n")
    if not usable:
        print("none found — corpus images are smaller than 512")
        return 1

    lp = None
    try:
        lp = M.LPIPS()
    except Exception:
        pass

    cfg = DegradationConfig()
    print(f"{'regime':<20}{'pSNR':>9}{'SSIM':>9}{'LPIPS':>9}")
    print("-" * 47)
    out = {}
    for hr_size, label in [(256, "128->256 (trained)"), (512, "256->512 (unseen)")]:
        rows = []
        rng = np.random.default_rng(1234)
        for img in usable:
            h, w = img.shape[:2]
            y, x = (h - hr_size) // 2, (w - hr_size) // 2
            hr = np.clip(img[y:y + hr_size, x:x + hr_size], 0, 1).astype(np.float32)
            p = cfg.sample(rng)
            p["scale"] = 2
            lr = degrade(hr, p, rng)
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=dev.type == "cuda"):
                    yy = model(torch.from_numpy(lr[None, None]).to(dev))
                yy = yy.float().clamp_(0, 1)
            pred = yy.cpu().numpy()[0, 0]
            r = {"psnr": M.psnr(pred, hr), "ssim": M.ssim(pred, hr)}
            if lp:
                r["lpips"] = lp(pred, hr)
            rows.append(r)
        a = M.aggregate(rows)
        out[label] = a
        lpv = f"{a.get('lpips', float('nan')):9.4f}"
        print(f"{label:<20}{a['psnr']:9.3f}{a['ssim']:9.4f}{lpv}")

    k1, k2 = "128->256 (trained)", "256->512 (unseen)"
    d_ssim = out[k2]["ssim"] - out[k1]["ssim"]
    d_psnr = out[k2]["psnr"] - out[k1]["psnr"]
    print(f"\n  difference (unseen regime minus trained): "
          f"pSNR {d_psnr:+.3f}  SSIM {d_ssim:+.4f}")
    print("\n  NOTE: larger crops contain more context and are not intrinsically")
    print("  equally hard, so some difference is expected either way. What matters")
    print("  is whether the larger regime is markedly WORSE.")
    if d_ssim < -0.01:
        print("\n  ! The unseen regime is worse. Fine-tuning on larger crops is worth it.")
    else:
        print("\n  The model handles the larger regime fine. Larger-crop fine-tuning")
        print("  is not indicated by this measurement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
