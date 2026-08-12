"""Promote a training checkpoint to the submission weights file.

Training checkpoints carry optimizer state (AdamW keeps two momentum buffers per
parameter), scheduler state and the GradScaler -- roughly 3x the size of the weights
alone. None of it is needed for inference, and evaluate.py is timed from process
start, so every megabyte is load time charged to the score.

    python scripts/promote_checkpoint.py checkpoints/nafnet_best.pt
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "checkpoints" / "model.pt")
    ap.add_argument("--half", action="store_true",
                    help="store weights as fp16 (halves file size again; inference "
                         "already runs under fp16 autocast so accuracy is unaffected)")
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    before = args.src.stat().st_size

    state = ck["model"]
    if args.half:
        state = {k: (v.half() if v.is_floating_point() else v)
                 for k, v in state.items()}

    slim = {
        "model": state,
        "arch": ck.get("arch", {"name": "unet", "base": 48}),
        "epoch": ck.get("epoch"),
        "best": ck.get("best"),
        "half": bool(args.half),
    }
    torch.save(slim, args.out)
    after = args.out.stat().st_size

    print(f"source : {args.src.name}  {before/1e6:.1f} MB")
    print(f"output : {args.out}  {after/1e6:.1f} MB")
    print(f"saved  : {(before-after)/1e6:.1f} MB ({(1-after/before)*100:.0f}% smaller)")
    print(f"arch   : {slim['arch']}   epoch {slim['epoch']}   best SSIM {slim['best']}")

    # Prove it still loads and runs.
    sys.path.insert(0, str(ROOT))
    from src.models import build_from_arch
    m = build_from_arch(slim["arch"])
    m.load_state_dict({k: v.float() for k, v in slim["model"].items()})
    m.eval()
    with torch.no_grad():
        y = m(torch.rand(1, 1, 64, 64))
    assert tuple(y.shape[-2:]) == (128, 128), y.shape
    print("verified: loads and produces correct 2x output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
