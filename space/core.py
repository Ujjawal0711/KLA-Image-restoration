"""All demo logic, deliberately free of any Gradio import.

Separated so the substantive parts — model loading, degradation, restoration, metrics —
can be tested without the UI framework installed, and so a UI change cannot break them.
`app.py` is then only wiring.
"""

from __future__ import annotations

import pathlib
import time

import cv2
import matplotlib
matplotlib.use("Agg")
# Figure() rather than pyplot.subplots(): pyplot keeps a global registry of every
# figure it creates, so a Space that redraws the histogram on each slider move would
# leak them indefinitely ("More than 20 figures have been opened"). A bare Figure is
# owned by the caller and garbage-collected normally.
from matplotlib.figure import Figure
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.degradation import degrade
from src.models import build_from_arch

HERE = pathlib.Path(__file__).resolve().parent
torch.set_num_threads(2)          # free Spaces give 2 vCPU

_ck = torch.load(HERE / "model.pt", map_location="cpu", weights_only=False)
MODEL = build_from_arch(_ck["arch"])
MODEL.load_state_dict({k: v.float() for k, v in _ck["model"].items()})
MODEL.eval()
N_PARAMS = sum(p.numel() for p in MODEL.parameters())
ARCH = _ck["arch"]

SAMPLES: dict[str, str] = {}
_idx = HERE / "samples" / "index.txt"
if _idx.exists():
    for line in _idx.read_text(encoding="utf-8").splitlines():
        if "|" in line:
            fn, label = line.split("|", 1)
            SAMPLES[label.strip()] = fn.strip()

# Measured on 300 held-out images. LPIPS is shown from the benchmark rather than
# computed live, because it needs a 230 MB network this Space does not otherwise need.
BENCH = {"psnr": 25.395, "ssim": 0.6994, "lpips": 0.1711,
         "b_psnr": 21.517, "b_ssim": 0.5110, "b_lpips": 0.4341}


def load_gt(sample_label: str | None, uploaded: np.ndarray | None) -> np.ndarray:
    if uploaded is not None:
        img = uploaded
        if img.ndim == 3:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
        img = img.astype(np.float32)
        img = img / 255.0 if img.max() > 1.5 else img
        if min(img.shape[:2]) > 512:      # keep CPU inference responsive
            y, x = (img.shape[0] - 512) // 2, (img.shape[1] - 512) // 2
            img = img[y:y + 512, x:x + 512]
        # Even dimensions only: the task is an exact 2x, so an odd size has no
        # well-defined half. Trim rather than pad, so nothing invented is scored.
        h, w = img.shape[:2]
        img = img[:h - h % 2, :w - w % 2]
        return np.clip(img, 0, 1)

    fn = SAMPLES.get(sample_label or "") or next(iter(SAMPLES.values()))
    a = cv2.imread(str(HERE / "samples" / fn), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(f"sample {fn} missing")
    return a.astype(np.float32) / 255.0


def to_disp(x: np.ndarray) -> np.ndarray:
    """Clip for display only. Degraded values legitimately exceed 1.0 — that is what
    the histogram exists to show — but a screen cannot render past white."""
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def hist_figure(gt: np.ndarray, lr: np.ndarray):
    fig = Figure(figsize=(6.4, 3.0), dpi=110)
    ax = fig.add_subplot(111)
    hi = max(1.6, float(lr.max()) * 1.05)
    ax.hist(gt.ravel(), bins=80, range=(0, hi), alpha=.65,
            label="ground truth", color="#2E74B5", density=True)
    ax.hist(lr.ravel(), bins=80, range=(0, hi), alpha=.65,
            label="degraded", color="#E8A33D", density=True)
    ax.axvline(1.0, color="#9C2D2D", ls="--", lw=1.6)
    over = float((lr > 1.0).mean()) * 100
    ax.set_xlabel("pixel value")
    ax.set_ylabel("density")
    ax.set_title(f"{over:.2f}% of degraded pixels exceed 1.0  "
                 f"(max {lr.max():.3f})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_degraded(sample_label, uploaded, L, sigma, kernel, seed):
    """-> (gt_display, lr_display, histogram, info_markdown, (gt, lr))"""
    gt = load_gt(sample_label, uploaded)
    rng = np.random.default_rng(int(seed))
    lr = degrade(gt, {"L": float(L), "sigma": float(sigma),
                      "kernel": kernel, "scale": 2}, rng)
    info = (f"**Model input** {lr.shape[1]}×{lr.shape[0]}  →  "
            f"**output** {gt.shape[1]}×{gt.shape[0]}\n\n"
            f"Degraded range **[{lr.min():.3f}, {lr.max():.3f}]** against a ground "
            f"truth of [0, 1]. Speckle multiplies, so it scales with brightness and "
            f"pushes bright pixels past white.")
    return to_disp(gt), to_disp(lr), hist_figure(gt, lr), info, (gt, lr)


def restore(state):
    """-> (bicubic_display, model_display, metrics_markdown)"""
    if state is None:
        return None, None, "Move a slider first."
    gt, lr = state
    x = torch.from_numpy(np.ascontiguousarray(lr[None, None], dtype=np.float32))
    t0 = time.perf_counter()
    with torch.inference_mode():
        pred = MODEL(x).clamp_(0, 1).numpy()[0, 0]
    dt = time.perf_counter() - t0

    bic = np.clip(cv2.resize(lr, (gt.shape[1], gt.shape[0]),
                             interpolation=cv2.INTER_CUBIC), 0, 1).astype(np.float32)

    def sc(p):
        return (peak_signal_noise_ratio(gt, p, data_range=1.0),
                structural_similarity(gt, p, data_range=1.0))

    bp, bs = sc(bic)
    mp, ms = sc(pred.astype(np.float32))
    md = (
        f"| | bicubic | **model** | gain |\n|---|---|---|---|\n"
        f"| pSNR | {bp:.2f} | **{mp:.2f}** | **{mp - bp:+.2f} dB** |\n"
        f"| SSIM | {bs:.4f} | **{ms:.4f}** | **{ms - bs:+.4f}** |\n\n"
        f"Restored in **{dt:.2f}s** on 2 CPU threads.\n\n"
        f"*LPIPS is not computed live (it needs a 230 MB network). Measured on 300 "
        f"held-out images: bicubic {BENCH['b_lpips']:.4f} → model "
        f"**{BENCH['lpips']:.4f}**.*"
    )
    return to_disp(bic), to_disp(pred), md
