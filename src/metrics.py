"""Restoration quality metrics: pSNR, SSIM, LPIPS.

Convention used everywhere in this repo:
    images are float32 numpy arrays, shape (H, W) greyscale or (H, W, 3),
    with ground truth nominally in [0, 1].

`data_range` is pinned to 1.0 rather than derived from the arrays. Deriving it
(skimage's default when dtype is float) would make the score depend on the
prediction's own range, so a model that overshoots would silently be graded on a
wider scale. The organisers grade against a [0,1] ground truth; we do the same.

Predictions are clipped to [0,1] before scoring. Degraded *inputs* legitimately
exceed that range (speckle is multiplicative — see NOTES.md §0.3), but a
restored *output* is compared against a [0,1] target, and anything outside is
error either way. Clipping keeps LPIPS in the range its backbone expects.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DATA_RANGE = 1.0


def _prep(x: np.ndarray) -> np.ndarray:
    """float32, clipped to [0,1], channel dim squeezed if singleton."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[2] == 1:
        x = x[:, :, 0]
    return np.clip(x, 0.0, 1.0)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _prep(pred), _prep(gt)
    if np.array_equal(pred, gt):
        return float("inf")
    return float(peak_signal_noise_ratio(gt, pred, data_range=DATA_RANGE))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _prep(pred), _prep(gt)
    kwargs = {"data_range": DATA_RANGE}
    if pred.ndim == 3:
        kwargs["channel_axis"] = 2
    return float(structural_similarity(gt, pred, **kwargs))


class LPIPS:
    """Lazy, batched LPIPS. Constructing this imports torch, so it is kept out of
    module scope — `evaluate.py` is timed from process start and must not pay for
    an import it does not use.

    The backbone is fixed to AlexNet per CLAUDE.md Step 3.
    """

    def __init__(self, net: str = "alex", device: str | None = None):
        import torch
        import lpips as _lpips

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _lpips.LPIPS(net=net).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _to_tensor(self, x: np.ndarray):
        """(H,W) or (H,W,3) in [0,1]  ->  (1,3,H,W) in [-1,1]."""
        x = _prep(x)
        if x.ndim == 2:
            x = np.repeat(x[:, :, None], 3, axis=2)
        t = self._torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return (t * 2.0 - 1.0).to(self.device)

    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        with self._torch.no_grad():
            d = self.model(self._to_tensor(pred), self._to_tensor(gt))
        return float(d.item())

    def batch(self, preds: list[np.ndarray], gts: list[np.ndarray]) -> list[float]:
        """Score a list of pairs. Falls back to one-at-a-time when shapes differ,
        since the test set mixes 256x256 and 512x512 outputs."""
        torch = self._torch
        shapes = {p.shape[:2] for p in preds} | {g.shape[:2] for g in gts}
        if len(shapes) != 1:
            return [self(p, g) for p, g in zip(preds, gts)]
        with torch.no_grad():
            a = torch.cat([self._to_tensor(p) for p in preds])
            b = torch.cat([self._to_tensor(g) for g in gts])
            d = self.model(a, b)
        return d.flatten().cpu().tolist()


def score_all(pred: np.ndarray, gt: np.ndarray, lpips_fn: LPIPS | None = None) -> dict:
    out = {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt)}
    if lpips_fn is not None:
        out["lpips"] = lpips_fn(pred, gt)
    return out


def aggregate(rows: list[dict]) -> dict:
    """Mean over per-image dicts, ignoring non-finite pSNR (identical pairs)."""
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


if __name__ == "__main__":
    # Self-test: metrics must move in the right direction as degradation increases.
    rng = np.random.default_rng(0)
    gt = rng.random((128, 128)).astype(np.float32)
    gt = np.clip(gt * 0.5 + 0.25, 0, 1)  # mid-range so noise is not clipped away

    print(f"{'sigma':>8}  {'pSNR':>8}  {'SSIM':>7}")
    prev_psnr, prev_ssim = float("inf"), 1.0
    for sigma in [0.0, 0.01, 0.05, 0.1, 0.2]:
        pred = gt + rng.normal(0, sigma, gt.shape).astype(np.float32) if sigma else gt.copy()
        p, s = psnr(pred, gt), ssim(pred, gt)
        print(f"{sigma:>8.2f}  {p:>8.2f}  {s:>7.4f}")
        assert p <= prev_psnr + 1e-6, "pSNR must not improve as noise grows"
        assert s <= prev_ssim + 1e-6, "SSIM must not improve as noise grows"
        prev_psnr, prev_ssim = p, s

    assert psnr(gt, gt) == float("inf") and abs(ssim(gt, gt) - 1.0) < 1e-6
    print("\npSNR/SSIM self-test passed.")

    try:
        lp = LPIPS()
    except Exception as e:  # noqa: BLE001
        print(f"LPIPS unavailable ({type(e).__name__}: {e}) — skipped.")
    else:
        print(f"LPIPS on {lp.device}")
        d0 = lp(gt, gt)
        d1 = lp(np.clip(gt + rng.normal(0, 0.1, gt.shape), 0, 1).astype(np.float32), gt)
        print(f"  identical: {d0:.6f}\n  sigma=0.10: {d1:.6f}")
        assert d0 < 1e-4, "LPIPS of an image against itself must be ~0"
        assert d1 > d0, "LPIPS must increase with degradation"
        print("LPIPS self-test passed.")
