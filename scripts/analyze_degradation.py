"""Step 2 — reverse-engineer the degradation.

Tests the hypothesis recorded in NOTES.md §0.3:

    clean_lr = downsample(GT, 2, kernel=?)
    degraded = clean_lr * g + n,   g ~ Gamma(L, 1/L),  n ~ N(0, sigma^2)

Four questions, in dependency order:

  A. Which downsampling kernel?      -> everything below needs clean_lr
  B. Is the noise multiplicative?    -> residual spread must scale with intensity
  C. What are L and sigma?           -> conditional-variance regression
  D. Is there blur on top?           -> gradient energy vs. a clean downsample

The key trick is C. Under the model above, conditioning on the clean value c:

    Var[degraded | c] = c^2 / L + sigma^2

so a linear fit of per-bin residual variance against c^2 yields slope = 1/L and
intercept = sigma^2 in one shot. It also *tests* the model: if the relationship is
not linear in c^2, the multiplicative-Gamma-plus-additive-Gaussian story is wrong.

    python scripts/analyze_degradation.py [data_dir] [--n 40]
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from inspect_data import load, side_of, strip_tokens  # noqa: E402

KERNELS = {
    "area": cv2.INTER_AREA,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "nearest": cv2.INTER_NEAREST,
}


def kernel_equivalence_classes(hw: tuple[int, int] = (512, 512),
                               scale: int = 2, tol: float = 1e-5) -> list[list[str]]:
    """Group kernels that are numerically identical at this exact scale factor.

    At an integer 2x downscale, INTER_AREA and INTER_LINEAR reduce to the same
    operation — the mean of each disjoint 2x2 block — agreeing to float32 epsilon
    (verified: max abs diff 6e-8, and both match a manual block mean exactly). At a
    non-integer factor they differ by ~0.44, so this is specific to the 2x case,
    which is the only case this problem has.

    Consequence: "identify the downsampling kernel" has fewer distinguishable answers
    than it appears. Reporting a single winner would be false precision, and a test
    demanding one specific name fails on a coin flip between equivalent options.
    """
    probe = np.random.default_rng(0).random(hw).astype(np.float32)
    dst = (hw[1] // scale, hw[0] // scale)
    outs = {k: cv2.resize(probe, dst, interpolation=v) for k, v in KERNELS.items()}
    classes: list[list[str]] = []
    for k in KERNELS:
        for cl in classes:
            if float(np.abs(outs[cl[0]] - outs[k]).max()) < tol:
                cl.append(k)
                break
        else:
            classes.append([k])
    return classes


def equivalent_kernels(name: str, hw: tuple[int, int] = (512, 512),
                       scale: int = 2) -> list[str]:
    for cl in kernel_equivalence_classes(hw, scale):
        if name in cl:
            return cl
    return [name]


def to_float(a: np.ndarray) -> np.ndarray:
    """Scale integer dtypes to [0,1]; leave floats alone (GT is already [0,1])."""
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    return a.astype(np.float32)


def collect_pairs(root: pathlib.Path, limit: int) -> list[tuple[np.ndarray, np.ndarray]]:
    exts = {".png", ".tif", ".tiff", ".npy", ".jpg", ".jpeg", ".bmp"}
    groups: dict[str, dict[str, pathlib.Path]] = {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        key, tok = strip_tokens(p.stem)
        side = side_of(tok, p)
        if side:
            groups.setdefault(key, {})[side] = p

    pairs = []
    for key, v in groups.items():
        if len(v) != 2:
            continue
        d, c = load(v["degraded"]), load(v["clean"])
        if d is None or c is None:
            continue
        d, c = to_float(d), to_float(c)
        if d.ndim == 3:
            d = d[..., 0]
        if c.ndim == 3:
            c = c[..., 0]
        # Trust resolution over the filename token: whichever is smaller IS the
        # degraded one. A reversed naming convention would otherwise skip every
        # pair and report "no pairs found", which reads like missing data.
        if d.shape[0] > c.shape[0]:
            d, c = c, d
        if c.shape[0] != 2 * d.shape[0]:
            continue  # not a 2x pair; skip rather than guess
        pairs.append((d, c))
        if len(pairs) >= limit:
            break
    return pairs


# ---------------------------------------------------------------- A. kernel
def identify_kernel(pairs) -> str:
    """Compare candidate downsamples against the degraded image by RAW MSE.

    Deliberately does NOT pre-blur to suppress speckle. Blurring is the intuitive move
    but it is counterproductive: the kernels differ almost entirely in how they treat
    frequencies near Nyquist, and a Gaussian blur removes precisely that. Measured on
    synthetic data with a known kernel, blurring collapsed the area-vs-bilinear margin
    to 0.0% — the test could not identify its own ground truth.

    Raw MSE is the right statistic because the noise is zero-mean and independent of
    the candidate:

        MSE(cand, degraded) = MSE(cand, clean_lr) + Var(noise)

    Var(noise) is identical across candidates, so it shifts every score by the same
    constant and cancels from the ranking. The absolute MSE is dominated by noise, but
    the *differences* are exactly the differences in fit to the true clean image.

    Per-pair win counts are reported alongside the mean: a 1% margin that holds on
    24/24 pairs is conclusive, while a 5% margin from one outlier is not.
    """
    print("=== A. DOWNSAMPLING KERNEL ===")
    errs = {k: [] for k in KERNELS}
    for d, c in pairs:
        h, w = d.shape
        for name, interp in KERNELS.items():
            cand = cv2.resize(c, (w, h), interpolation=interp)
            errs[name].append(float(np.mean((cand - d) ** 2)))

    hw = pairs[0][1].shape[:2]
    classes = kernel_equivalence_classes(hw)
    wins = collections.Counter()
    for i in range(len(pairs)):
        wins[min(KERNELS, key=lambda k: errs[k][i])] += 1

    ranked = sorted(((float(np.mean(v)), k) for k, v in errs.items()))
    best_mse = ranked[0][0]
    print(f"  {'kernel':>9}  {'MSE':>12}  {'excess vs best':>15}  {'wins':>6}")
    for mse, k in ranked:
        print(f"  {k:>9}  {mse:.10f}  {mse - best_mse:>15.3e}  "
              f"{wins[k]:>3}/{len(pairs)}")

    if any(len(cl) > 1 for cl in classes):
        print("  equivalence classes at this exact scale factor "
              "(indistinguishable by construction):")
        for cl in classes:
            if len(cl) > 1:
                print(f"    {{{', '.join(cl)}}}")

    best = ranked[0][1]
    cls = next(cl for cl in classes if best in cl)
    class_wins = sum(wins[k] for k in cls)
    # Compare against the best kernel OUTSIDE the winner's equivalence class; comparing
    # against a numerically identical kernel would always report a ~0% margin.
    outside = [(m, k) for m, k in ranked if k not in cls]
    print(f"  -> best: {{{', '.join(cls)}}}  wins {class_wins}/{len(pairs)} pairs")
    if outside:
        margin = (outside[0][0] - ranked[0][0]) / max(ranked[0][0], 1e-12)
        print(f"     {margin * 100:.2f}% below nearest distinct kernel "
              f"({outside[0][1]})")
    if class_wins < 0.9 * len(pairs):
        print("  ! Wins on under 90% of pairs — NOT conclusive. The true kernel may be")
        print("    one not in the candidate list, or applied with antialiasing settings")
        print("    that cv2 does not reproduce (e.g. PIL/torch resize).")
    return best


# --------------------------------------------- B/C. noise model and parameters
def analyse_noise(pairs, kernel: str, nbins: int = 24):
    print(f"\n=== B/C. NOISE MODEL (clean_lr via {kernel}) ===")
    interp = KERNELS[kernel]

    cs, rs = [], []
    for d, c in pairs:
        h, w = d.shape
        clr = cv2.resize(c, (w, h), interpolation=interp)
        cs.append(clr.ravel())
        rs.append((d - clr).ravel())
    c_all = np.concatenate(cs)
    r_all = np.concatenate(rs)

    # Equal-count bins: intensity histograms here are extremely skewed toward black,
    # so equal-width bins would put almost every pixel in bin 0.
    order = np.argsort(c_all)
    c_all, r_all = c_all[order], r_all[order]
    edges = np.linspace(0, len(c_all), nbins + 1).astype(int)

    cc, vv, mm = [], [], []
    print(f"  {'c_mean':>8}  {'residual_std':>13}  {'residual_mean':>14}  {'n':>9}")
    for i in range(nbins):
        s, e = edges[i], edges[i + 1]
        if e - s < 500:
            continue
        cb, rb = c_all[s:e], r_all[s:e]
        cc.append(float(cb.mean()))
        vv.append(float(rb.var()))
        mm.append(float(rb.mean()))
        print(f"  {cc[-1]:>8.5f}  {np.sqrt(vv[-1]):>13.6f}  {mm[-1]:>14.6f}  {e - s:>9}")

    cc, vv = np.array(cc), np.array(vv)

    # B: multiplicative noise means the std grows with c. Report the correlation.
    if len(cc) > 3:
        corr = float(np.corrcoef(cc, np.sqrt(vv))[0, 1])
        print(f"\n  corr(c, residual_std) = {corr:+.4f}")
        print("  -> multiplicative" if corr > 0.8 else
              "  -> NOT clearly multiplicative — model in NOTES.md §0.3 is suspect")

    # C: Var = c^2/L + sigma^2. Least squares on (c^2, Var).
    x = cc ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, vv, rcond=None)
    pred = A @ np.array([slope, intercept])
    ss_res = float(np.sum((vv - pred) ** 2))
    ss_tot = float(np.sum((vv - vv.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\n  fit Var = c^2/L + sigma^2")
    print(f"    slope     = {slope:.6g}   -> L     = {1 / slope:.3f}" if slope > 0
          else f"    slope     = {slope:.6g}   -> NEGATIVE, model wrong")
    print(f"    intercept = {intercept:.6g}   -> sigma = "
          f"{np.sqrt(max(intercept, 0)):.6f}")
    print(f"    R^2       = {r2:.5f}")
    if r2 < 0.9:
        print("  ! Poor fit. Speckle may be applied before downsampling, or the")
        print("    kernel from step A is wrong. Both break the synthetic generator.")

    L_hat = 1 / slope if slope > 0 else float("nan")

    # Prefer the dark-pixel sigma: the intercept above is the difference between two
    # much larger quantities and is biased low (~35% on synthetic data with known sigma).
    interp = KERNELS[kernel]
    dark = []
    for d, c in pairs:
        h, w = d.shape
        clr = cv2.resize(c, (w, h), interpolation=interp)
        s = sigma_from_dark(d, clr, L_hat)
        if np.isfinite(s):
            dark.append(s)
    sigma_hat = float(np.sqrt(np.mean(np.array(dark) ** 2))) if dark else float("nan")
    print(f"\n  sigma (dark-pixel estimator, preferred) = {sigma_hat:.6f}")
    print(f"  sigma (regression intercept, biased low) = {np.sqrt(max(intercept, 0)):.6f}")

    print("\n  Deck reference (NOTES.md §0.2): L=16.86 sigma=0.008594 | L=18.13 sigma=0.001065")
    return L_hat, sigma_hat


def check_residual_whiteness(pairs, kernel: str):
    """Was speckle applied at LR, or at HR before downsampling? (open question #3)

    If applied independently per LR pixel, the residual is spatially white. If applied
    at HR and then downsampled with a kernel of overlapping support (bilinear, bicubic,
    lanczos), neighbouring LR pixels share HR samples and the residual gains a positive
    lag-1 correlation.

    The residual must be normalised by the clean value first. Raw `d - clean_lr` has
    magnitude proportional to the signal, so it inherits the image's own strong spatial
    correlation and every sample would look correlated regardless of the noise model.
    Dividing by clean_lr leaves (g-1) + n/c, which is white iff g is white.

    LIMITATION: `area` downsampling averages *disjoint* 2x2 blocks, so HR-applied speckle
    stays white at LR under that kernel. This test cannot rule that case out; it would
    show up instead as an inflated L (variance divided by ~4). Cross-check the fitted L
    against the deck's stated values (NOTES.md §0.2) on the matching samples.
    """
    print("\n=== B2. RESIDUAL WHITENESS (speckle at LR or at HR?) ===")
    interp = KERNELS[kernel]
    hs, vs = [], []
    for d, c in pairs:
        h, w = d.shape
        clr = cv2.resize(c, (w, h), interpolation=interp)
        rn = (d - clr) / np.maximum(clr, 1e-3)
        # Bright pixels only: where clean_lr ~ 0 the division explodes.
        m = clr > np.percentile(clr, 60)
        for arr, axis_pair in ((hs, (rn[:, :-1], rn[:, 1:], m[:, :-1] & m[:, 1:])),
                               (vs, (rn[:-1, :], rn[1:, :], m[:-1, :] & m[1:, :]))):
            a, b, mm = axis_pair
            if mm.sum() < 500:
                continue
            x, y = a[mm], b[mm]
            if x.std() > 1e-9 and y.std() > 1e-9:
                arr.append(float(np.corrcoef(x, y)[0, 1]))

    if not hs and not vs:
        print("  insufficient bright pixels to measure")
        return
    mh = float(np.median(hs)) if hs else float("nan")
    mv = float(np.median(vs)) if vs else float("nan")
    print(f"  lag-1 autocorrelation of normalised residual: "
          f"horizontal={mh:+.4f}  vertical={mv:+.4f}")
    worst = max(abs(mh), abs(mv))
    if worst < 0.05:
        print("  -> white: speckle applied at LR (after downsampling). Matches NOTES.md §0.3.")
        print("     (Cannot exclude HR speckle + area downsampling — see docstring.)")
    else:
        print("  -> correlated: speckle likely applied at HR BEFORE downsampling.")
        print("     Change the generator to blur/noise at HR then downsample.")


def sigma_from_dark(d, clr, L: float, pct: float = 10.0) -> float:
    """Estimate sigma from the darkest pixels of one pair.

    Where c ~ 0 the speckle term c^2/L nearly vanishes and the additive term dominates,
    inverting the conditioning problem that makes the regression intercept unreliable.
    The small remaining speckle contribution is subtracted using the fitted L.
    """
    thr = np.percentile(clr, pct)
    m = clr <= thr
    if m.sum() < 500:
        return float("nan")
    r = d[m] - clr[m]
    speckle = float(np.mean(clr[m].astype(np.float64) ** 2)) / L if L > 0 else 0.0
    return float(np.sqrt(max(float(r.var()) - speckle, 0.0)))


def per_pair_params(pairs, kernel: str):
    """Per-sample L and sigma, to recover the ranges the generator sampled from.

    L comes from the per-pair Var-vs-c^2 regression; sigma comes from the dark-pixel
    estimator instead of that regression's intercept, which is too noisy per pair to
    be usable (it lands on exactly 0 for roughly half of samples).
    """
    print("\n=== C2. PER-SAMPLE PARAMETER SPREAD ===")
    interp = KERNELS[kernel]
    Ls, sigmas, sigmas_intercept = [], [], []
    for d, c in pairs:
        h, w = d.shape
        clr = cv2.resize(c, (w, h), interpolation=interp)
        r = (d - clr).ravel()
        cv_ = clr.ravel()
        order = np.argsort(cv_)
        cv_s, r_s = cv_[order], r[order]
        edges = np.linspace(0, len(cv_s), 17).astype(int)
        cc, vv = [], []
        for i in range(16):
            s, e = edges[i], edges[i + 1]
            if e - s < 200:
                continue
            cc.append(cv_s[s:e].mean() ** 2)
            vv.append(r_s[s:e].var())
        if len(cc) < 4:
            continue
        A = np.vstack([np.array(cc), np.ones(len(cc))]).T
        (sl, ic), *_ = np.linalg.lstsq(A, np.array(vv), rcond=None)
        if sl > 0:
            L_i = 1.0 / sl
            Ls.append(L_i)
            sigmas_intercept.append(np.sqrt(max(ic, 0)))
            s_i = sigma_from_dark(d, clr, L_i)
            if np.isfinite(s_i):
                sigmas.append(s_i)

    if not Ls:
        print("  no usable fits")
        return
    for name, arr in (("L", np.array(Ls)),
                      ("sigma", np.array(sigmas)),
                      ("sigma_int", np.array(sigmas_intercept))):
        if arr.size == 0:
            continue
        print(f"  {name:>9}: min={arr.min():.5g}  p25={np.percentile(arr, 25):.5g}  "
              f"median={np.median(arr):.5g}  p75={np.percentile(arr, 75):.5g}  "
              f"max={arr.max():.5g}")
    print("  (sigma = dark-pixel estimator, preferred; sigma_int = regression intercept,")
    print("   shown only as a cross-check — it collapses to 0 on many samples.)")
    print("\n  Use the L and sigma rows as sampling ranges for synthetic degradation (Step 4).")


# ---------------------------------------------------------------- D. blur
def check_blur(pairs, kernel: str):
    """If slide 9's third degradation really were blur, the provided degraded image
    would have systematically less gradient energy than a clean downsample of GT.
    Noise pushes gradient energy *up*, so denoise both sides before comparing."""
    print("\n=== D. BLUR CHECK ===")
    interp = KERNELS[kernel]
    ratios = []
    for d, c in pairs:
        h, w = d.shape
        clr = cv2.resize(c, (w, h), interpolation=interp)
        a = cv2.GaussianBlur(d, (0, 0), 1.5)
        b = cv2.GaussianBlur(clr, (0, 0), 1.5)
        ga = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0, 3), cv2.Sobel(a, cv2.CV_32F, 0, 1, 3))
        gb = np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0, 3), cv2.Sobel(b, cv2.CV_32F, 0, 1, 3))
        ratios.append(float(ga.mean() / max(gb.mean(), 1e-9)))

    r = np.array(ratios)
    print(f"  mean gradient ratio (degraded / clean_downsample):")
    print(f"    median={np.median(r):.4f}  p05={np.percentile(r, 5):.4f}  "
          f"p95={np.percentile(r, 95):.4f}")
    if np.median(r) < 0.9:
        print("  -> degraded is SMOOTHER: blur is present. Add it to the generator.")
    elif np.median(r) > 1.1:
        print("  -> degraded is SHARPER: residual noise energy, no blur. Expected under")
        print("     the NOTES.md §0.3 model (additive+multiplicative noise, no blur).")
    else:
        print("  -> comparable: no strong evidence of blur.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data/raw", type=pathlib.Path)
    ap.add_argument("--n", type=int, default=40, help="pairs to analyse")
    args = ap.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: {args.data_dir.resolve()} does not exist.")
        return 1

    pairs = collect_pairs(args.data_dir, args.n)
    print(f"loaded {len(pairs)} 2x pairs from {args.data_dir.resolve()}\n")
    if not pairs:
        print("No pairs found — run scripts/inspect_data.py first and fix the pairing.")
        return 1

    kernel = identify_kernel(pairs)
    analyse_noise(pairs, kernel)
    check_residual_whiteness(pairs, kernel)
    per_pair_params(pairs, kernel)
    check_blur(pairs, kernel)
    print("\nRecord all of the above in NOTES.md Step 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
