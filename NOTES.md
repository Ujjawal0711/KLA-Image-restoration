# NOTES.md — running log of empirical findings

Source of truth for everything empirical. Steps refer to the build order in `CLAUDE.md`.

---

## Status

> **2026-08-04 — PIVOT: no training data will be released.** Submission is direct.
> Steps 1 and 2 can never run. Everything trains on synthetic pairs built from
> externally sourced clean images, which the deck explicitly permits ("synthetic data
> generation is explicitly encouraged", slide 20). The degradation model in §0.3 is now
> a **load-bearing assumption that cannot be verified** — see §2 for how that changes
> the strategy. Deadline 16 Aug 2026.

| Step | State |
|---|---|
| 0. Problem statement extraction | **done** (this doc) |
| 0.5 Toolchain verification | **done** — §0.7, all self-tests pass |
| 1. Data inspection | **N/A** — no data released. Scripts kept in case that changes. |
| 2. Reverse-engineer degradation | **N/A** — same. Estimators validated but nothing to run them on. |
| 3. Metrics + baselines | **done** — §3 |
| 4. Dataset + synthetic degradation | **done** — `src/dataset.py` |
| 5. U-Net baseline | **done and verified** — §4 |
| 6. Loss design | implemented (`src/losses.py`); weights not yet tuned |
| 7. NAFNet | **implemented and verified** — §4 |
| 8. Inference optimisation | **done** — §5, 47% faster than first working version |
| 9. Package submission | `evaluate.py` verified end-to-end; needs final weights + requirements.txt |

---

## Step 0 — What the KLA deck actually says

Extracted from `Problem Statement 01_KLA.pptx` (22 slides, July 2026) by unzipping the OOXML
and reading `ppt/slides/*.xml` + `ppt/media/*`.

### 0.1 Degradation types — the deck contradicts itself

| Slide | Degradations listed |
|---|---|
| 3 (Challenge at a Glance) | speckle noise, spatial resolution reduction |
| 4 (Why This Matters) | speckle noise, downsampling |
| 6 (Dataset Quick Look) | downsampling only |
| 10 (Training Set) | downsampling, speckle |
| **9 (Problem Statement)** | speckle, **"Gausian noise"**, downsampling |

Only **slide 9** mentions a third degradation, spelled "Gausian", described as
*"reduction in image sharpness and edge detail"*. Four other slides list only two
degradations. `CLAUDE.md` flagged this description as reading like blur rather than
additive noise, and marked it unresolved.

### 0.2 The figure titles resolve it — additive noise, not blur

The two sample figures embedded in the deck (`ppt/media/image10.png` → slides 5 & 11;
`image11.png` → slide 13) are matplotlib renders whose **titles carry the generator's
per-sample degradation parameters**:

```
Training - Sample 000000    Source: 0001.png | L=16.86, σ=0.008594     (texture)
Training - Sample 000500    Source: 0186.png | L=18.13, σ=0.001065     (dendrite)
```

Read at 4× upscale to confirm each digit. Interpretation:

- **`L` = number of looks**, the shape parameter of a Gamma-distributed multiplicative
  speckle model (standard in SAR / coherent imaging).
- **`σ` = standard deviation of additive Gaussian noise**, in normalised `[0,1]` intensity units.

**σ cannot be a Gaussian blur kernel width.** σ = 0.0011 and σ = 0.0086 px are deeply
sub-pixel — such a kernel is an exact no-op on a sampled grid. As an *additive noise* std on a
`[0,1]` scale these values are small but meaningful. So the slide-9 wording is a description
error in the deck; the third degradation is **additive Gaussian noise**.

> ⚠️ **Confidence: high, but not confirmed.** This is inferred from two figure titles, not
> measured on data. Step 2 must verify against real pairs before the synthetic degradation
> pipeline is built on it. Everything in §0.3 is a hypothesis.

### 0.3 Working degradation model (hypothesis)

```
clean_lr = downsample(GT, factor=2, kernel=???)      # kernel unidentified — Step 2
speckled = clean_lr * g,   g ~ Gamma(shape=L, scale=1/L)      # E[g]=1, Var[g]=1/L
degraded = speckled + n,   n ~ Normal(0, σ²)
# no clipping — deck repeatedly states values may exceed the GT range
```

Consistency check against the deck's own histograms:

- L ≈ 17 → speckle std = `1/√L` ≈ **0.24 relative**. Dominant term.
- σ ≈ 0.001–0.009 → **~25–200× smaller** than the speckle contribution. Secondary.
- Sample 000000: GT range `[0,1]`, NoisyLR histogram extends to ≈ 1.2. A pixel at 0.8 with
  +2σ_speckle reaches ≈ 1.19. ✓ consistent.
- Sample 000500 (dark dendrite, bright thin structures near 1.0): NoisyLR colourbar reaches
  1.6. A pixel at 1.0 with +2.5σ_speckle reaches ≈ 1.6. ✓ consistent.

The multiplicative model also explains the deck's repeated "intensity range exceeds GT" note
without needing clipping anywhere in the chain — **do not clip degraded inputs in `evaluate.py`.**

### 0.4 Parameter ranges — insufficient data

Two samples only. `L ∈ {16.86, 18.13}` and `σ ∈ {0.001065, 0.008594}`.

- `L` is non-integer → drawn from a continuous distribution, not an integer look count.
- The two σ values differ by **8×** while the two L values differ by 7% — hints σ is sampled
  log-uniformly over a wide range and L over a narrow one. **Two points cannot establish this.**
  Step 2 must estimate both ranges from the real pairs.

### 0.5 Dataset structure hints

- GT sources are numbered PNGs (`0001.png`, `0186.png`); training samples use 6-digit indices.
- Sample `000000` → source `0001`; sample `000500` → source `0186`. Ratio ≈ 2.7 samples per
  source, suggesting **multiple degradations generated per GT image**. Speculative.
- If true, the train/val split must be **grouped by source image**, not by sample index —
  otherwise the same GT leaks across the split and validation scores are inflated.
- Both deck figures are 512→256. **No 256→128 example appears anywhere in the deck.**
- Both figures are single-channel greyscale rendered through a grey colourmap, GT scaled `[0,1]`.

### 0.6 Requirements confirmed verbatim (slides 15, 17, 21, 22)

- `evaluate.py`: standalone script, non-notebook, args = test image dir + output dir, no manual edits.
- Timed on H100, wall clock includes **script startup + model init + disk read + inference + disk write**.
- Deliverables: eval script, training script/notebook, test outputs, full `pip freeze`.
- No architecture restrictions. External data permitted; synthetic generation explicitly encouraged.

---

## §0.7 — Toolchain verification (2026-08-03)

No real data yet, but the analysis code can still be falsified: generate pairs from the
§0.3 model with **known** parameters and check the estimators recover them
(`scripts/validate_analysis.py`). Results on 24 synthetic 512→256 pairs:

| quantity | true | recovered | error |
|---|---|---|---|
| kernel | `area` | `{area, bilinear}`, 24/24 pairs | correct class |
| `L` | 17.053 | 16.817 | **1.4%** |
| `σ` | 0.006279 | 0.006266 | **0.2%** |

`src/metrics.py` self-test also passes: pSNR/SSIM decrease monotonically with added
noise, LPIPS is 0.000000 on identical inputs and runs on CUDA.

Three things this exposed, all now fixed:

### (a) `area` and `bilinear` are the SAME operation at exactly 2×

Verified directly: `max|INTER_AREA − INTER_LINEAR|` = 6e-8 (float32 epsilon) at 512→256
and at 256→128, and both equal a manual disjoint-2×2-block mean **exactly**. At a
non-integer factor (512→300) they diverge by 0.44.

This problem is *always* exactly 2×, so these two kernels are **indistinguishable in
principle** — no estimator can separate them and the choice does not matter. Step 2
therefore reports an equivalence *class*, not a single winner. Expect
`{area, bilinear}` to be the answer if the organisers used any standard box/linear
downsample.

### (b) Do not pre-blur when identifying the kernel

`CLAUDE.md` Step 2 suggests blurring both images to suppress speckle before comparing.
Measured: that drops the margin between the correct kernel and the wrong ones to
**0.0%** — the test could not identify its own ground truth. Kernels differ almost
entirely near Nyquist, which is exactly what the blur removes.

Raw MSE is the correct statistic instead, because the noise is zero-mean and
independent of the candidate:

```
MSE(cand, degraded) = MSE(cand, clean_lr) + Var(noise)
```

`Var(noise)` is the same constant for every candidate, so it cancels from the ranking.
With no blur the correct class wins 24/24 pairs at a 5.33% margin over the nearest
distinct kernel.

### (c) σ is weakly identifiable from the variance regression

The `Var = c²/L + σ²` intercept is the difference of two much larger quantities: at
mid intensities σ² is ~2% of the total. The intercept estimate came out **35% low**
pooled, and per-sample it collapsed to exactly 0 on half the samples (clamped from
negative).

Fix: estimate σ from the **darkest decile**, where the `c²/L` term nearly vanishes and
σ² dominates, then subtract the small residual speckle contribution using the fitted
`L`. Error dropped 35% → **0.2%**, and the per-sample spread now recovers the true
range (`0.0008–0.0119` vs true `0.00081–0.01175`). `L` is strongly identified either way.

> Caveat that remains: the whiteness test (§B2) cannot distinguish "speckle at LR" from
> "speckle at HR + `area` downsampling", because averaging disjoint 2×2 blocks keeps the
> residual white. That case would instead show up as an inflated `L` (~4×). Cross-check
> the fitted `L` against the deck's stated 16.86 / 18.13 on the matching samples.

### (d) LPIPS pulls a 233 MB AlexNet download on first use

Cached to `~/.cache/torch/hub` afterwards. Harmless in training, **fatal in
`evaluate.py`** — it is timed from process start, so a cold cache on the H100 box would
add minutes. `evaluate.py` must not import `lpips` at all.

---

## §1 — Submission hardening (Phase 1)

Done before chasing further quality, because these are the failure modes that score
**zero** rather than "slightly worse".

### Edge cases (`scripts/edge_cases.py`) — ALL PASS

| case | result |
|---|---|
| single image | ok |
| non-square (120×200) | ok |
| odd sizes (97×131, 63×63, 129×129) | ok, output exactly 2× |
| RGB input instead of greyscale | ok |
| mixed dtypes + extensions in one folder | ok |
| **corrupt file among valid ones** | **skipped, neighbours still written, exit 0** |
| nested sub-directories | ok |
| empty input directory | clean error, exit 2, no traceback |
| image larger than the stated max (512×512) | ok |

The corrupt-file and empty-directory cases matter most: a crash partway through loses
every remaining image, which is far worse than a lower score on one.

### Self-containment (`scripts/check_submission.py`)

`evaluate.py` reaches only `cv2`, `numpy`, `torch` — all in `requirements.txt`, and no
heavy extras (`lpips`, `matplotlib`, `scipy` all absent, confirmed programmatically).

### Isolation test (`scripts/package_submission.py`)

The submission folder is assembled (24 files) and `evaluate.py` is then run **from
inside it**, with the working directory set there and `PYTHONPATH` cleared, so any
accidental dependence on an unshipped file surfaces here rather than on the graders'
machine. **Passed** — 3 inputs, 3 outputs, exit 0.

### Checkpoint slimming (`scripts/promote_checkpoint.py`)

Training checkpoints carry AdamW's two momentum buffers per parameter plus scheduler
and scaler state — about 3× the weights alone. Stripping them:

```
nafnet_best.pt   133.3 MB  ->  model.pt   44.4 MB   (67% smaller)
```

Verified the slim file still loads and produces correct 2× output. This is load time
charged directly to the scored clock.

### Phase 0 is now one command

`scripts/finalize.py` runs the whole post-training sequence: promote → regenerate
requirements → edge cases → self-containment → mock-test round trip → repackage →
print the final numbers. Any failing step aborts with non-zero exit, since a
half-finalised submission is worse than an obviously unfinished one.

---

## §2 — Strategy under an unverifiable degradation model

With measurable parameters you tune to match them. Without, the correct move is to
**train across a deliberately wide range** and accept slightly worse performance at any
single setting in exchange for not falling off a cliff at the true one. A wrong point
estimate is far worse than a range that contains the truth.

Ranges in `src/degradation.py` (`DegradationConfig`):

| parameter | range | reasoning |
|---|---|---|
| `L` | 8 – 40 | deck observed 16.86 and 18.13; widened ~2× in both directions |
| `σ` | 5e-4 – 2e-2, log-uniform | deck observed 0.001065 and 0.008594, an 8× spread → log scale |
| kernel | area, bilinear, bicubic, lanczos | all four sampled, since we cannot identify the real one |

Every training sample draws a fresh `(kernel, L, σ)`, so the model never sees the same
degradation twice and cannot overfit to one noise level.

## §3 — Ground-truth corpus and baselines

No paired data, so every GT image is externally sourced (`scripts/fetch_data.py`,
`scripts/make_procedural.py`). All stored as single-channel 16-bit PNG.

| source | count | contributes |
|---|---|---|
| EPFL electron microscopy (via HF) | 3367 | real EM imagery — closest public analogue to inspection |
| DTD describable textures | ~5600 | 47 texture families; structural diversity for the OOD half |
| procedural | 1500 | dendrite / granular / grain / lattice / fibers — matches deck figs |

Procedural generators exist because neither corpus is semiconductor imagery, and SR has
to learn what plausible high-frequency detail looks like *for this domain*. `lattice`
in particular (periodic die and line-and-space patterns) has no analogue in the
downloaded sets. Visual check: `python scripts/make_procedural.py --contact-sheet`.

Validation split holds out whole images, stratified by corpus prefix.

### Baseline floors (400 val samples, `scripts/baseline_scores.py --n 400`)

| baseline | pSNR | SSIM | LPIPS |
|---|---|---|---|
| bicubic | 19.924 | 0.5124 | 0.4709 |
| **bicubic + median 3×3** | 21.395 | **0.5560** | **0.4189** |
| bicubic + bilateral | 21.055 | 0.5547 | 0.4343 |
| **median 3×3 at LR, then bicubic** | **21.802** | 0.5214 | 0.4352 |

> ⚠️ **Correction (2026-08-04).** An earlier run at **150 samples** reported
> bicubic+median at 20.894/0.5430/0.4782 and median@LR at 21.903/0.5360/0.4483, and
> was written up as "denoise-before-upsample wins by 1.0 dB pSNR and 0.03 LPIPS —
> independent support for the architecture."
>
> **That claim does not survive a larger sample.** At 400 samples the pSNR gap is
> **+0.407 dB**, not +1.0, and denoise-first *loses* on both SSIM (−0.035) and LPIPS
> (+0.016). The 150-sample estimate was simply too noisy for a difference that size.
>
> The architecture choice still stands on the compute argument — processing at LR is
> exactly 4× cheaper, which is arithmetic, not statistics — but it is **not** cleanly
> supported by this experiment. All baseline numbers elsewhere in this file and in the
> report have been updated to the 400-sample values.

**Any trained model below SSIM 0.5560 / pSNR 21.802 is broken.**

## §3.5 — Final results and the shipped model

Two full 60-epoch runs. Because each run validates against **its own** degradation
range, training-log SSIM is not comparable across runs — everything below is
re-measured on 400 identical validation samples under the default config
(`scripts/compare_models.py --n 400`).

| | pSNR | SSIM | LPIPS |
|---|---|---|---|
| best baseline (per metric) | 21.802 | 0.5560 | 0.4189 |
| run 1 — NAFNet w48, 11.1M, L 8–40 | 24.696\* | 0.7102\* | 0.1961\* |
| **run 2 — NAFNet w64, 19.6M, L 5–60 (SHIPPED)** | **24.420** | **0.6925** | **0.1511** |

\* run 1 measured at n=120; run 2 at n=400. Head-to-head at n=120 run 2 led on all
three (+0.121 pSNR, +0.0053 SSIM, −0.0120 LPIPS).

**vs best baseline: +2.618 dB pSNR, +0.1365 SSIM (+24.6%), −0.2678 LPIPS.**

### Why run 2 ships

| | run 1 | run 2 |
|---|---|---|
| worst-case SSIM outside trained range | 0.4163 | **0.5620** |
| worst-case drop vs in-range mean | 40.5% | **20.0%** |
| inference, 60 mixed images | 4.855 s | 4.913 s |

It wins every cell of the robustness grid, roughly halves the worst-case collapse, and
costs no measurable inference time despite 78% more parameters — at these batch sizes
the workload is memory-bandwidth bound, so extra capacity is close to free.

### Robustness sweep (`scripts/sensitivity_sweep.py`)

This is the experiment that directly probes the project's central risk. Run 1, SSIM:

| L \ σ | 0 | 0.0086 | 0.05 |
|---|---|---|---|
| 4 | 0.4862 | 0.4788 | 0.4163 |
| 8 | 0.6511 | 0.6491 | 0.5948 |
| 17 | 0.7042 | 0.7011 | 0.6335 |
| 100 | 0.7853 | 0.7827 | 0.6921 |

Degrades **gracefully upward** (less noise than expected simply scores better) but falls
away sharply below L=8, losing 0.17 SSIM between L=8 and L=4. That finding is what
motivated run 2's wider range.

---

## §3.55 — Run 3 (2026-08-04; shipped until run 5 superseded it — see §3.60)

Four changes stacked on run 2, each targeting something measured rather than guessed:

| change | motivation |
|---|---|
| **EMA of weights** (decay 0.999, warmed up) | standard in restoration; simply omitted before |
| **Denser procedural images** | measured at SSIM 0.9337 vs em's 0.6282 — mostly dark background contributing little gradient. Now denser structure on a textured substrate |
| **+800 real brightfield microscopy** | corpus 10,506 → 11,306; em is the hardest and most target-like corpus |
| **Deeper, not wider** (middle blocks 8→12, 27.0M params) | width bought only +0.005 SSIM in run 2, so depth was the untested axis |

**Crashed at epoch 55/60 with a system-RAM `MemoryError` in a DataLoader worker.**

> **Cause: a game (FIFA 26) was launched on the same machine mid-run**, taking several
> GB of RAM and VRAM. Not a fault in the worker configuration, which had been stable
> for 55 epochs and for all of run 2 before it. Recorded because the first diagnosis
> here blamed the training setup, which was wrong.

Cost nothing: epochs 51–54 all sat at SSIM 0.6967–0.6968, fully plateaued, and
per-epoch checkpointing meant epoch 53's best was already saved.

**Consequence for the timing numbers:** the first v3-vs-run-2 speed comparison was taken
while the game was running and showed 14.9 s for both. Re-measured on an idle machine:
8.34 s (run 2) vs 8.11 s (v3), three runs each. Conclusion unchanged — v3 is free — but
see the caution below.

### Absolute inference timings on this laptop are not trustworthy

The same 60-image workload has measured **4.8 s, 8.3 s and 14.9 s** across sessions,
depending on thermal state, clock boost and background load. Only back-to-back A/B
comparisons mean anything. The scored measurement happens on an H100 and will not
resemble these numbers; what transfers is the pipeline design (deferred imports,
overlapped I/O, batching by resolution, autotuning disabled), not the seconds.

### Head-to-head vs run 2, 300 identical samples

| | pSNR | SSIM | LPIPS |
|---|---|---|---|
| run 2 (19.6M) | 25.283 | 0.6958 | 0.1805 |
| **run 3 (27.0M)** | **25.395** | **0.6994** | **0.1711** |
| delta | +0.112 | +0.0036 | −0.0094 |

Inference time identical (14.9 s vs 14.9 s on the same 60 images). Same pattern as
run 1 → run 2: modest, consistent, free.

### Run-3 numbers (300 samples, corpus as of 2026-08-04)

| | pSNR | SSIM | LPIPS |
|---|---|---|---|
| best baseline per metric | 21.517 | 0.5110 | 0.4341 |
| **run 3** | **25.395** | **0.6994** | **0.1711** |
| **gain** | **+3.878** | **+0.1884 (+36.9%)** | **−0.2630** |

> Superseded on 2026-08-05 by run 5 (34.4M): **25.448 / 0.7012 / 0.1672**. See §3.60.
> `checkpoints/model.pt` and `space/model.pt` both carry run 5.

### Side effects worth recording

- **Brightfield LPIPS more than halved**, 0.2631 → 0.1113. That was run 2's worst
  perceptual score of any corpus; adding 800 brightfield images fixed it directly.
- **The procedural fix worked.** Bicubic's SSIM on `proc` fell 0.8511 → 0.8165,
  confirming those images are genuinely harder than before.

### Two bugs found while doing this

1. **Smoke runs clobbered real checkpoints.** `--smoke` defaulted to `--tag <model>`,
   which overwrote `nafnet_best.pt` — run 1's weights were lost this way. `--smoke` now
   forces a `smoke_` tag. Run 1 was already superseded so nothing of value was lost.
2. **Resume was broken by the EMA change.** `ck["model"]` became the *averaged* weights
   (they are what gets validated and shipped), but the resume path still loaded it into
   the live model — so a resumed run would have silently restarted optimisation from
   the average and discarded the accumulated EMA. Now loads `raw_model` and restores
   `ema_shadow`/`ema_step`.

   Found only because a question about pausing prompted a check of the resume path
   *before* claiming it was safe. The failure mode is silent: every log line would have
   looked healthy while the run quietly produced a worse final model.

   Verified twice — a 40-step smoke test, and then for real when run 5 was paused at
   epoch 52 and resumed: **`restored EMA average (62400 steps accumulated)`**, which is
   exactly 52 × 1200 iterations. The average survived the restart intact.

   Note the ordering hazard this also exposed: the epoch line is printed at train.py:325
   but the checkpoint is not written until :347. Anything that stops training on the log
   line alone loses that epoch. The pause watcher waits for the checkpoint file's mtime
   to update, not for the log.

## §3.57 — Loss weights: swept, and left unchanged (`scripts/tta_eval.py` sibling)

The 1.0 Charbonnier / 0.2 SSIM / 0.1 LPIPS split came from the project plan and was the
**last unvalidated knob in the pipeline** — every other setting had been measured.

Three challengers fine-tuned from v3 for 8 epochs each at LR 5e-5 (a full 3×3 grid from
scratch would be 9 × 5 h, which does not fit the deadline). Re-scored on 300 identical
samples, since each run's own validation is not comparable:

| weights (Charb/SSIM/LPIPS) | pSNR | SSIM | LPIPS |
|---|---|---|---|
| **1.0 / 0.2 / 0.1 — v3, incumbent** | **25.395** | 0.6994 | 0.1711 |
| 1.0 / 0.4 / 0.2 | 25.344 | 0.7000 | 0.1697 |
| 1.0 / 0.6 / 0.1 | 25.380 | **0.7039** | 0.1807 |
| 1.0 / 0.2 / 0.3 | 25.306 | 0.6951 | **0.1672** |

**Decision rule, committed to before the data was seen:** swap only on a win across all
three metrics, or two wins with a third-metric loss inside noise; ties go to the
incumbent, which is already verified and packaged.

**Result: no swap.** The 0.4/0.2 run wins SSIM by 0.0006 and LPIPS by 0.0014 — both far
inside the noise floor (the same model has measured 0.7156 and 0.6925 SSIM on different
samples of the same corpus). The other two are clean trades: weight SSIM up and LPIPS
degrades, weight LPIPS up and SSIM degrades.

**The knob is real but there is no free lunch on it.** You can move along the frontier;
you cannot move off it. The plan's weights were already a sensible point.

## §3.58 — Corpus expansion (materials micrographs)

Added `kvriza8/microscopy_images` — 20.9K SEM/TEM images scraped from publication
figures. Scraped figures means plots, tables and multi-panel composites mixed in with
real micrographs, so `looks_like_micrograph()` filters them.

**The first filter was not good enough, and only visual inspection caught it.** It used
a near-white *fraction* test, which passed an XRD plot, a table of numbers, a photo of
a lab instrument and several multi-panel composites — roughly 6 of 24 sampled images
were unusable. Training on those teaches the model to reconstruct chart axes and text.

Fix: test the largest **contiguous** near-white blob rather than the total fraction.
Plot margins, table backgrounds and the gutters between figure panels are all one big
connected white region; real micrographs essentially never contain one. Added the same
test for uniform dark backgrounds, and tightened the entropy and gradient thresholds.

| | kept | of 20,886 |
|---|---|---|
| first filter | 3,050 | 15% (≈25% of those unusable) |
| **tightened filter** | **2,052** | **10%** (24/24 sampled were genuine micrographs) |

Precision over recall is the right trade for training data. **Corpus: 13,358 images** —
5,639 dtd, 3,367 em, 2,052 mat, 1,500 proc, 800 bright.

## §3.99 — Dress rehearsal (`scripts/dress_rehearsal.py`)

Final check before submission. Differs from every earlier measurement in three ways: it
runs **`submission/evaluate.py`**, not the dev copy, from inside the package with
`PYTHONPATH` cleared; the images come from held-out *and* out-of-distribution sources,
mirroring the deck's statement that the test set mixes both; and both scale regimes sit
in one unsorted directory, as they will on the day.

200 images — 105 held-out, **95 OOD** — L from 8.9 to 39.9, sigma 0.0005 to 0.0198.

| group | n | pSNR | SSIM | LPIPS |
|---|---|---|---|---|
| **ALL** | 200 | **26.316** | **0.6033** | **0.2218** |
| held-out | 105 | 24.390 | 0.6424 | 0.2092 |
| OOD | 95 | 28.445 | 0.5602 | 0.2357 |

**Gain over bicubic on the same files: +5.818 dB, +0.2117 SSIM, −0.2777 LPIPS.**
200 in → 200 out, all uint8, **all exactly 2×**. PASSED.

### Do not misread the per-regime split

The run also reports 256→512 at SSIM 0.5035 against 128→256 at 0.6315. That is **not** a
regime penalty: 512 crops can only be taken from large source images, so the two groups
contain different pictures. The controlled test in §3.65 — same sources, only crop size
varying — gives 0.7320 vs 0.7347. Confounded comparison, not a weakness.

### Import time is the largest single cost, and it is irreducible

| phase | cold cache | warm |
|---|---|---|
| imports | 20.08 s | **4.10 s** |
| inference (200 images) | 7.17 s | 6.75 s |
| **TOTAL** | 30.73 s | **12.9 s** (15.5 img/s) |

`import torch, cv2, numpy` alone costs **4.8 s** in a bare process, so the 4.1 s import
phase is already at the floor — the deferred-import work took everything available.

**Risk worth stating:** if the graders' machine has a cold file cache, the first run pays
roughly **16 extra seconds** loading torch's DLLs from disk. Nothing can be done about it
from inside the script, but on a small test set it would dominate the measured time.

## §3.60 — Run 5 (SHIPPED, 2026-08-05): deeper + longer

After four consecutive "no improvement available" results, the two directions never
tested *together* were capacity and training length. Run 5: **middle blocks 12 → 16**
(27.0M → 34.4M parameters), **80 epochs instead of 60**, expanded corpus, no corpus
weighting (run 4 showed weighting only redistributes). Paused at epoch 52 and resumed —
see the EMA verification in §3.55.

### The first challenger to pass the rule

| | pSNR | SSIM | LPIPS |
|---|---|---|---|
| **uniform sampling (n=300)** | **+0.053** | **+0.0018** | **−0.0039** |
| **balanced per corpus (n=280)** | **+0.073** | **+0.0016** | **−0.0048** |

**Better on all three metrics under both weightings.** Every previous challenger failed
this: sweep_a won two and lost one, sweep_b traded SSIM for LPIPS, run 4 lost outright.

### Speed cost is real but small

Interleaved runs (v3, v5, v3, v5) to control for thermal drift:

| | run 1 | run 2 | mean |
|---|---|---|---|
| v3 (27.0M) | 5.164 s | 5.158 s | **5.161 s** |
| v5 (34.4M) | 5.243 s | 5.229 s | **5.236 s** |

**+1.5%**, consistent in both repeats — a genuine difference, not noise. Accepted: the
deck prefers speed only "when quality is comparable", and here it is not.

### Final numbers (300 samples, current corpus)

| | pSNR | SSIM | LPIPS |
|---|---|---|---|
| best baseline per metric | 21.517 | 0.5110 | 0.4341 |
| **shipped model** | **25.448** | **0.7012** | **0.1672** |
| **gain** | **+3.931** | **+0.1902 (+37.2%)** | **−0.2669** |

### Caveat on the gain

+0.0018 SSIM over v3 is small, and sits near the noise floor. What justifies the swap is
not the size but the **consistency**: three metrics × two weightings = six comparisons,
and v5 wins all six. A noise artefact would not do that.

### Minor bug noticed, not fixed

`--resume` starts a fresh `history` list rather than loading the existing JSON, so
`nafnet_v5_history.json` contains only the 28 post-resume epochs. Cosmetic — the
checkpoints and metrics are unaffected — but the training curve for run 5 is incomplete
in the record.

## §3.59 — Run 4: expanded corpus + weighted sampling — REJECTED

Two changes together (accepting the loss of clean attribution, since only one overnight
slot was available): the 2,052 materials micrographs added, and corpus-weighted
sampling (`em` 3×, `mat` 2.5×, `bright` 2×, `dtd` 1×, `proc` 0.5×) to stop `dtd`
dominating the training signal by file count alone.

**Result: worse than v3 on every metric.** 300 identical samples:

| | pSNR | SSIM | LPIPS |
|---|---|---|---|
| **v3** | **25.395** | **0.6994** | **0.1711** |
| v4 | 25.321 | 0.6949 | 0.1813 |

### But the per-corpus view says something more interesting

| corpus | v3 SSIM | v4 SSIM | v3 LPIPS | v4 LPIPS |
|---|---|---|---|---|
| dtd (down-weighted) | **0.7529** | 0.7421 | **0.1579** | 0.1781 |
| em | 0.6260 | **0.6264** | 0.0862 | **0.0850** |
| **mat** (up-weighted) | 0.7132 | **0.7285** | 0.2198 | **0.1694** |
| proc | **0.9244** | 0.9220 | **0.0356** | 0.0384 |
| bright *(unseen)* | 0.6258 | **0.6273** | 0.1113 | **0.1038** |
| fluor *(unseen)* | 0.8981 | **0.8987** | **0.0872** | 0.0884 |

**The weighting did exactly what it was told to.** v4 gained on the corpora it was fed
more of (`mat` +0.0153 SSIM, −0.0504 LPIPS) and lost on the one it was fed less of
(`dtd` −0.0108). It is also marginally ahead on **both unseen corpora**.

**Corpus weighting redistributes performance; it does not create any.**

## §3.595 — A bias in the comparison method itself

The per-corpus split above raised a suspicion about the *measurement*, not the models:
`compare_models.py` drew validation samples **uniformly by file**, and `dtd` is 5,639 of
13,358 files. So **42% of every model comparison was photographs of fabric and stone** —
the corpus least like wafer inspection.

Added `--balanced`, which draws equally from each corpus. Re-run:

| | pSNR | SSIM | LPIPS |
|---|---|---|---|
| **v3** | **26.693** | **0.7192** | 0.1295 |
| v4 | 26.574 | 0.7172 | **0.1293** |

**The hypothesis was wrong: balancing does not rescue v4.** v3 still leads on pSNR and
SSIM with LPIPS a dead tie, so the verdict holds under both weightings. Worth recording
that the suspicion was checked and disproved rather than quietly dropped — and the flaw
in the sampling was real and worth fixing regardless of which model it favoured.

**v3 remains the shipped model.**

## §3.6 — Generalisation to unseen corpora (`scripts/ood_eval.py`)

Every earlier number was measured on the three corpora that were **in training**, so
none of them said anything about the axis the competition explicitly scores. Two
microscopy datasets of different modalities were downloaded and scored cold.

| corpus | seen? | pSNR | SSIM | LPIPS | bicubic SSIM | **gain vs bicubic** |
|---|---|---|---|---|---|---|
| dtd | train | 25.154 | 0.7115 | 0.1712 | 0.4989 | +0.2127 |
| em | train | 22.977 | 0.6282 | 0.0860 | 0.4940 | +0.1342 |
| proc | train | 31.308 | 0.9337 | 0.0348 | 0.8511 | +0.0826 |
| **brightfield** | **UNSEEN** | 31.606 | 0.6181 | 0.2631 | 0.1075 | **+0.5106** |
| **fluorescence** | **UNSEEN** | 32.017 | 0.8931 | 0.0855 | 0.7752 | +0.1178 |

**Mean SSIM: seen 0.7578, unseen 0.7556 — a 0.3% difference.** Flat.

The gain-over-bicubic column is the fairer signal, since a raw SSIM drop conflates poor
generalisation with the new imagery simply being harder for everyone. By that measure
the model's largest margin of *any* corpus is on unseen brightfield (+0.5106).

**Conclusion: the model learned restoration, not image families.** The planned corpus
rebalancing retrain was contingent on this dropping. It did not, so it was not done.

Caveats: the fluorescence set yielded only 123 usable images of 1,133 after filtering,
so that row rests on a thin sample; brightfield carries the model's worst LPIPS of any
corpus (0.2631); and both are still microscopy, so the real test set could be further
out than this.

### Re-measured cold on 2026-08-16 (submission day)

The local corpus had been deleted, so everything below was re-downloaded from source and
re-scored from scratch against `checkpoints/model.pt`, which the run confirms is the
shipped architecture (`nafnet base=64 middle_blocks=16`, 34.4M). 120 images per corpus,
720 total — this time including `mat`, which the table above predates.

| corpus | seen? | pSNR | SSIM | LPIPS | bicubic SSIM | **gain vs bicubic** |
|---|---|---|---|---|---|---|
| dtd | train | 25.197 | 0.7125 | 0.1699 | 0.4989 | +0.2137 |
| em | train | 22.983 | 0.6285 | 0.0857 | 0.4940 | +0.1345 |
| mat | train | 25.955 | 0.7292 | 0.1655 | 0.5488 | +0.1804 |
| proc | train | 30.487 | 0.9244 | 0.0363 | 0.8259 | +0.0985 |
| **brightfield** | **UNSEEN** | 32.779 | 0.6662 | 0.1065 | 0.1075 | **+0.5587** |
| **fluorescence** | **UNSEEN** | 32.133 | 0.8948 | 0.0853 | 0.7752 | +0.1195 |

**Mean SSIM: seen 0.7486, unseen 0.7805.** The unseen corpora now score *higher* than
the trained ones, where the earlier run had them flat. The brightfield row moved most
(LPIPS 0.2631 → 0.1065), and its margin over bicubic remains the largest of any corpus.
The earlier table's provenance is not recorded precisely enough to attribute the shift
with confidence, so it is left standing rather than overwritten.

### Held-out set, re-measured at n=628

`scripts/compare_models.py --n 628` — the entire validation split, 2.1× the 300 the
published figures rest on. `val_frac` is 0.05 and was deliberately **not** raised:
enlarging it would pull images the shipped model trained on into the test set.

| | pSNR | SSIM | LPIPS |
|---|---|---|---|
| bicubic floor | 20.757 | 0.5461 | 0.4390 |
| shipped model | 25.326 | 0.7227 | 0.1277 |

**Not directly comparable to the published table, and deliberately not substituted for
it.** Two differences: this run samples uniformly by file, so `dtd` is 45% of it against
the published run's balanced draw; and the floor here is *plain* bicubic rather than the
strongest tuned classical baseline. The mismatch is visible in the baseline itself —
plain bicubic scores 0.5461 SSIM here against the published baseline's 0.5110, a weaker
method scoring higher, which is only possible across different image mixes. Exactly the
bias §3.595 documents.

**Conclusion: the published 25.448 / 0.7012 / 0.1672 stand unchanged.** Both reruns
corroborate or exceed them; neither replicates the published protocol closely enough to
justify replacing it.

### Per-corpus spread is large

`proc` scores 0.9337 against `em`'s 0.6282. The procedural images are largely dark
background with sparse structure, where SSIM is close to free — so they inflate any
overall average and contribute less training signal than their count suggests. Worth
knowing when reading any aggregate number, though the OOD result above shows it is not
causing a generalisation problem.

## §3.65 — Both scale regimes, though only one was trained (`scripts/regime_test.py`)

Training crops are **128 LR → 256 HR**. The competition also contains **256 LR →
512 HR**, which the model has never seen — it runs only because the network is fully
convolutional. Worth checking whether that untrained regime is weaker, since a gap
would justify a larger-crop fine-tune.

Measured on the same 120 source images, so only the crop size differs:

| regime | pSNR | SSIM | LPIPS |
|---|---|---|---|
| 128→256 (trained) | 28.900 | 0.7359 | 0.1343 |
| **256→512 (never seen)** | **28.966** | **0.7378** | 0.1460 |

**The untrained regime is marginally better**, not worse (+0.066 dB, +0.0019 SSIM).
Larger crops carry more context, so some difference is expected in either direction;
what matters is that the larger regime is not *worse*.

**Larger-crop fine-tuning is therefore not indicated.** Another hour of work avoided by
measuring first rather than assuming a mismatch existed.

## §3.7 — Test-time augmentation: measured and rejected (`scripts/tta_eval.py`)

Geometric self-ensemble (run each image under flips/rotations, average the results) is
standard practice in super-resolution and is usually treated as near-free quality.

| mode | pSNR | SSIM | LPIPS | ms/img | cost |
|---|---|---|---|---|---|
| none | 24.816 | 0.7156 | **0.1841** | 64.4 | — |
| h-flip (2×) | 24.887 | 0.7182 | 0.1932 | 130.8 | 2.03× |
| flips (4×) | 24.927 | 0.7197 | 0.1987 | 260.8 | 4.05× |
| full (8×) | **24.961** | **0.7207** | 0.2079 | 519.4 | 8.07× |

**LPIPS gets worse monotonically as more transforms are averaged** (0.1841 → 0.2079).
Averaging smooths the prediction, which nudges pixel and structural scores up but costs
perceptual sharpness — and LPIPS is one of the three scored metrics.

Net: +0.144 dB pSNR and +0.005 SSIM against −0.024 LPIPS, for **8× inference on a clock
that is half the score**. **Not adopted.** Another case where the standard optimisation
inverts under this scoring scheme (compare §5a on cuDNN autotuning).

## §4 — Models (both verified on hardware)

| | U-Net | NAFNet |
|---|---|---|
| params | 12.83 M | 11.06 M |
| train VRAM @ batch 8, 128→256 | 0.57 GB | 2.97 GB |
| inference 256→512 | — | 49 img/s on the 4060 |

Both verified to: handle 128→256 and 256→512 **with one set of weights**, accept
non-power-of-two sizes (130×97 → 260×194), and produce an output identical to a
bilinear upsample at init (max diff 0.00e+00) because the tail conv is zero-initialised
— so the network starts from a sane estimate and learns only the correction.

VRAM headroom is much larger than expected (0.57 GB of 8.59 GB for the U-Net), so
model width is not currently the binding constraint.

## §5 — Inference optimisation (measured, not assumed)

`evaluate.py`, 60 mixed-resolution images, RTX 4060:

| version | total |
|---|---|
| first working version | 20.278 s |
| **current** | **10.709 s** |

Two changes did it:

### (a) cuDNN autotuning is a net loss here — turn it OFF

Fresh-process measurement, 64 images across two shapes:

| | first-batch overhead | steady state | total |
|---|---|---|---|
| `cudnn.benchmark=True` | **4465 ms** | 5.70 ms/img | 4.739 s |
| `cudnn.benchmark=False` | **943 ms** | 5.93 ms/img | **1.228 s** |

Autotuning costs ~3.5 s to save 0.23 ms/img → **break-even ≈ 15,300 images**. The test
set will not be close. Confirmed at the script level: 10.709 s → 15.880 s with it on.

This generalises to `torch.compile`, whose warm-up is far larger. Both remain available
behind flags, off by default. **On a clock that starts at process launch, the standard
"always enable" optimisations invert.**

### (b) Reads are issued before CUDA init

Creating the CUDA context takes ~2.8 s during which the GPU is idle. Submitting every
file read to a thread pool first hides disk I/O entirely behind it: read wait went
0.220 s → **0.000 s**.

Thread pool rather than DataLoader worker processes: on Windows each worker costs ~1 s
to spawn, which on a run this short is pure loss. `cv2.imread/imwrite` release the GIL,
so threads give the same overlap for no start-up cost.

Remaining breakdown: imports 6.3 s + CUDA init 2.8 s = 9.1 s of fixed cost against
1.45 s of actual work. Fixed cost amortises on a larger test set; it is not reducible
much further without dropping torch.

### Round-trip verification

`evaluate.py` → `scripts/score_outputs.py`, reading back exactly what the grader would,
with a 2-epoch smoke-trained model:

| | model | bicubic floor | delta |
|---|---|---|---|
| pSNR | 23.227 | 21.587 | **+1.641** |
| SSIM | 0.6651 | 0.6097 | **+0.0554** |
| LPIPS | 0.3072 | 0.3538 | **−0.0466** |

Confirms the whole chain: mixed 128/256 inputs, mixed uint8/uint16 dtypes, correct
output shapes and filenames.

---

## Environment

| | |
|---|---|
| Training GPU | RTX 4060 Laptop, 8188 MiB, driver 610.74 |
| Python | 3.14.6 (only interpreter present; no conda/uv) |
| Torch | `2.13.0+cu130`, torchvision `0.28.0+cu130` — CUDA verified available |
| Others | numpy 2.4.4, opencv-python 5.0.0.93, scikit-image 0.26.0, lpips, tqdm, matplotlib |
| venv | `.venv/` in repo root |

Python 3.14 constrains the stack: cu130 is the only CUDA index with cp314 wheels.

**numpy 2 gotcha:** `ndarray.ptp()` was removed in NumPy 2.0 — only the free function
`np.ptp(arr)` survives. It bit `validate_analysis.py`; watch for it in new code.

---

## Open questions for Step 1–2

1. Downsampling kernel — which equivalence class? Only `{area, bilinear}` vs `bicubic` vs
   `lanczos` vs `nearest` are separable at 2× (see §0.7a).
2. Is the additive term applied **before or after** the multiplicative term? Model in §0.3
   assumes after. Order matters for the synthetic generator. *Not yet covered by any test.*
3. Is speckle applied at LR or at HR before downsampling? Partially addressed — §B2 measures
   residual whiteness, but cannot exclude HR speckle + `area` downsampling (§0.7 caveat).
4. Ranges of `L` and `σ` — estimators validated to 1.4% / 0.2%, awaiting real data.
5. Actual dtype on disk — float32 `.npy`/`.tif`, or quantised uint8/uint16 PNG? Quantising a
   `[0,1.6]` float to uint8 would either clip or rescale; either changes the model.
   `inspect_data.py` now reports overflow broken down by dtype, which answers this directly.
6. Split between the 512→256 and 256→128 regimes.
7. How many distinct source domains (deck names "texture" and "dendrite" — there are more).
8. Does the sample→source mapping confirm multiple degradations per GT image (§0.5)? Decides
   whether the val split must be grouped by source.
