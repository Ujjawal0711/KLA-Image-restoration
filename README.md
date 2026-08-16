# AI-Based Restoration of Degraded Images — KLA / i4C Hackathon 2026

Joint 2× super-resolution and multiplicative-speckle denoising for semiconductor
inspection imagery.

**No training data was released for this problem.** The entire training set is
synthetic, generated from a degradation model reverse-engineered out of the problem
statement's own slide deck.

## Results

Measured on 300 held-out images against the strongest non-learned baseline
(`scripts/compare_models.py --n 300`):

| | best classical baseline | **model** | gain |
|---|---|---|---|
| pSNR | 21.517 | **25.448** | **+3.931 dB** |
| SSIM | 0.5110 | **0.7012** | **+0.1902 (37%)** |
| LPIPS | 0.4341 | **0.1672** | **−0.2669** |

Full end-to-end dress rehearsal on 200 unseen images — 95 of them from corpora never
used in training — run through the packaged submission under grading conditions:
**+5.818 dB, +0.2117 SSIM, −0.2777 LPIPS**, 200 in / 200 out, all exactly 2×. That
rehearsal was run against the run-3 checkpoint (see NOTES.md §3.99), so it understates
the shipped run-5 model rather than flattering it.

## Work split

The project divides into three self-contained areas, each with an owner — see
[`OWNERSHIP.md`](OWNERSHIP.md) for scope, files and the key decisions behind each.

| Part | Area | Owner |
|---|---|---|
| 1 | Data & degradation modelling | **Shubhankar Ohri** |
| 2 | Model, training & experiments | **Ujjawal Chaudhary** |
| 3 | Inference, packaging & demo | **Rashiv Saran** |

They connect through narrow interfaces: Part 1 produces `(degraded, clean)` pairs,
Part 2 turns them into a checkpoint, Part 3 turns the checkpoint into scored output
files.

## Submission deliverables

| # | Required | Here |
|---|---|---|
| 1 | Standalone evaluation script | [`evaluate.py`](evaluate.py) |
| 2 | Training script | [`src/train.py`](src/train.py) |
| 3 | Restored test outputs | produced by `evaluate.py` |
| 4 | Environment spec | [`requirements.txt`](requirements.txt) (full `pip freeze`) |

```bash
python evaluate.py <test_image_dir> <output_dir>
```

Runs unmodified with the two required positional arguments. One set of weights handles
**both** 128→256 and 256→512 — the network is fully convolutional and the scale factor
is always exactly 2. Verified: the 256→512 case, which the model never trained on,
scores marginally *better* than the case it did.

## The central problem: reverse-engineering the damage

With no paired data, everything depends on knowing exactly how the organisers degraded
their images. A `.pptx` is a zip archive, and the deck's sample figures turned out to be
auto-generated charts whose **titles carry the generator's own parameters**:

```
Training - Sample 000000    Source: 0001.png | L=16.86, σ=0.008594
Training - Sample 000500    Source: 0186.png | L=18.13, σ=0.001065
```

That is the only quantitative evidence about the degradation in existence. It yields:

```
clean_lr = downsample(GT, 2)
degraded = clean_lr * g + n,     g ~ Gamma(L, 1/L),     n ~ N(0, σ²)
```

Speckle is **multiplicative**, so it scales with brightness and pushes bright pixels
past the ground-truth range — which is exactly the behaviour the deck flags as expected.
Nothing is clipped on input anywhere in this pipeline; clipping would train the model on
images the test set will not contain.

It also settled an ambiguity: one slide describes a third degradation as "reduction in
image sharpness and edge detail", which reads like blur. The recovered σ is far too
small to be a blur kernel and is plainly an additive-noise standard deviation.

**This model is inferred and can never be verified.** The mitigation is to train across
roughly double the observed noise range in both directions, trading a little peak
accuracy for not collapsing if the inference is wrong. See §3.6 of
[`NOTES.md`](NOTES.md) for the robustness sweep that measures exactly how much that
buys.

## Interactive demo

Pick a sample or upload your own, drag the damage controls and watch speckle scale with
brightness, then restore. **Every restore is a real forward pass** — nothing is
pre-computed, and the page reports how long the network took. Corruption is pure NumPy
and returns in milliseconds; restoration runs the 34.4M-parameter model in ~0.7 s for a
128×128 input on CPU. That contrast is the point: the damage is trivial to apply and
hard to undo.

Run it locally:

```bash
python demo_app.py          # then open http://localhost:7860
```

**Live version:** <https://huggingface.co/spaces/Ujjawal0711/kla-image-restoration>

[`space/`](space/) holds that deployment — a Gradio SDK Space, and a separate front end
from `demo_app.py` (which is dependency-free stdlib `http.server`). All of its logic
lives in [`space/core.py`](space/core.py), which imports no Gradio at all so
[`space/test_core.py`](space/test_core.py) can exercise every path without it installed.

### Images to try it on

[`Gallery/`](Gallery/) has 18 ready-to-upload 512×512 images spanning all six corpora,
with [suggested damage settings](Gallery/README.md) — including the exact parameters
recovered from the problem statement's own figures (`L = 16.86, σ = 0.0086`).

**None of them were trained on.** The four in-training corpora draw only from the
deterministic validation split (`val_frac=0.05`, `seed=0`), the same split the published
metrics use; brightfield and fluorescence were never in training at all.

## Reproducing from scratch

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```

```bash
python scripts/fetch_data.py          # download ground-truth corpora
```

```bash
python scripts/make_procedural.py --n 1500    # generate wafer-like structures
```

```bash
python scripts/baseline_scores.py     # non-learned floors the model must beat
```

```bash
python -m src.train --model nafnet --base 64 --middle-blocks 16 --epochs 80 --ema 0.999
```

## Model

**NAFNet**, 34.4M parameters. All computation happens at low resolution with a single
`PixelShuffle` upscale at the very end — 4× cheaper than upscaling first. The network
is initialised so that it outputs exactly a bilinear upscale of its input, and learns
only the correction on top.

Trained on an 8 GB laptop GPU under mixed precision with EMA weight averaging, automatic
batch-size backoff on OOM, and per-epoch checkpointing. Loss combines Charbonnier
(robust to speckle outliers), SSIM and LPIPS.

Training data: **13,358 ground-truth images** across electron microscopy, materials
micrographs, textures, brightfield microscopy and procedurally generated wafer
structures. Every image is degraded **freshly on each use** with newly drawn parameters,
so the model never sees the same corruption twice.

## Notable findings

Full detail and every measurement in [`NOTES.md`](NOTES.md).

- **`area` and `bilinear` downsampling are the same operation at exactly 2×**, verified
  to float32 epsilon. The downsampling kernel is therefore only identifiable up to an
  equivalence class — reporting a single winner would be false precision.

- **cuDNN autotuning is a net loss here.** It costs ~3.5 s of warm-up to save 0.23 ms
  per image, so it only pays past ~15,300 images. The competition clock starts at
  process launch, so it is disabled — which made the pipeline **47% faster**.

- **The model generalises.** Scored cold on two microscopy modalities never seen in
  training, mean SSIM differs by 0.8% from the corpora it trained on, and its largest
  margin over bicubic of any corpus is on unseen brightfield imagery.

## Five things that did not work

Six improvement directions were tested; five measured worse or neutral. They are
documented as carefully as the one that worked.

- **Test-time augmentation** — the textbook free win. Measured: it makes LPIPS *worse*
  (0.1841 → 0.2079) because averaging smooths away the sharpness that metric rewards,
  at 8× the inference cost.
- **Larger training crops** — no regime gap exists to close.
- **Loss weight tuning** — a 3-point sweep moves you along the trade-off between the
  three metrics but not off it. The original weights were already a sensible point.
- **Rebalancing the training corpus** — improves the corpora fed more, degrades those
  fed less, nets to nothing. Capacity is fixed.
- **Reweighting the downsampling kernels** — would encode a prior the evidence does not
  support.

One correction worth flagging, recorded in full in `NOTES.md` §3: an early measurement
on 150 samples showed denoise-before-upsample winning by 1.0 dB and was written up as
independent support for the architecture. Re-measuring on 400 samples cut the gap to
~0.3 dB and reversed the SSIM and LPIPS results. The architecture choice still stands on
the compute argument, which is arithmetic rather than statistics — but that experiment
does not support it, and the original claim was overstated.

## Layout

```
evaluate.py                  the deliverable — inference pipeline
NOTES.md                     every empirical finding and measurement
Project-Report.pdf           32-page written report
Gallery/                     18 held-out images to try the demo on
space/                       the live demo, a Gradio Space
src/
  degradation.py             forward degradation model
  dataset.py                 on-the-fly synthetic pair generation
  procedural.py              dendrite / granular / grain / lattice / fiber generators
  models/nafnet.py           shipped architecture
  models/unet.py             baseline
  losses.py                  Charbonnier + SSIM + LPIPS
  metrics.py                 pSNR / SSIM / LPIPS
  train.py                   training loop
scripts/
  fetch_data.py              download and filter ground-truth corpora
  make_procedural.py         generate synthetic wafer structures
  baseline_scores.py         non-learned floors
  compare_models.py          score checkpoints on identical data (--balanced)
  sensitivity_sweep.py       quality across noise levels beyond the trained range
  ood_eval.py                score on corpora never seen in training
  regime_test.py             128→256 vs 256→512
  tta_eval.py                test-time augmentation cost/benefit
  ordering_test.py           does the assumed noise order matter?
  edge_cases.py              corrupt files, odd sizes, empty dirs, RGB input
  check_submission.py        self-containment audit
  package_submission.py      assemble the package and test it in isolation
  dress_rehearsal.py         full end-to-end run under grading conditions
  finalize.py                one command for everything post-training
  validate_analysis.py       validates the estimators against known parameters
  analyze_degradation.py     degradation estimators (unused — no data released)
  inspect_data.py            dataset inspection (unused — no data released)
```

---

## Team

Built for the **KLA problem statement**, i4C / SEMICON India Hackathon 2026.

| | |
|---|---|
| **Ujjawal Chaudhary** | Model, training & experiments |
| **Shubhankar Ohri** | Data & degradation modelling |
| **Rashiv Saran** | Inference, packaging & demo |

See [`OWNERSHIP.md`](OWNERSHIP.md) for what each area covers and the reasoning behind
the decisions in it.
