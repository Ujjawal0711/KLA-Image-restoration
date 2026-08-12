# Work split

The project divides into three self-contained areas. Each owner maintains their area,
reviews changes to it, and is the person to ask about it.

| Part | Area | Owner |
|---|---|---|
| 1 | Data & degradation modelling | **Shubhankar Ohri** |
| 2 | Model, training & experiments | **Ujjawal Chaudhary** |
| 3 | Inference, packaging & demo | **Rashiv Saran** |

The three parts are roughly equal in size and depend on each other through narrow,
stable interfaces: Part 1 produces `(degraded, clean)` tensor pairs, Part 2 consumes
them and produces a checkpoint, Part 3 consumes the checkpoint and produces scored
output files.

---

## Part 1 — Data & degradation modelling  ·  Shubhankar Ohri

**The question this part answers:** no paired training data was ever released, so where
does training data come from, and how exactly did the organisers damage their images?

### Files

```
src/degradation.py              forward degradation model
src/procedural.py               dendrite / granular / grain / lattice / fibre generators
src/dataset.py                  on-the-fly synthetic pair generation, corpus weighting
scripts/fetch_data.py           corpus download, micrograph filtering
scripts/make_procedural.py      procedural corpus generation
scripts/analyze_degradation.py  parameter estimators
scripts/validate_analysis.py    estimator validation against known parameters
scripts/inspect_data.py         dataset layout discovery
scripts/preview_degradation.py  visual check against the deck's own figures
```

### What the work involved

Recovering the degradation model from the problem statement itself — the deck's sample
figures are auto-generated charts whose titles carry the generator's parameters
(`L=16.86, σ=0.008594`). Building the forward model from those, then assembling a
13,358-image ground-truth corpus from five sources and generating wafer-like structures
in code to cover what public datasets do not.

### Key things this owner should be able to explain

- **Why the noise is multiplicative, and why nothing is clipped.** Speckle scales with
  brightness, which is why degraded pixels legitimately exceed the ground-truth range.
  Clipping the input anywhere would train the model on images the test set will not
  contain.
- **`area` and `bilinear` are the same operation at exactly 2×**, verified to float32
  epsilon. The downsampling kernel is only identifiable up to an equivalence class, so
  reporting a single winner would be false precision.
- **Why the estimators were validated on synthetic data with planted answers** before
  being trusted — recovering L to 1.4% and σ to 0.2% on data that provably follows the
  model. Code that cannot recover known parameters cannot be trusted on unknown ones.
- **The corpus filter had to be rewritten after visual inspection.** The first version
  measured the *fraction* of white pixels and let through an XRD plot, a table of
  numbers and several multi-panel figures — about 1 in 4. The fix tests the largest
  *contiguous* white region instead: plot margins and panel gutters are one connected
  blob, real micrographs never are. Kept images dropped 3,050 → 2,052, deliberately.

---

## Part 2 — Model, training & experiments  ·  Ujjawal Chaudhary

**The question this part answers:** what architecture, trained how, and is it actually
better than the alternatives?

### Files

```
src/models/nafnet.py            shipped architecture
src/models/unet.py              baseline
src/models/__init__.py          architecture reconstruction from a checkpoint
src/losses.py                   Charbonnier + SSIM + LPIPS
src/metrics.py                  pSNR / SSIM / LPIPS
src/train.py                    training loop, EMA, OOM backoff, resume
scripts/baseline_scores.py      non-learned floors the model must beat
scripts/compare_models.py       score checkpoints on identical data
scripts/sensitivity_sweep.py    quality beyond the trained noise range
scripts/ood_eval.py             score on corpora never seen in training
scripts/tta_eval.py             test-time augmentation cost/benefit
scripts/regime_test.py          128→256 vs 256→512
scripts/ordering_test.py        does the assumed noise order matter?
```

### What the work involved

Five full training runs on an 8 GB laptop GPU, and the experiments that decided between
them. NAFNet at 34.4M parameters, all computation at low resolution with a single
upscale at the end, initialised to output exactly a bilinear upscale so it learns only
the correction.

### Key things this owner should be able to explain

- **Why upsample last.** Processing at low resolution is exactly 4× cheaper, which is
  arithmetic rather than statistics.
- **Why models are never compared on their own validation numbers.** Each run validates
  against its own degradation range, so those numbers are not comparable across runs.
  `compare_models.py` exists to score every candidate on byte-identical inputs.
- **The decision rule for swapping models, fixed before results were seen:** win on all
  three metrics, or two with a third-metric loss inside noise; ties go to the incumbent.
  With four candidates and three metrics it is always possible to find a story
  afterwards in which something won.
- **Five improvement directions measured worse or neutral** — test-time augmentation
  (hurts LPIPS at 8× cost), larger crops, loss-weight tuning, corpus rebalancing, kernel
  reweighting. Those results are worth as much as the one that worked.
- **A correction in the record.** An early 150-sample measurement showed
  denoise-before-upsample winning by 1.0 dB and was written up as support for the
  architecture. At 400 samples the gap is ~0.3 dB and the SSIM and LPIPS results
  reverse. The architecture still stands on the compute argument; that experiment does
  not support it.

---

## Part 3 — Inference, packaging & demo  ·  Rashiv Saran

**The question this part answers:** the competition scores wall-clock time as heavily as
quality, so how fast can the delivered pipeline be, and does it survive contact with
inputs nobody anticipated?

### Files

```
evaluate.py                     the scored deliverable
scripts/promote_checkpoint.py   strip optimizer state, fp16 weights
scripts/edge_cases.py           corrupt files, odd sizes, empty dirs, RGB input
scripts/check_submission.py     self-containment audit
scripts/package_submission.py   assemble the package, test it in isolation
scripts/dress_rehearsal.py      full end-to-end run under grading conditions
scripts/finalize.py             one command for everything post-training
space/                          interactive Gradio demo
```

### What the work involved

The timed deliverable and everything that protects it. Deferred imports, disk reads
overlapped behind CUDA initialisation, batching by resolution, fp16 inference, and a
packaging pipeline that proves the submission runs from inside its own folder with
nothing else on the path.

### Key things this owner should be able to explain

- **cuDNN autotuning is a net loss here, and that is counter-intuitive.** It costs ~3.5 s
  of warm-up to save 0.23 ms per image, so it only pays past ~15,300 images. The clock
  starts at process launch, not at first inference. Disabling it made the pipeline
  **47% faster**.
- **Why a crash matters more than a low score.** A failure partway through loses every
  remaining image. `edge_cases.py` deliberately feeds corrupt files, odd dimensions,
  RGB input, nested folders and empty directories; a corrupt file is skipped while its
  neighbours are still written.
- **Why the isolation test exists.** `evaluate.py` is run from *inside* the packaged
  folder with `PYTHONPATH` cleared, so any accidental dependence on an unshipped file
  surfaces here rather than on the graders' machine.
- **Import time is the largest single cost and it is irreducible.** 4.1 s of a 12.9 s
  run for 200 images, and `import torch` alone accounts for essentially all of it. A
  cold file cache adds ~16 s more, which nothing in the script can prevent.
- **`cv2.imwrite` returns `False` rather than raising** on a full disk or a permissions
  error. Unchecked, that silently produces a short output set.

---

## Shared

`NOTES.md`, `README.md` and `Project-Report.pdf` are shared — every measurement in them
comes from one of the three areas above, and each owner is responsible for the sections
covering their own work.
