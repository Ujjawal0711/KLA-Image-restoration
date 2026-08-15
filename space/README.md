---
title: Restoring Degraded Semiconductor Imagery
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Live 2x super-resolution and speckle denoising for wafers
---

# Restoring degraded semiconductor imagery

Live demo for the **KLA problem statement**, i4C / SEMICON India Hackathon 2026.

**Every restore on this page is a real forward pass.** Nothing is pre-computed, and the
page reports how long the network actually took.

Code and full engineering log: <https://github.com/ujjawal0711/KLA-Image-restoration>

---

## 1. The problem

Semiconductor wafer inspection produces images that are degraded twice over at once:

1. **Half resolution** — the image arrives downsampled by 2× in each axis.
2. **Multiplicative speckle** — coherent-imaging noise that *scales with brightness*
   instead of being added on top of it.

The task is to invert both simultaneously and recover the clean, full-resolution image.

### What made this hard: no data, and no stated degradation

**No training pairs were ever released, and no degradation model was documented.** There
was nothing to fit to and nothing to validate against. Before any network could be
trained, the damage itself had to be worked out.

The only quantitative evidence that existed anywhere was embedded in the metadata of the
problem statement's own sample figures — two settings, recovered from the figure
captions inside the slide deck:

| deck sample | `L` | `σ` |
|---|---|---|
| figure 1 | 16.86 | 0.008594 |
| figure 2 | 18.13 | 0.001065 |

From those two points the forward model was reconstructed as:

```
clean_lr = downsample(ground_truth, 2)
degraded = clean_lr * g + n        g ~ Gamma(L, 1/L)      n ~ N(0, σ²)
```

`g` has mean 1 and variance `1/L`, so it darkens and brightens pixels in proportion to
their own value. **Nothing is clipped anywhere.** Multiplicative noise legitimately
pushes bright pixels past 1.0 — you can watch this in the demo's overflow readout, where
at `L = 4` roughly a third of all pixels exceed pure white.

The estimators were validated against synthetic data with known ground truth, recovering
`L` to **1.4%** and `σ` to **0.2%**.

### The strategy that follows from an unverifiable model

Two data points do not pin down a distribution. With parameters you can measure, you
tune to match them; **without them, the correct move is to train across a deliberately
wide range** and accept slightly worse performance at any single setting in exchange for
not falling off a cliff at the true one. A wrong point estimate is far worse than a
range that contains the truth.

| parameter | range trained | reasoning |
|---|---|---|
| `L` | 8 – 40 | deck showed 16.86 and 18.13; widened ~2× both directions |
| `σ` | 5e-4 – 2e-2, log-uniform | deck showed an 8× spread → log scale |
| kernel | area, bilinear, bicubic, lanczos | the real one cannot be identified, so all four are sampled |

Every training sample draws a **fresh** `(kernel, L, σ)`, so the network never sees the
same degradation twice and cannot overfit to one noise level.

---

## 2. Approaches considered, and why they were not chosen

### Rejected at design time, on domain reasoning

| Approach | Pros | Cons | Verdict |
|---|---|---|---|
| **Classical DSP** — bicubic + median / bilateral | Zero training, instant, fully interpretable | Hits a hard ceiling; cannot invent detail | Kept as the **baseline to beat**, not the solution. Best variant reaches 21.8 dB and stops |
| **GAN** (Real-ESRGAN style) | Sharpest perceptual output, excellent LPIPS | **Hallucinates plausible texture** | **Disqualifying here.** This is *inspection* imagery — an invented speckle that looks like a defect is a false positive on a wafer. Sharpness that isn't real is worse than mild blur |
| **Diffusion** (SR3 / StableSR) | Best-in-class perceptual quality | 10–1000× slower; same hallucination risk | Wall-clock is scored, and unfaithful detail is unacceptable for the same reason as GANs |
| **Transformer** (SwinIR / Restormer) | Very strong on restoration benchmarks | Heavier, slower, more data-hungry | With a *fully synthetic* training set, the extra capacity buys generalisation risk, not accuracy |
| **Two-stage: denoise → then upscale** | Modular, each stage debuggable | Two models, two error sources | Measured, not assumed — see the correction below |

### Rejected after measurement

These are the more interesting ones, because five of six ideas that *should* have worked
did not. All were measured before being discarded:

| Idea | Result | Kept? |
|---|---|---|
| **Test-time augmentation** (8× self-ensemble) | pSNR +0.144, SSIM +0.005, but **LPIPS degrades monotonically** 0.1841 → 0.2079, at **8× inference cost** | ❌ Averaging smooths away perceptual sharpness, and LPIPS is scored |
| **Larger training crops** (256 → 512) | The untrained 256→512 regime already scores *better* than the trained one | ❌ No mismatch existed to fix |
| **Loss-weight sweep** | Already sitting on the frontier; every variant traded one metric for another | ❌ No variant won on all three |
| **Corpus reweighting** | Gains exactly what it is fed more of, loses what it is fed less of | ❌ **Redistributes performance; does not create any** |
| **Kernel reweighting** | Would encode a prior about the true kernel | ❌ Declined — no evidence supports one kernel over another |
| **More capacity + longer training** | Won on all three metrics under both sampling schemes | ✅ **This is the shipped model** |

> **A correction worth recording.** "Denoise-first beats upscale-first by 1.0 dB" was
> measured early on **150 samples** and written up as independent support for the
> architecture. Re-measured at **400 samples**, the gap shrinks to ~0.3 dB and SSIM and
> LPIPS *reverse outright*. The original claim did not survive a larger sample and was
> retracted. Sample size was the bug, not the architecture.

---

## 3. The solution

### NAFNet, single-stage

A **34.4M-parameter NAFNet** (Nonlinear Activation Free Network) handles denoising and
super-resolution in one pass. Three properties made it the right fit:

- **No activation functions.** NAFNet replaces ReLU/GELU with **SimpleGate** — split the
  channels in half and multiply the halves together. Cheaper, and it consistently
  outperforms activation-based blocks on restoration.
- **Simplified channel attention**, which keeps global context without attention's cost.
- **Restoration-native.** It was designed for denoising and deblurring, not
  classification, so nothing is fighting the task.

### Three implementation decisions that mattered

**Upscale last, not first.** The `PixelShuffle(2)` sits at the very end of the network,
so every convolution runs at low resolution. Upscaling first would have made the entire
network **4× more expensive** for no quality gain.

**Initialised to output an exact bilinear upscale.** The network starts life already
producing a competent answer, so it only ever has to learn the *correction*. This
removes the early phase where the model would otherwise waste epochs rediscovering
interpolation.

**EMA weight averaging**, with a warmup ramp so early noisy weights do not dominate the
average.

### A training set built from nothing

With no released data, the entire corpus is synthetic: **13,358 ground-truth images**,
each degraded freshly at use time.

| corpus | images | why it is in there |
|---|---|---|
| `dtd` | 5,639 | 47 texture families — structural diversity |
| `em` | 3,367 | real electron microscopy — closest public analogue to wafer inspection |
| `mat` | 2,052 | materials-science micrographs (SEM/TEM) |
| `proc` | 1,500 | procedural dendrite / grain / wafer-lattice structures |
| `bright` | 800 | brightfield microscopy |

The `mat` corpus is scraped from publication figures, so it needed a rejection filter —
the discriminator that finally worked tests for the **largest contiguous near-white
blob**, since plot margins and panel gutters form one big connected white region while
real micrographs essentially never do. That cut 3,050 candidates to 2,052.

---

## 4. Results

Measured on **300 held-out images**, against the strongest *non-learned* baseline
(itself tuned — plain bicubic is considerably weaker):

| | best classical baseline | **model** | gain |
|---|---|---|---|
| pSNR | 21.517 | **25.448** | **+3.931 dB** |
| SSIM | 0.5110 | **0.7012** | **+0.1902 (37%)** |
| LPIPS | 0.4341 | **0.1672** | **−0.2669** |

### It generalises, and this was tested cold

The competition explicitly scores behaviour on unfamiliar imagery, so the model was
evaluated on microscopy modalities **never present in training** — brightfield and
fluorescence, neither of which is electron microscopy:

**Mean SSIM: seen corpora 0.7578, unseen corpora 0.7556 — a 0.3% difference.**

Better still: measured as *gain over bicubic on the same files*, the model's **largest
margin of any corpus is on unseen brightfield**. It learned restoration, not image
families.

### It degrades gracefully

Pushed well outside its trained range, the model bends rather than breaks — less noise
than expected simply scores better, and the failure mode below the trained floor is a
gradual decline rather than a collapse.

---

## 5. Using it

**In this demo:** pick a sample (all held out from training) or upload your own, drag
**L** and **σ** to set the damage, then press **Restore**. Damage updates instantly
because corrupting an image is pure NumPy; restoring runs the real network on the free
CPU this Space is hosted on, so give it a few seconds.

Good place to start: **Brightfield (unseen corpus)** at **L = 4** — bicubic scores about
0.05 SSIM there, the model about 0.70, on a corpus it never trained on.

**On your own machine**, from the GitHub repo:

```bash
python evaluate.py <input_dir> <output_dir>
```

One set of weights handles **both** 128→256 and 256→512 — the network is fully
convolutional and the scale factor is always exactly 2. Input is deliberately *not*
clipped (speckle legitimately exceeds the ground-truth range, and the model was trained
that way); output is clipped to [0,1], and dtype, extension and filename all mirror the
input.
