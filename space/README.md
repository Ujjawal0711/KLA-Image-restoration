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

A degraded inspection image arrives at **half resolution** covered in **multiplicative
speckle**. The task is to undo both at once — denoise *and* double the resolution.

**Every restore here is a real forward pass.** Nothing is pre-computed, and the page
reports how long the network took.

## Try it

- Pick a sample (all held out from training) or upload your own image
- Drag **L** — speckle strength. Lower is noisier. The damage updates instantly, because
  corrupting an image is pure NumPy
- Press **Restore** — the 34.4M-parameter network runs on CPU, about 0.7 s for a
  128×128 input
- Watch the overflow readout: at L = 4 roughly a third of pixels exceed 1.0, because
  the noise multiplies rather than adds

Good place to start: **Brightfield (unseen corpus)** at **L = 4**. Bicubic scores 0.05
SSIM there; the model scores 0.70 — on a corpus it never trained on.

## Results

Measured on 300 held-out images against the strongest non-learned baseline:

| | baseline | model | gain |
|---|---|---|---|
| pSNR | 21.517 | **25.395** | **+3.878 dB** |
| SSIM | 0.5110 | **0.6994** | **+0.1884 (37%)** |
| LPIPS | 0.4341 | **0.1711** | **−0.2630** |

Against plain bicubic: **+5.8 dB, +48% SSIM**.

**No training data was released for this problem.** The entire training set is
synthetic, built from a degradation model reverse-engineered out of parameters embedded
in the problem statement's own sample figures.

Code and full write-up: https://github.com/ujjawal0711/KLA-Image-restoration
