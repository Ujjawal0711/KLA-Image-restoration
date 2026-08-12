---
title: Degraded Image Restoration — KLA / i4C 2026
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Joint 2x super-resolution and speckle denoising for semiconductor inspection imagery
---

# Restoring degraded semiconductor imagery

Interactive demo for the **KLA problem statement**, i4C / SEMICON India Hackathon 2026.

A degraded inspection image arrives at **half resolution** and covered in
**multiplicative speckle**. The task is to undo both at once — denoise *and* double the
resolution — and to do it quickly, because end-to-end inference time is half the
competition score.

**No training data was released for this problem.** The entire training set is
synthetic: 13,358 ground-truth images across electron microscopy, materials
micrographs, textures, brightfield microscopy and procedurally generated wafer
structures, each damaged freshly on every use.

## What you can do here

- **Try it** — drag the damage controls and watch speckle scale with brightness, then
  restore. Corruption is pure NumPy and updates instantly; restoration runs a 34M
  parameter network in about 0.7 s on this Space's CPU.
- **Why it spills past white** — the histogram showing degraded pixels exceeding the
  ground-truth range, which is what makes this noise unusual.
- **Does it hold up?** — quality across noise levels far outside what the model trained
  on, plus results on microscopy types it has never seen.
- **How it was built** — including the five improvement ideas that measured *worse*.

## Results

Measured on 300 held-out images against the strongest non-learned baseline:

| | baseline | model | gain |
|---|---|---|---|
| pSNR | 21.517 | **25.395** | **+3.878 dB** |
| SSIM | 0.5110 | **0.6994** | **+0.1884 (37%)** |
| LPIPS | 0.4341 | **0.1711** | **−0.2630** |

Against plain bicubic upscaling: **+5.8 dB, +48% SSIM**, perceptual error cut to about
a third.

## Model

NAFNet, 34.4M parameters. All computation happens at low resolution with a single
`PixelShuffle` upscale at the end — 4× cheaper than upscaling first. Fully
convolutional, so the same weights serve both competition regimes (128→256 and
256→512) despite only ever training on the smaller one.

Trained on an 8 GB laptop GPU under mixed precision with EMA weight averaging.
