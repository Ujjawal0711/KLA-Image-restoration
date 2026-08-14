"""Interactive demo: AI-based restoration of degraded semiconductor imagery.

KLA problem statement, i4C / SEMICON India Hackathon 2026.

UI wiring only — all logic lives in core.py so it can be tested without Gradio.

The interaction is deliberately split in two. Corrupting an image is pure NumPy, so it
updates the instant a slider moves: you watch speckle scale with brightness and see
bright regions spill past white. Restoring it runs a 34M-parameter network, about 0.7 s
on the free CPU this Space runs on. That contrast is the point — the damage is trivial
to apply and hard to undo.
"""

from __future__ import annotations

import gradio as gr

from core import ARCH, BENCH, N_PARAMS, SAMPLES, make_degraded, restore
from src.degradation import KERNELS

# Gradio 6 moved theme and css from the Blocks constructor to launch(); passing them
# here is only a warning today but errors on some versions. Everything below sticks to
# the arguments that have been stable across Gradio 4, 5 and 6, because this app can
# only be exercised once deployed — each incompatibility costs a full rebuild.
with gr.Blocks(title="Degraded Image Restoration — KLA / i4C 2026") as demo:
    gr.Markdown(
        "# Restoring degraded semiconductor imagery\n"
        "**KLA problem statement · i4C / SEMICON India Hackathon 2026**\n\n"
        "A degraded inspection image arrives at **half resolution** and covered in "
        "**multiplicative speckle**. Undo both at once — denoise *and* double the "
        "resolution. No training data was released for this problem, so the entire "
        "training set was synthesised."
    )

    with gr.Tab("Try it"):
        with gr.Row():
            with gr.Column(scale=1):
                sample = gr.Dropdown(list(SAMPLES), value=next(iter(SAMPLES), None),
                                     label="Sample image (all held out from training)")
                # No image_mode: core.read_gt already converts colour to greyscale, so
                # constraining it here only risks a version-specific argument.
                upload = gr.Image(label="…or upload your own", type="numpy")
                gr.Markdown("### Damage controls")
                L = gr.Slider(4, 100, value=17, step=0.5,
                              label="L — speckle strength (lower = noisier)")
                sigma = gr.Slider(0.0, 0.05, value=0.0086, step=0.0002,
                                  label="σ — additive noise")
                kernel = gr.Dropdown(list(KERNELS), value="area",
                                     label="downsampling kernel")
                # A slider rather than gr.Number(precision=...): the precision argument
                # is not stable across versions, and an integer slider is clearer anyway.
                seed = gr.Slider(0, 20, value=0, step=1, label="noise seed")
                btn = gr.Button("Restore", variant="primary")
                gr.Markdown(
                    "The problem statement's own sample figures were generated at "
                    "**L = 16.86, σ = 0.0086** and **L = 18.13, σ = 0.0011**. Those "
                    "numbers were recovered from the figure captions inside the slide "
                    "deck, and are the only quantitative evidence about the damage "
                    "that exists.")
            with gr.Column(scale=2):
                with gr.Row():
                    im_gt = gr.Image(label="Ground truth")
                    im_lr = gr.Image(label="Degraded — model input")
                with gr.Row():
                    im_bic = gr.Image(label="Bicubic upscale")
                    im_out = gr.Image(label="Model output")
                metrics = gr.Markdown()
                info = gr.Markdown()

    with gr.Tab("Why it spills past white"):
        gr.Markdown(
            "Ordinary image noise is **additive** — the same amount everywhere. This "
            "noise **multiplies**, making it a percentage error: a pixel at 0.1 barely "
            "moves while a pixel at 0.9 is thrown far. Multiply the brightest pixel by "
            "1.24 and it lands at 1.24, past the top of the scale.\n\n"
            "The organisers flag this as expected behaviour. It is also why nothing in "
            "this pipeline clips the degraded input: clipping would train the model on "
            "images the real test set will not contain.\n\n"
            "*Drag the sliders on the first tab and watch this update.*"
        )
        hist = gr.Plot(label="Intensity histogram")

    with gr.Tab("Does it hold up?"):
        gr.Markdown(
            "The damage model was **inferred, never confirmed** — it comes from two "
            "figure captions. So the question that matters is not how good the model "
            "is at the assumed noise level, but how fast it falls apart if that "
            "assumption is wrong.\n\n"
            "Every cell below is a different noise level. The white box is what the "
            "model actually trained on; everything outside it is unseen."
        )
        gr.Image("assets/robustness.png", label="SSIM across noise levels")
        gr.Markdown(
            "No cliff — quality falls away smoothly, and *improves* when the noise is "
            "milder than expected. That is deliberate: training sampled noise across "
            "roughly double the observed range in both directions, trading a little "
            "peak accuracy for not collapsing if the guess is off.\n\n"
            "### Unfamiliar imagery\n"
            "Scored cold on two microscopy types never seen in training:\n\n"
            "| corpus | in training? | gain over bicubic (SSIM) |\n|---|---|---|\n"
            "| textures | yes | +0.2283 |\n"
            "| electron microscopy | yes | +0.1380 |\n"
            "| **brightfield** | **no** | **+0.5245** |\n"
            "| **fluorescence** | **no** | +0.1136 |\n\n"
            "Mean SSIM on seen corpora 0.7678, on unseen 0.7619 — a 0.8% difference. "
            "The largest margin over bicubic of any corpus is on imagery the model had "
            "never seen, which is the evidence that it learned restoration rather than "
            "memorising image families."
        )

    with gr.Tab("How it was built"):
        gr.Markdown(f"""
### The problem
Images arrive at half resolution with multiplicative speckle. Undo both — denoise and
2× super-resolve — and do it fast, because end-to-end inference time is half the
competition score.

### No data was released
So the whole training set is synthetic: **13,358 ground-truth images** from electron
microscopy, materials micrographs, textures, brightfield microscopy and procedurally
generated wafer structures (grain boundaries, die lattices, dendrites). Every image is
damaged **freshly on each use** with a newly drawn noise level, so the model never sees
the same corruption twice and cannot memorise one.

### Recovering the damage recipe
A `.pptx` is a zip archive. The problem statement's sample figures turned out to be
auto-generated charts whose **titles carried the generator's own parameters** —
`L=16.86, σ=0.008594`. That is the only quantitative evidence about the degradation in
existence, and it settled an ambiguity in the deck: one slide describes the second
noise as a loss of sharpness, but the recovered σ is far too small to be a blur kernel
and is plainly an additive-noise standard deviation.

### The model
**NAFNet**, {N_PARAMS / 1e6:.1f}M parameters (`base={ARCH.get('base')}`,
`middle_blocks={ARCH.get('middle_blocks')}`). All computation happens at low resolution
with a single upscale at the very end — 4× cheaper than upscaling first. It starts from
a plain bilinear enlargement and learns only the correction on top. Fully
convolutional, so the same weights serve both competition regimes (128→256 and
256→512) despite only ever training on the smaller one.

Trained on an 8 GB laptop GPU under mixed precision with EMA weight averaging. The loss
combines pixel accuracy (a robust form, so speckle outliers do not dominate),
structural similarity and perceptual similarity — two of those three are competition
metrics.

### Results
| | best classical baseline | **model** | gain |
|---|---|---|---|
| pSNR | {BENCH['b_psnr']:.3f} | **{BENCH['psnr']:.3f}** | **+{BENCH['psnr'] - BENCH['b_psnr']:.3f} dB** |
| SSIM | {BENCH['b_ssim']:.4f} | **{BENCH['ssim']:.4f}** | **+{BENCH['ssim'] - BENCH['b_ssim']:.4f}** ({(BENCH['ssim'] / BENCH['b_ssim'] - 1) * 100:.0f}%) |
| LPIPS | {BENCH['b_lpips']:.4f} | **{BENCH['lpips']:.4f}** | **−{BENCH['b_lpips'] - BENCH['lpips']:.4f}** |

### Five things that did *not* work
Six improvement directions were tested and five measured worse or neutral. Those are
worth as much as the one that worked:

- **Test-time augmentation** — the textbook free win. Measured: it makes the perceptual
  score *worse* (averaging smooths away the sharpness that metric rewards), at 8× the
  inference cost.
- **Larger training crops** — the model has never trained on the 256→512 case, yet
  handles it marginally *better* than the case it did train on. Nothing to fix.
- **Loss weight tuning** — you can move along the trade-off between the three metrics
  but not off it. The original weights were already a sensible point.
- **Rebalancing the training corpus** — improves the corpora fed more, degrades those
  fed less, nets to nothing. Capacity is fixed.
- **Reweighting the downsampling kernels** — would encode a prior the evidence does not
  support.

### An optimisation that reversed
The clock starts at process launch, not at first inference. cuDNN autotuning — normally
free speed — costs **3.5 s of warm-up to save 0.23 ms per image**, so it only pays past
about 15,000 images. Turning it *off* made the pipeline **47% faster**. Reading the
scoring rules changed the answer.
""")

    # --- wiring: damage updates instantly, restoration on demand --------------
    state = gr.State(None)
    ins = [sample, upload, L, sigma, kernel, seed]
    outs = [im_gt, im_lr, hist, info, state]
    for ctl in ins:
        ctl.change(make_degraded, ins, outs)
    demo.load(make_degraded, ins, outs)
    btn.click(restore, state, [im_bic, im_out, metrics])

if __name__ == "__main__":
    demo.launch()
