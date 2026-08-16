# Gallery — images to try the restorer on

18 clean ground-truth images, 512×512, 8-bit greyscale. Upload any of them to the demo
(local `demo_app.py` or the [live Space](https://huggingface.co/spaces/Ujjawal0711/kla-image-restoration)),
set the damage with the sliders, and press **Restore**.

The demo treats an uploaded image as the *ground truth*: it degrades it for you — halves
the resolution and applies speckle — then restores it, so you can compare against the
original. A 512×512 upload becomes a 256×256 model input and a 512×512 output.

## What's here, and what each one is for

| file | corpus | in training? | why demo with it |
|---|---|---|---|
| `brightfield_1-3` | brightfield microscopy | **never** | **The strongest case.** Bicubic manages 0.1075 SSIM here; the model reaches 0.6662 — a **+0.5587** margin, the largest of any corpus, on imagery it has never seen |
| `fluorescence_1-3` | fluorescence microscopy | **never** | Second unseen modality. High contrast, so the before/after is easy to see on video |
| `electron-microscopy_1-3` | EM | held out | Closest public analogue to real wafer inspection — the most *relevant* imagery here |
| `materials-micrograph_1-3` | SEM/TEM micrographs | held out | Real materials-science imagery from publication figures |
| `wafer-procedural_1-3` | procedural | held out | Grain boundaries and die lattices, matching the structures in the problem statement's own samples. Highest contrast in the set |
| `textures_1-3` | describable textures | held out | Generic structure — the diversity ballast in the training corpus |

**"Held out" means held out.** The four trained corpora draw only from the deterministic
validation split (`val_frac=0.05`, `seed=0`) — the same split the published metrics use.
The model has not seen any image in this folder.

## Settings worth trying

| goal | L | σ | note |
|---|---|---|---|
| The deck's own conditions | **16.86** | **0.0086** | The actual parameters recovered from the problem statement's figures |
| Deck's second sample | **18.13** | **0.0011** | Much less additive noise |
| Show it holding up | **4** | 0.02 | Far below the trained floor of L=8. Degrades, does not collapse |
| Show the overflow | **4** | 0 | Roughly a third of pixels exceed 1.0 — speckle multiplies, so it scales with brightness |
| Easy case | 40+ | 0.001 | Top of the trained range |

Lower `L` means *noisier*. The model trained on `L` 8–40 and `σ` 5e-4 to 2e-2, so
anything outside that is genuinely untrained territory.

## Note on brightfield

Brightfield images look washed out — low contrast, mean intensity around 210. That is
what the modality actually looks like, and it is precisely why bicubic fails so badly on
it. The gain is real and it is the largest in the project, but it reads better in the
metrics panel than it does at a glance, so quote the numbers when demoing it.
