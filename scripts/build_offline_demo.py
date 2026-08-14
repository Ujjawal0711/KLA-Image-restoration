"""Build a self-contained interactive demo as a single HTML file.

Gradio could not be installed on this machine (PyPI's CDN refuses the TLS handshake for
that package), so the interactive demo is pre-rendered instead: every combination of
sample image and noise level is computed up front, embedded as base64, and switched
client-side with a slider. No server, no dependencies, works offline in any browser.

    python scripts/build_offline_demo.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation import degrade  # noqa: E402
from src.models import build_from_arch  # noqa: E402

L_VALUES = [4, 8, 17, 40, 100]
SIGMA = 0.0086


def png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", (np.clip(img, 0, 1) * 255).round().astype(np.uint8))
    if not ok:
        raise RuntimeError("encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "demo.html")
    ap.add_argument("--max-size", type=int, default=256,
                    help="cap sample size; keeps the HTML small")
    args = ap.parse_args()

    ck = torch.load(ROOT / "checkpoints" / "model.pt", map_location="cpu",
                    weights_only=False)
    model = build_from_arch(ck["arch"])
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    model.eval()
    torch.set_num_threads(4)
    print(f"model {ck['arch']}  "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    samples = ROOT / "space" / "samples"
    index = {}
    for line in (samples / "index.txt").read_text(encoding="utf-8").splitlines():
        if "|" in line:
            fn, label = line.split("|", 1)
            index[fn.strip()] = label.strip()

    data = []
    for fn, label in sorted(index.items()):
        gt = cv2.imread(str(samples / fn), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        gt = gt.astype(np.float32) / 255.0
        if gt.shape[0] > args.max_size:      # keep the file a sensible size
            o = (gt.shape[0] - args.max_size) // 2
            gt = gt[o:o + args.max_size, o:o + args.max_size]
        entry = {"label": label, "gt": png_b64(gt), "levels": []}
        for L in L_VALUES:
            rng = np.random.default_rng(0)
            lr = degrade(gt, {"L": float(L), "sigma": SIGMA,
                              "kernel": "area", "scale": 2}, rng)
            with torch.inference_mode():
                pred = model(torch.from_numpy(
                    np.ascontiguousarray(lr[None, None], dtype=np.float32))
                ).clamp_(0, 1).numpy()[0, 0]
            bic = np.clip(cv2.resize(lr, (gt.shape[1], gt.shape[0]),
                                     interpolation=cv2.INTER_CUBIC), 0, 1)

            def sc(p):
                return (peak_signal_noise_ratio(gt, p.astype(np.float32), data_range=1.0),
                        structural_similarity(gt, p.astype(np.float32), data_range=1.0))

            bp, bs = sc(bic)
            mp, ms = sc(pred)
            entry["levels"].append({
                "L": L,
                "lr": png_b64(lr), "bic": png_b64(bic), "out": png_b64(pred),
                "over": round(float((lr > 1.0).mean()) * 100, 2),
                "max": round(float(lr.max()), 3),
                "bp": round(bp, 2), "bs": round(bs, 4),
                "mp": round(mp, 2), "ms": round(ms, 4),
            })
            print(f"  {label[:34]:<34} L={L:>3}  "
                  f"model {ms:.4f} vs bicubic {bs:.4f}")
        data.append(entry)

    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    args.out.write_text(html, encoding="utf-8")
    print(f"\nwrote {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    print("Open it in any browser — no server, no dependencies.")
    return 0


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Restoring Degraded Semiconductor Imagery — KLA / i4C 2026</title>
<style>
:root{--bg:#0f1216;--card:#161b22;--line:#2a323d;--fg:#e6edf3;--dim:#9aa7b4;
      --accent:#58a6ff;--good:#3fb950;--warn:#e8a33d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:26px;margin:0 0 4px}
.sub{color:var(--dim);margin-bottom:28px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:20px;margin-bottom:20px}
label{display:block;font-size:13px;color:var(--dim);margin-bottom:6px}
select,input[type=range]{width:100%}
select{background:#0d1117;color:var(--fg);border:1px solid var(--line);
       border-radius:6px;padding:8px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:18px}
.panel{text-align:center}
.panel img{width:100%;image-rendering:pixelated;border-radius:6px;
           border:1px solid var(--line);background:#000}
.panel .cap{font-size:12px;color:var(--dim);margin-top:6px}
.panel.hero .cap{color:var(--good);font-weight:600}
table{width:100%;border-collapse:collapse;margin-top:16px;font-size:14px}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:500}
.win{color:var(--good);font-weight:600}
.note{font-size:13px;color:var(--dim);margin-top:14px}
.big{font-size:22px;font-weight:600}
.row{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}.row{grid-template-columns:1fr}}
kbd{background:#0d1117;border:1px solid var(--line);border-radius:4px;padding:1px 5px;
    font-size:12px}
</style></head><body><div class="wrap">

<h1>Restoring degraded semiconductor imagery</h1>
<div class="sub">KLA problem statement · i4C / SEMICON India Hackathon 2026 ·
NAFNet 34.4M parameters</div>

<div class="card">
  <div class="row">
    <div>
      <label>Sample image — all held out from training</label>
      <select id="img"></select>
    </div>
    <div>
      <label>Speckle strength <b id="lval"></b> — drag left for more noise</label>
      <input type="range" id="L" min="0" max="4" step="1" value="2">
    </div>
  </div>

  <div class="grid">
    <div class="panel"><img id="p_gt"><div class="cap">Ground truth</div></div>
    <div class="panel"><img id="p_lr"><div class="cap">Degraded — half size, model input</div></div>
    <div class="panel"><img id="p_bic"><div class="cap">Bicubic upscale</div></div>
    <div class="panel hero"><img id="p_out"><div class="cap">Model output</div></div>
  </div>

  <table>
    <tr><th>metric</th><th>bicubic</th><th>model</th><th>gain</th></tr>
    <tr><td>pSNR (dB)</td><td id="bp"></td><td id="mp"></td><td id="gp" class="win"></td></tr>
    <tr><td>SSIM</td><td id="bs"></td><td id="ms"></td><td id="gs" class="win"></td></tr>
  </table>

  <div class="note" id="over"></div>
</div>

<div class="card">
  <b>Why this noise is unusual</b>
  <div class="note">
    Ordinary image noise is <i>additive</i> — the same amount everywhere. This noise
    <b>multiplies</b>, so it is a percentage error: a pixel at 0.1 barely moves while a
    pixel at 0.9 is thrown far. Multiply the brightest pixel by 1.24 and it lands at
    1.24 — past the top of the scale. That is why nothing in this pipeline clips the
    degraded input: clipping would train the model on images the real test set will not
    contain.
  </div>
</div>

<div class="card">
  <b>Results</b> — measured on 300 held-out images against the strongest non-learned baseline
  <table>
    <tr><th>metric</th><th>baseline</th><th>model</th><th>gain</th></tr>
    <tr><td>pSNR (dB)</td><td>21.517</td><td>25.448</td><td class="win">+3.931</td></tr>
    <tr><td>SSIM</td><td>0.5110</td><td>0.7012</td><td class="win">+0.1902 (37%)</td></tr>
    <tr><td>LPIPS</td><td>0.4341</td><td>0.1672</td><td class="win">−0.2669</td></tr>
  </table>
  <div class="note">No training data was released for this problem — the entire
  training set is synthetic, generated from a degradation model reverse-engineered from
  parameters embedded in the problem statement's own sample figures.</div>
</div>

<script>
const DATA = __DATA__;
const sel = document.getElementById('img'), sl = document.getElementById('L');
DATA.forEach((d,i)=>{const o=document.createElement('option');o.value=i;
  o.textContent=d.label;sel.appendChild(o)});
function render(){
  const d = DATA[+sel.value], lv = d.levels[+sl.value];
  document.getElementById('p_gt').src = d.gt;
  document.getElementById('p_lr').src = lv.lr;
  document.getElementById('p_bic').src = lv.bic;
  document.getElementById('p_out').src = lv.out;
  document.getElementById('lval').textContent = 'L = ' + lv.L;
  bp.textContent = lv.bp.toFixed(2); mp.textContent = lv.mp.toFixed(2);
  bs.textContent = lv.bs.toFixed(4); ms.textContent = lv.ms.toFixed(4);
  gp.textContent = '+' + (lv.mp-lv.bp).toFixed(2) + ' dB';
  gs.textContent = '+' + (lv.ms-lv.bs).toFixed(4);
  document.getElementById('over').innerHTML =
    '<span class="big">' + lv.over.toFixed(2) + '%</span> of degraded pixels exceed 1.0 '
    + '(brightest is ' + lv.max.toFixed(3) + ', where ground truth stops at 1.0). '
    + 'Lower L means stronger speckle.';
}
sel.onchange = sl.oninput = render; render();
</script>
</div></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
