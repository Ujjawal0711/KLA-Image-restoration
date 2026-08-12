"""Live local demo — a real web app, no extra dependencies.

    python demo_app.py          then open http://localhost:7860

Gradio could not be installed on this machine (PyPI's CDN refuses the TLS handshake for
that package), so this serves the same experience from the standard library alone:
`http.server` for the backend, plain HTML/JS for the frontend, and the actual model run
on demand. Nothing is pre-computed — every restore is a real forward pass, and the page
reports how long it took.

The split matters for the demo: corrupting an image is pure NumPy and returns in
milliseconds, so the damage updates as you drag. Restoring runs a 34M-parameter network,
about 0.7 s for a 128x128 input on CPU. Damage is cheap, undoing it is not.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.degradation import KERNELS, degrade  # noqa: E402
from src.models import build_from_arch  # noqa: E402

MODEL = None
ARCH: dict = {}
N_PARAMS = 0
SAMPLES: dict[str, str] = {}
_LOCK = threading.Lock()          # one model, many browser requests


def load_model(weights: pathlib.Path):
    global MODEL, ARCH, N_PARAMS
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    ARCH = ck.get("arch", {"name": "unet", "base": 48})
    MODEL = build_from_arch(ARCH)
    MODEL.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    MODEL.eval()
    N_PARAMS = sum(p.numel() for p in MODEL.parameters())


def load_samples():
    idx = ROOT / "space" / "samples" / "index.txt"
    if not idx.exists():
        return
    for line in idx.read_text(encoding="utf-8").splitlines():
        if "|" in line:
            fn, label = line.split("|", 1)
            SAMPLES[label.strip()] = fn.strip()


def to_png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", (np.clip(img, 0, 1) * 255).round().astype(np.uint8))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def read_gt(sample: str | None, upload_b64: str | None) -> np.ndarray:
    if upload_b64:
        raw = base64.b64decode(upload_b64.split(",", 1)[-1])
        a = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if a is None:
            raise ValueError("could not decode that image")
        a = a.astype(np.float32) / 255.0
        if min(a.shape[:2]) > 512:        # keep CPU inference responsive
            y, x = (a.shape[0] - 512) // 2, (a.shape[1] - 512) // 2
            a = a[y:y + 512, x:x + 512]
        # Even dimensions only — the task is an exact 2x, so an odd size has no half.
        h, w = a.shape[:2]
        return np.clip(a[:h - h % 2, :w - w % 2], 0, 1)
    fn = SAMPLES.get(sample or "") or next(iter(SAMPLES.values()))
    a = cv2.imread(str(ROOT / "space" / "samples" / fn), cv2.IMREAD_GRAYSCALE)
    return a.astype(np.float32) / 255.0


def api_degrade(p: dict) -> dict:
    gt = read_gt(p.get("sample"), p.get("upload"))
    rng = np.random.default_rng(int(p.get("seed", 0)))
    t0 = time.perf_counter()
    lr = degrade(gt, {"L": float(p["L"]), "sigma": float(p["sigma"]),
                      "kernel": p.get("kernel", "area"), "scale": 2}, rng)
    ms = (time.perf_counter() - t0) * 1000
    return {"gt": to_png_b64(gt), "lr": to_png_b64(lr),
            "in_w": int(lr.shape[1]), "in_h": int(lr.shape[0]),
            "out_w": int(gt.shape[1]), "out_h": int(gt.shape[0]),
            "over": round(float((lr > 1.0).mean()) * 100, 2),
            "max": round(float(lr.max()), 3), "min": round(float(lr.min()), 3),
            "ms": round(ms, 1)}


def api_restore(p: dict) -> dict:
    gt = read_gt(p.get("sample"), p.get("upload"))
    rng = np.random.default_rng(int(p.get("seed", 0)))
    lr = degrade(gt, {"L": float(p["L"]), "sigma": float(p["sigma"]),
                      "kernel": p.get("kernel", "area"), "scale": 2}, rng)

    x = torch.from_numpy(np.ascontiguousarray(lr[None, None], dtype=np.float32))
    with _LOCK:                       # the model is not re-entrant
        t0 = time.perf_counter()
        with torch.inference_mode():
            pred = MODEL(x).clamp_(0, 1).numpy()[0, 0]
        infer_ms = (time.perf_counter() - t0) * 1000

    bic = np.clip(cv2.resize(lr, (gt.shape[1], gt.shape[0]),
                             interpolation=cv2.INTER_CUBIC), 0, 1).astype(np.float32)

    def sc(a):
        return (round(float(peak_signal_noise_ratio(gt, a, data_range=1.0)), 2),
                round(float(structural_similarity(gt, a, data_range=1.0)), 4))

    bp, bs = sc(bic)
    mp, ms_ = sc(pred.astype(np.float32))
    return {"bic": to_png_b64(bic), "out": to_png_b64(pred),
            "bp": bp, "bs": bs, "mp": mp, "ms": ms_,
            "infer_ms": round(infer_ms, 1),
            "px": int(gt.shape[0] * gt.shape[1])}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # keep the console readable during a demo
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = PAGE.replace("__SAMPLES__", json.dumps(list(SAMPLES)))
            page = page.replace("__KERNELS__", json.dumps(list(KERNELS)))
            page = page.replace("__ARCH__", json.dumps(
                {"name": ARCH.get("name"), "base": ARCH.get("base"),
                 "middle_blocks": ARCH.get("middle_blocks"),
                 "params_m": round(N_PARAMS / 1e6, 1)}))
            self._send(200, page, "text/html; charset=utf-8")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/degrade":
                self._send(200, json.dumps(api_degrade(p)))
            elif self.path == "/api/restore":
                self._send(200, json.dumps(api_restore(p)))
            else:
                self._send(404, json.dumps({"error": "no such endpoint"}))
        except Exception as e:  # noqa: BLE001
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Restoring Degraded Semiconductor Imagery — live demo</title><style>
:root{--bg:#0f1216;--card:#161b22;--line:#2a323d;--fg:#e6edf3;--dim:#9aa7b4;
      --accent:#58a6ff;--good:#3fb950;--warn:#e8a33d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:26px 20px 60px}
h1{font-size:24px;margin:0 0 2px}.sub{color:var(--dim);margin-bottom:22px;font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:18px;margin-bottom:16px}
.cols{display:grid;grid-template-columns:300px 1fr;gap:20px}
label{display:block;font-size:12px;color:var(--dim);margin:12px 0 5px}
select,input[type=file]{width:100%;background:#0d1117;color:var(--fg);
  border:1px solid var(--line);border-radius:6px;padding:7px}
input[type=range]{width:100%}
button{width:100%;padding:11px;border:0;border-radius:7px;background:var(--accent);
  color:#04121f;font-weight:700;font-size:15px;cursor:pointer;margin-top:14px}
button:disabled{opacity:.55;cursor:wait}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.panel{text-align:center}
.panel img{width:100%;image-rendering:pixelated;border-radius:6px;
  border:1px solid var(--line);background:#000;min-height:120px}
.cap{font-size:12px;color:var(--dim);margin-top:5px}
.panel.hero .cap{color:var(--good);font-weight:600}
table{width:100%;border-collapse:collapse;margin-top:12px;font-size:14px}
th,td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}th{color:var(--dim);font-weight:500}
.win{color:var(--good);font-weight:600}
.note{font-size:13px;color:var(--dim);margin-top:10px}
.big{font-size:20px;font-weight:700;color:var(--warn)}
.val{float:right;color:var(--fg);font-weight:600}
.badge{display:inline-block;background:#0d1117;border:1px solid var(--line);
  border-radius:5px;padding:2px 7px;font-size:12px;color:var(--dim);margin-right:6px}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>Restoring degraded semiconductor imagery</h1>
<div class="sub">KLA problem statement · i4C / SEMICON India Hackathon 2026 ·
<span id="arch"></span> · <b>live inference — nothing pre-computed</b></div>

<div class="cols">
 <div class="card">
  <label>Sample image <span class="val" style="font-weight:400;font-size:11px">held out from training</span></label>
  <select id="sample"></select>
  <label>…or upload your own</label>
  <input type="file" id="file" accept="image/*">
  <label>Speckle strength <span class="val" id="Lv">17</span></label>
  <input type="range" id="L" min="4" max="100" step="0.5" value="17">
  <label>Additive noise σ <span class="val" id="Sv">0.0086</span></label>
  <input type="range" id="S" min="0" max="0.05" step="0.0002" value="0.0086">
  <label>Downsampling kernel</label>
  <select id="kernel"></select>
  <label>Noise seed <span class="val" id="Dv">0</span></label>
  <input type="range" id="D" min="0" max="20" step="1" value="0">
  <button id="go">Restore</button>
  <div class="note" id="timing"></div>
 </div>

 <div>
  <div class="card">
   <div class="grid">
    <div class="panel"><img id="p_gt"><div class="cap">Ground truth</div></div>
    <div class="panel"><img id="p_lr"><div class="cap" id="cap_lr">Degraded — model input</div></div>
    <div class="panel"><img id="p_bic"><div class="cap">Bicubic upscale</div></div>
    <div class="panel hero"><img id="p_out"><div class="cap">Model output</div></div>
   </div>
   <table id="tbl" style="display:none">
    <tr><th>metric</th><th>bicubic</th><th>model</th><th>gain</th></tr>
    <tr><td>pSNR (dB)</td><td id="bp"></td><td id="mp"></td><td id="gp" class="win"></td></tr>
    <tr><td>SSIM</td><td id="bs"></td><td id="ms"></td><td id="gs" class="win"></td></tr>
   </table>
   <div class="note" id="over"></div>
  </div>
  <div class="card note">
   <b style="color:var(--fg)">Why it spills past white.</b> Ordinary noise is
   <i>additive</i> — the same everywhere. This noise <b>multiplies</b>, so it is a
   percentage error: a pixel at 0.1 barely moves while a pixel at 0.9 is thrown far.
   Multiply the brightest pixel by 1.24 and it lands at 1.24, past the top of the scale.
   That is why nothing here clips the degraded input — clipping would train the model on
   images the real test set will not contain.
  </div>
 </div>
</div>

<script>
const S=__SAMPLES__, K=__KERNELS__, A=__ARCH__;
const $=id=>document.getElementById(id);
$('arch').textContent = A.name.toUpperCase()+' '+A.params_m+'M params';
S.forEach(s=>{const o=document.createElement('option');o.textContent=s;$('sample').appendChild(o)});
K.forEach(k=>{const o=document.createElement('option');o.textContent=k;$('kernel').appendChild(o)});
let upload=null;
$('file').onchange=e=>{const f=e.target.files[0];if(!f)return;
  const r=new FileReader();r.onload=()=>{upload=r.result;deg()};r.readAsDataURL(f)};
function params(){return {sample:$('sample').value,upload:upload,
  L:+$('L').value,sigma:+$('S').value,kernel:$('kernel').value,seed:+$('D').value}}
async function post(p,b){const r=await fetch(p,{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json(); if(j.error) throw new Error(j.error); return j}
let t=null;
async function deg(){
  $('Lv').textContent=(+$('L').value).toFixed(1);
  $('Sv').textContent=(+$('S').value).toFixed(4);
  $('Dv').textContent=$('D').value;
  try{
    const d=await post('/api/degrade',params());
    $('p_gt').src=d.gt; $('p_lr').src=d.lr;
    $('cap_lr').textContent='Degraded — model input ('+d.in_w+'×'+d.in_h+')';
    $('p_bic').src=''; $('p_out').src=''; $('tbl').style.display='none';
    $('over').innerHTML='<span class="big">'+d.over.toFixed(2)+'%</span> of degraded '
      +'pixels exceed 1.0 &nbsp;·&nbsp; range ['+d.min.toFixed(3)+', '+d.max.toFixed(3)
      +'] against a ground truth of [0, 1] &nbsp;·&nbsp; damage applied in '+d.ms+' ms';
    $('timing').innerHTML='<span class="badge">'+d.in_w+'×'+d.in_h+' → '+d.out_w+'×'+d.out_h+'</span>';
  }catch(e){$('over').textContent='Error: '+e.message}
}
function debounce(){clearTimeout(t);t=setTimeout(deg,90)}
['L','S','D'].forEach(i=>$(i).oninput=debounce);
['sample','kernel'].forEach(i=>$(i).onchange=()=>{if(i==='sample')upload=null;
  $('file').value='';deg()});
$('go').onclick=async()=>{
  const b=$('go'); b.disabled=true; b.textContent='Running the network…';
  try{
    const d=await post('/api/restore',params());
    $('p_bic').src=d.bic; $('p_out').src=d.out;
    $('bp').textContent=d.bp.toFixed(2); $('mp').textContent=d.mp.toFixed(2);
    $('bs').textContent=d.bs.toFixed(4); $('ms').textContent=d.ms.toFixed(4);
    $('gp').textContent='+'+(d.mp-d.bp).toFixed(2)+' dB';
    $('gs').textContent='+'+(d.ms-d.bs).toFixed(4);
    $('tbl').style.display='';
    $('timing').innerHTML='<span class="badge">forward pass '+d.infer_ms+' ms</span>'
      +'<span class="badge">'+(d.px/1000).toFixed(0)+'k pixels out</span>'
      +'<span class="badge">CPU</span>';
  }catch(e){$('over').textContent='Error: '+e.message}
  b.disabled=false; b.textContent='Restore';
};
deg();
</script></div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=pathlib.Path,
                    default=ROOT / "checkpoints" / "model.pt")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"ERROR: weights not found at {args.weights}")
        return 1

    torch.set_num_threads(args.threads)
    print(f"loading {args.weights.name} ...")
    load_model(args.weights)
    load_samples()
    print(f"  {ARCH}  {N_PARAMS/1e6:.1f}M params")
    print(f"  {len(SAMPLES)} sample images")

    url = f"http://localhost:{args.port}"
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  demo running at  {url}\n  Ctrl+C to stop\n")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
