"""Training loop.

    python -m src.train --model unet --epochs 40
    python -m src.train --smoke              # 2 short epochs, verifies the loop end to end

Notes specific to this setup:
  - AMP is on by default. The 4060 has 8 GB and thermal-throttles, so batch size backs
    off automatically on OOM rather than dying an hour in.
  - A checkpoint is written every epoch, plus a separate best-by-SSIM copy. Restarting
    after a throttle stall or a crash must never cost more than one epoch.
  - Validation is deterministic: the val dataset seeds its degradation from the sample
    index, so the same noise is drawn every epoch and metric movement reflects the
    model rather than the draw.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.dataset import build_datasets  # noqa: E402
from src.degradation import DegradationConfig  # noqa: E402
from src.losses import CombinedLoss  # noqa: E402
from src import metrics as M  # noqa: E402


class EMA:
    """Exponential moving average of the weights.

    Standard in image restoration and reliably worth a fraction of a dB: the averaged
    weights sit nearer the centre of the loss basin than any single noisy SGD iterate,
    so they generalise slightly better. Costs one extra copy of the weights and a cheap
    update per step, and nothing at inference.

    Validation runs against the EMA weights, and those are what get checkpointed --
    otherwise the average would be computed and then thrown away.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        # Warm-up: at decay 0.999 the average needs ~1000 steps to escape the random
        # initialisation, so early checkpoints would score near-random and the
        # best-so-far tracker would latch onto noise. Ramping the decay in makes the
        # average track the live weights closely at first and settle to `decay` later.
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)

    def state_for(self, model):
        """EMA weights cast back to the model's own dtypes."""
        ref = model.state_dict()
        return {k: v.to(ref[k].dtype) for k, v in self.shadow.items()}


def make_loader(ds, batch_size: int, workers: int, shuffle: bool = False,
                drop_last: bool = False):
    """Build a DataLoader, degrading to in-process loading if workers cannot start.

    Worker start-up failures on Windows are memory-pressure dependent, so they show up
    intermittently and hours into a run. Falling back to workers=0 is far slower but
    strictly better than losing the run.
    """
    for w in ([workers, 0] if workers > 0 else [0]):
        try:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                num_workers=w, pin_memory=True, drop_last=drop_last,
                                persistent_workers=w > 0)
            iter(loader).__next__() if w > 0 else None
            return loader
        except (OSError, RuntimeError, EOFError) as e:
            if w == 0:
                raise
            print(f"  ! DataLoader with {w} workers failed ({type(e).__name__}: "
                  f"{str(e)[:120]}); falling back to in-process loading")
    raise RuntimeError("unreachable")


def build_model(name: str, **kw):
    if name == "unet":
        from src.models.unet import build
        return build(**kw)
    if name == "nafnet":
        from src.models.nafnet import build
        return build(**kw)
    raise ValueError(f"unknown model {name}")


@torch.no_grad()
def validate(model, loader, device, lpips_fn, max_batches: int | None = None):
    model.eval()
    rows = []
    for i, (lr, hr) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        lr = lr.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(lr)
        pred = pred.float().clamp(0, 1).cpu().numpy()
        gt = hr.numpy()
        preds = [pred[b, 0] for b in range(pred.shape[0])]
        gts = [gt[b, 0] for b in range(gt.shape[0])]
        lp = lpips_fn.batch(preds, gts) if lpips_fn is not None else [float("nan")] * len(preds)
        for p, g, l in zip(preds, gts, lp):
            rows.append({"psnr": M.psnr(p, g), "ssim": M.ssim(p, g), "lpips": l})
    model.train()
    return M.aggregate(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="data/raw/gt")
    ap.add_argument("--model", default="unet", choices=["unet", "nafnet"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hr-size", type=int, default=256)
    ap.add_argument("--iters-per-epoch", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=24)
    # Deliberately low. Windows DataLoader workers use spawn, so each one re-imports
    # torch and loads ~400 MB of cuDNN DLLs. On this box (16 GB total, ~4 GB free)
    # six workers died with "WinError 1114: DLL initialization routine failed".
    # Two workers keep the GPU fed since per-sample CPU work is only a PNG decode
    # plus a Gamma draw.
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--w-ssim", type=float, default=0.2)
    ap.add_argument("--w-lpips", type=float, default=0.1)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--tag", default=None)
    # Degradation ranges are exposed so robustness can be tuned without editing code.
    # The sensitivity sweep measures where the model falls off; these widen it.
    ap.add_argument("--L-min", type=float, default=8.0)
    ap.add_argument("--L-max", type=float, default=40.0)
    ap.add_argument("--sigma-min", type=float, default=5e-4)
    ap.add_argument("--sigma-max", type=float, default=2e-2)
    ap.add_argument("--ema", type=float, default=0.999,
                    help="EMA decay for the weights; 0 disables")
    ap.add_argument("--middle-blocks", type=int, default=None,
                    help="NAFNet middle block count (depth). Default 8.")
    ap.add_argument("--corpus-weights", default=None,
                    help="e.g. 'em=3,bright=2,dtd=1,proc=0.6' — relative sampling "
                         "probability per corpus, independent of file counts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                    help="load weights from a checkpoint but start a FRESH optimiser "
                         "and schedule — for fine-tuning experiments. Unlike --resume, "
                         "which restores the finished schedule and would run at ~0 LR.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.iters_per_epoch, args.val_batches = 2, 20, 2
        args.workers = 0
        # Force a distinct tag: a smoke run defaulting to the model name silently
        # overwrites a real run's checkpoints, which is how run 1's weights were lost.
        if args.tag is None:
            args.tag = f"smoke_{args.model}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or args.model
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Seed everything reproducibly. Deliberately NOT setting cudnn.deterministic: it
    # forces slower fallback kernels for a 5-hour run, and models are A/B-compared on
    # identical inputs afterwards (scripts/compare_models.py), which is what actually
    # needs to be controlled. Seeding removes the init and data-order variance cheaply.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cw = None
    if args.corpus_weights:
        cw = {kv.split("=")[0]: float(kv.split("=")[1])
              for kv in args.corpus_weights.split(",") if "=" in kv}

    cfg = DegradationConfig(L_range=(args.L_min, args.L_max),
                            sigma_range=(args.sigma_min, args.sigma_max))
    train_ds, val_ds = build_datasets(
        args.gt_dir, hr_size=args.hr_size, cfg=cfg,
        train_len=args.iters_per_epoch * args.batch,
        seed=args.seed, corpus_weights=cw,
    )
    if cw:
        print(f"corpus sampling weights: {cw}")
    print(f"GT images: {len(train_ds.paths)} train / {len(val_ds.paths)} val")
    print(f"degradation: L{cfg.L_range} sigma{cfg.sigma_range} kernels{cfg.kernels}")

    batch = args.batch
    mkw = {"base": args.base}
    if args.middle_blocks is not None and args.model == "nafnet":
        mkw["middle_blocks"] = args.middle_blocks
    model = build_model(args.model, **mkw).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={n_par/1e6:.2f}M device={device} "
          f"arch_kw={mkw} ema={args.ema}")

    crit = CombinedLoss(w_ssim=args.w_ssim, w_lpips=args.w_lpips).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                            betas=(0.9, 0.99))
    total_steps = args.epochs * args.iters_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05,
        anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if args.init_from:
        ick = torch.load(args.init_from, map_location=device, weights_only=False)
        # Start from the EMA/shipped weights: they are the best version of that run,
        # and a fine-tune is meant to depart from the best point, not the last one.
        model.load_state_dict({k: v.float() for k, v in ick["model"].items()})
        print(f"initialised weights from {args.init_from} "
              f"(epoch {ick.get('epoch')}, fresh optimiser and schedule)")

    ema = EMA(model, args.ema) if args.ema > 0 else None

    start_epoch, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        # "model" holds the EMA weights (they are what gets validated and shipped);
        # "raw_model" holds the live SGD iterate. Training must continue from the live
        # weights -- resuming from the average would quietly restart optimisation from
        # a different point than where it stopped.
        model.load_state_dict(ck.get("raw_model") or ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema_shadow") is not None:
            ema.shadow = {k: v.to(device).float()
                          for k, v in ck["ema_shadow"].items()}
            ema.step = ck.get("ema_step", 0)
            print(f"  restored EMA average ({ema.step} steps accumulated)")
        elif ema is not None:
            print("  ! checkpoint has no EMA state; the average restarts from here")
        start_epoch, best = ck["epoch"] + 1, ck.get("best", -1.0)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    lpips_fn = None
    try:
        lpips_fn = M.LPIPS()
    except Exception as e:  # noqa: BLE001
        print(f"LPIPS unavailable for validation ({e}); reporting pSNR/SSIM only")

    val_loader = make_loader(val_ds, max(1, batch // 2), args.workers)
    history = []

    train_loader = None
    prev_batch = None
    for epoch in range(start_epoch, args.epochs):
        # Rebuilt only when the batch size changed (after an OOM backoff); respawning
        # workers every epoch is pure overhead on Windows.
        if train_loader is None or batch != prev_batch:
            train_loader = make_loader(train_ds, batch, args.workers,
                                       drop_last=True)
            prev_batch = batch
        model.train()
        t0 = time.time()
        run, seen = {}, 0
        for it, (lr, hr) in enumerate(train_loader):
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            try:
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    pred = model(lr)
                loss, parts = crit(pred.float(), hr)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                if ema is not None:
                    ema.update(model)
            except torch.cuda.OutOfMemoryError:
                # Back off instead of dying: an OOM 20 epochs in is otherwise a total loss.
                torch.cuda.empty_cache()
                if batch > 1:
                    batch = max(1, batch // 2)
                    print(f"\n  OOM -> batch size now {batch}; restarting epoch")
                    break
                raise

            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
            seen += 1
            if (it + 1) % 100 == 0:
                msg = "  ".join(f"{k}={run[k]/seen:.4f}" for k in sorted(run))
                print(f"  e{epoch} it{it+1}/{args.iters_per_epoch}  {msg}  "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

        # Validate (and checkpoint) the EMA weights, swapping them in temporarily.
        raw_state = None
        if ema is not None:
            raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_for(model))
        val = validate(model, val_loader, device, lpips_fn, args.val_batches)
        eval_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        if raw_state is not None:
            model.load_state_dict(raw_state)   # training continues from live weights
        dt = time.time() - t0
        print(f"[epoch {epoch}] {dt:.0f}s  "
              f"pSNR={val['psnr']:.3f}  SSIM={val['ssim']:.4f}  LPIPS={val['lpips']:.4f}"
              f"{'  (ema)' if ema is not None else ''}")

        history.append({"epoch": epoch, "seconds": dt, **val,
                        **{f"train_{k}": run[k] / max(seen, 1) for k in run}})
        (out / f"{tag}_history.json").write_text(json.dumps(history, indent=2))

        arch = {"name": args.model, "base": args.base}
        if args.middle_blocks is not None and args.model == "nafnet":
            arch["middle_blocks"] = args.middle_blocks
        # `model` holds the EMA weights that were actually validated, so that is what
        # ships; the live weights are kept separately so a resume continues correctly.
        ck = {"model": eval_state, "raw_model": model.state_dict(),
              "opt": opt.state_dict(),
              "sched": sched.state_dict(), "scaler": scaler.state_dict(),
              "epoch": epoch, "best": best, "args": vars(args), "arch": arch}
        if ema is not None:
            # Saved so a resumed run continues the same running average rather than
            # starting a fresh one from wherever it happened to stop.
            ck["ema_shadow"] = {k: v.cpu() for k, v in ema.shadow.items()}
            ck["ema_step"] = ema.step
        torch.save(ck, out / f"{tag}_last.pt")
        if val["ssim"] > best:
            best = val["ssim"]
            ck["best"] = best
            torch.save(ck, out / f"{tag}_best.pt")
            print(f"  new best SSIM {best:.4f} -> {tag}_best.pt")

    print(f"\ndone. best SSIM {best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
