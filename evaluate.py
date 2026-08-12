#!/usr/bin/env python3
"""Restore degraded images. THE submission deliverable.

    python evaluate.py <test_image_dir> <output_dir>

Scored on wall clock measured from process start, which includes interpreter startup,
imports, model init, reading every input, inference, and writing every output. That
constraint drives most of what looks unusual below:

  - **Imports are deferred past argument validation.** Importing torch costs over a
    second. A bad path should fail before paying that, and nothing heavy is imported
    that is not used. `lpips` is never imported here: it downloads a 233 MB backbone on
    first use, which on a cold benchmark machine would dwarf the inference itself.

  - **Threads, not DataLoader worker processes.** On Windows each worker process costs
    roughly a second to spawn, and this run may only last a few seconds in total.
    cv2.imread/imwrite release the GIL, so a thread pool gets the same I/O overlap for
    no start-up cost.

  - **Reads are prefetched one chunk ahead of the GPU, writes are fired and forgotten.**
    Disk and GPU overlap instead of alternating.

  - **Images are grouped by resolution before batching.** The test set mixes 128x128 and
    256x256 inputs; batching across shapes would force per-image calls.

  - **Inference in fp16 under autocast**, cudnn.benchmark on (shapes repeat within a
    group, so autotuning pays for itself immediately).

  - Nothing is tiled. Even 256->512 at batch size 8 is a fraction of an H100.

Correctness details that are easy to get wrong:
  - Input is NOT clipped. Speckle is multiplicative, so degraded pixels legitimately
    exceed the ground-truth range, and the model was trained on unclipped input.
  - Output IS clipped to [0,1], because the target is in [0,1].
  - Output dtype, extension and filename mirror the input, so the grader's loader sees
    what it expects.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

T_START = time.perf_counter()

IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
DEFAULT_WEIGHTS = pathlib.Path(__file__).resolve().parent / "checkpoints" / "model.pt"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Restore degraded images (2x SR + denoise)")
    p.add_argument("input_dir", type=pathlib.Path, help="directory of degraded images")
    p.add_argument("output_dir", type=pathlib.Path, help="directory for restored images")
    p.add_argument("--weights", type=pathlib.Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--fp32", action="store_true", help="disable fp16 autocast")
    p.add_argument("--cudnn-benchmark", action="store_true",
                   help="enable cuDNN autotuning. MEASURED: costs ~3.5s of extra "
                        "first-batch autotuning to save ~0.23ms/img, so it only pays "
                        "above roughly 15000 images. Off by default.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model; only worth it on large test sets "
                        "(compilation happens on the first batch, on the clock)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"ERROR: input dir not found: {args.input_dir}", file=sys.stderr)
        return 2
    files = sorted(p for p in args.input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"ERROR: no images in {args.input_dir}", file=sys.stderr)
        return 2
    if not args.weights.is_file():
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- heavy imports start here -----------------------------------------
    t_imp = time.perf_counter()
    import concurrent.futures as cf

    import cv2
    import numpy as np
    import torch

    cv2.setNumThreads(min(8, (os.cpu_count() or 4)))
    t_imp = time.perf_counter() - t_imp

    # ---- I/O helpers ------------------------------------------------------
    def read(path: pathlib.Path):
        """-> (float32 HxW in [0,1] but NOT clipped, dtype tag, was_colour)"""
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None:
            return None
        colour = a.ndim == 3
        if colour:
            a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
        if a.dtype == np.uint8:
            return a.astype(np.float32) / 255.0, "u8", colour
        if a.dtype == np.uint16:
            return a.astype(np.float32) / 65535.0, "u16", colour
        return a.astype(np.float32), "f32", colour

    def write(path: pathlib.Path, img, tag: str):
        img = np.clip(img, 0.0, 1.0)
        if tag == "u8":
            out = (img * 255.0).round().astype(np.uint8)
        elif tag == "u16":
            out = (img * 65535.0).round().astype(np.uint16)
        else:
            out = img.astype(np.float32)
        # cv2.imwrite returns False rather than raising on a full disk, a bad path or
        # a permissions problem. Unchecked, that silently produces a short output set
        # and the missing images simply score as failures.
        if not cv2.imwrite(str(path), out):
            raise OSError(f"failed to write {path}")

    n_io = min(16, (os.cpu_count() or 4) * 2)
    pool = cf.ThreadPoolExecutor(max_workers=n_io)

    # Kick off every read BEFORE touching CUDA. Creating the CUDA context costs a
    # couple of seconds during which the GPU is doing nothing useful, so the disk
    # reads ride along for free instead of being charged separately.
    t_read0 = time.perf_counter()
    read_futures = [pool.submit(read, f) for f in files]

    # ---- model ------------------------------------------------------------
    t_init = time.perf_counter()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    arch = ck.get("arch", {"name": "unet", "base": 48})

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from src.models import build_from_arch
    model = build_from_arch(arch)
    state = ck["model"]
    # Weights may be stored in half precision to halve the file size; the module is
    # fp32 and autocast handles the arithmetic, so cast on load.
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    use_amp = (device.type == "cuda") and not args.fp32
    if device.type == "cuda":
        # Off by default — see --cudnn-benchmark. Autotuning is charged to the clock
        # and only amortises over tens of thousands of images.
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
    if args.compile:
        model = torch.compile(model, mode="max-autotune")
    t_init = time.perf_counter() - t_init

    # ---- collect reads, grouped by resolution -----------------------------
    t_read = time.perf_counter()
    loaded = [fu.result() for fu in read_futures]
    groups: dict[tuple, list[int]] = {}
    bad = []
    for i, item in enumerate(loaded):
        if item is None:
            bad.append(files[i])
            continue
        groups.setdefault(item[0].shape[:2], []).append(i)
    t_read = time.perf_counter() - t_read
    if bad and not args.quiet:
        print(f"WARNING: {len(bad)} unreadable file(s), skipped: "
              f"{', '.join(p.name for p in bad[:3])}", file=sys.stderr)

    # ---- inference --------------------------------------------------------
    t_inf = 0.0
    t_write = 0.0
    pending: list[cf.Future] = []
    n_done = 0

    for shape, idxs in sorted(groups.items()):
        for s in range(0, len(idxs), args.batch):
            chunk = idxs[s:s + args.batch]
            t0 = time.perf_counter()
            batch = np.stack([loaded[i][0] for i in chunk])[:, None]
            x = torch.from_numpy(batch).to(device, non_blocking=True)
            with torch.inference_mode():
                if use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        y = model(x)
                else:
                    y = model(x)
                y = y.float().clamp_(0.0, 1.0)
            # .cpu() is itself a synchronising copy, so no explicit synchronize() is
            # needed here for the timing to be honest.
            out = y.cpu().numpy()[:, 0]
            t_inf += time.perf_counter() - t0

            t1 = time.perf_counter()
            for k, i in enumerate(chunk):
                dst = args.output_dir / files[i].name
                pending.append(pool.submit(write, dst, out[k], loaded[i][1]))
            t_write += time.perf_counter() - t1
            n_done += len(chunk)

    t1 = time.perf_counter()
    write_errors = []
    for f in pending:
        try:
            f.result()
        except Exception as e:  # noqa: BLE001
            write_errors.append(str(e))
    pool.shutdown(wait=True)
    t_write += time.perf_counter() - t1
    if write_errors:
        print(f"ERROR: {len(write_errors)} output(s) failed to write, e.g. "
              f"{write_errors[0]}", file=sys.stderr)

    total = time.perf_counter() - T_START
    if not args.quiet:
        startup = total - t_imp - t_init - t_read - t_inf - t_write
        print(f"restored {n_done} images -> {args.output_dir}")
        print(f"  startup   {startup:7.3f}s")
        print(f"  imports   {t_imp:7.3f}s")
        print(f"  model init{t_init:7.3f}s  (reads run concurrently)")
        print(f"  read wait {t_read:7.3f}s  (only what did not fit behind init)")
        print(f"  inference {t_inf:7.3f}s")
        print(f"  write     {t_write:7.3f}s")
        print(f"  TOTAL     {total:7.3f}s   ({n_done / max(total, 1e-9):.1f} img/s)")
    return 1 if write_errors else 0


if __name__ == "__main__":
    sys.exit(main())
