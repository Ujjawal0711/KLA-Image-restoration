"""Phase 1 — verify the submission is self-contained.

Traces what evaluate.py actually imports and touches, then checks every third-party
module appears in requirements.txt and every local file it needs exists. Catches the
classic failure where the script works on the dev machine because of a file that was
never going to be in the submission.

    python scripts/check_submission.py
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STDLIB = set(sys.stdlib_module_names)

# Files the grader must receive for evaluate.py to run.
REQUIRED = [
    "evaluate.py",
    "requirements.txt",
    "src/__init__.py",
    "src/models/__init__.py",
    "src/models/nafnet.py",
    "src/models/unet.py",
    "checkpoints/model.pt",
]


def imports_of(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
    return mods


def main() -> int:
    problems = []

    print("=== required files ===")
    for rel in REQUIRED:
        p = ROOT / rel
        ok = p.exists()
        size = f"{p.stat().st_size/1e6:.1f} MB" if ok else "-"
        print(f"  {'OK ' if ok else 'MISSING'}  {rel:<28} {size:>10}")
        if not ok:
            problems.append(f"missing file: {rel}")

    print("\n=== third-party imports reachable from evaluate.py ===")
    seen: set[pathlib.Path] = set()
    queue = [ROOT / "evaluate.py"]
    third: set[str] = set()
    while queue:
        f = queue.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        for m in imports_of(f):
            if m in STDLIB or m == "src":
                continue
            third.add(m)
        # follow local src.* imports
        if f.name == "evaluate.py":
            for cand in ["src/models/nafnet.py", "src/models/unet.py"]:
                queue.append(ROOT / cand)

    req_text = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    dist_for = {"cv2": "opencv-python", "PIL": "pillow", "skimage": "scikit-image",
                "yaml": "pyyaml", "sklearn": "scikit-learn"}
    for m in sorted(third):
        dist = dist_for.get(m, m)
        ok = dist.lower() in req_text
        print(f"  {'OK ' if ok else 'NOT IN REQS'}  {m:<14} -> {dist}")
        if not ok:
            problems.append(f"{m} not in requirements.txt")

    heavy = {"lpips", "matplotlib", "scipy", "skimage", "pandas", "tqdm"}
    bad = heavy & third
    if bad:
        problems.append(f"evaluate.py imports heavy unused deps: {sorted(bad)}")
        print(f"\n  ! evaluate.py pulls in {sorted(bad)} — these cost start-up time")
    else:
        print("\n  evaluate.py imports no heavy extras (lpips/matplotlib/scipy absent) OK")

    print("\n=== checkpoint self-description ===")
    ck_path = ROOT / "checkpoints" / "model.pt"
    if ck_path.exists():
        import torch
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        arch = ck.get("arch")
        print(f"  arch      : {arch}")
        print(f"  epoch     : {ck.get('epoch')}")
        print(f"  best SSIM : {ck.get('best')}")
        if not arch:
            problems.append("checkpoint has no 'arch' key; evaluate.py would guess")
        n = sum(v.numel() for v in ck["model"].values())
        print(f"  params    : {n/1e6:.2f}M")

    print("\n" + ("ALL CHECKS PASS" if not problems else "PROBLEMS:"))
    for p in problems:
        print(f"  - {p}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
