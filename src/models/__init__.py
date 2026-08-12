"""Model construction from a checkpoint's stored architecture description.

Every consumer (evaluate.py, promote_checkpoint, the sweeps, the comparison script)
needs to rebuild the exact network a checkpoint was trained with. Doing that inline in
each of them meant a new architecture argument had to be threaded through six places,
and missing one produces a confusing state_dict size mismatch rather than a clear error.
"""

from __future__ import annotations


def build_from_arch(arch: dict | None):
    """arch: {"name": "nafnet"|"unet", "base": int, ...extra kwargs}"""
    arch = dict(arch or {"name": "unet", "base": 48})
    name = arch.pop("name", "unet")
    if name == "nafnet":
        from .nafnet import build
    elif name == "unet":
        from .unet import build
    else:
        raise ValueError(f"unknown architecture {name!r}")
    return build(**arch)
