"""Dose-response: champion = base EMA + c*delta from one promo npz.

Sweeping several c and regressing archive Elo on c tests whether a promoted
delta carries real directional signal (monotone in c, consistent across
val/test) or is noise (random scatter). Reuses eval_full for the sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .ckpt import CheckpointManager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--which", default="ema_model")
    ap.add_argument("--delta", required=True, help="promo_XXXX.npz")
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = CheckpointManager.load(Path(args.base).expanduser(),
                                     map_location="cpu")
    state = {k: v.clone() for k, v in payload[args.which].items()}
    data = np.load(Path(args.delta).expanduser())
    for name in data["tensors"]:
        name = str(name)
        flat = state[name].view(-1)
        idx = torch.from_numpy(data[f"{name}::idx"]).long()
        val = torch.from_numpy(data[f"{name}::val"]).to(flat.dtype) * args.scale
        flat.index_add_(0, idx, val)
    out = Path(args.out).expanduser()
    torch.save({"ema_model": state, "step": payload.get("step", 0),
                "scale": args.scale}, out)
    print(f"scaled champion: base + {args.scale}*delta -> {out}")


if __name__ == "__main__":
    main()
