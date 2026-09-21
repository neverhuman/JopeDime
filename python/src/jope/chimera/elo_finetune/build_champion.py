"""Materialize the evolution champion as a checkpoint eval_full can sweep.

champion = base EMA weights + Σ(all promoted sparse deltas). Sparse deltas
compose linearly and index_add is order-independent (unique indices per
tensor), so applying them in promo order reproduces the fleet's champion
bit-for-bit (verified against the driver's full_checksum elsewhere).

Usage:
  python -m python.elo_finetune.build_champion \
      --base ~/fable_god_run/ckpt/step_1500.ckpt --which ema_model \
      --champion-dir ~/fable_god_run/evo/exchange/champion \
      --out ~/fable_god_run/ckpt/evo_champion.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .ckpt import CheckpointManager


def apply_promos(state: dict, champion_dir: Path) -> int:
    promos = sorted(champion_dir.glob("promo_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    for npz in promos:
        data = np.load(npz)
        for name in data["tensors"]:
            name = str(name)
            flat = state[name].view(-1)
            idx = torch.from_numpy(data[f"{name}::idx"]).long()
            val = torch.from_numpy(data[f"{name}::val"]).to(flat.dtype)
            flat.index_add_(0, idx, val)
    return len(promos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--which", default="ema_model")
    ap.add_argument("--champion-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = CheckpointManager.load(Path(args.base).expanduser(), map_location="cpu")
    state = {k: v.clone() for k, v in payload[args.which].items()}
    n = apply_promos(state, Path(args.champion_dir).expanduser())
    out = Path(args.out).expanduser()
    torch.save({"ema_model": state, "step": payload.get("step", 0), "n_promos": n}, out)
    print(f"champion: base {args.which} + {n} promo delta(s) -> {out}")


if __name__ == "__main__":
    main()
