"""Per-node pre-flight: pack mmap + strict-load + one seeded GPU forward.

Prints a prediction checksum; across nodes these must agree to ~1e-3 rel
(different GPU kernels, bf16). Run on every node before training.

Usage: python3 -m python.elo_finetune.load_check --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .eval_protocol import ProtocolRunner
from .model_io import load_regressor
from .pack import DEFAULT_PACK_DIR, FablePack, bucket_schedule, collate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed-a", type=int, default=1001)
    args = parser.parse_args()

    pack = FablePack(args.pack_dir / "fable_real_v1.pack")
    sha = (args.pack_dir / "fable_real_v1.pack.sha256").read_text().split()[0]
    model = load_regressor(device=args.device)
    runner = ProtocolRunner(model, torch.device(args.device))
    plans = bucket_schedule(pack, "val", run_seed=args.seed_a, epoch=0, micro_size=8)
    batch = collate(pack, plans[0], seed_a=args.seed_a)
    out = runner.tier1(batch, with_grad=False)
    utils = out.utilities_f64()
    print(
        json.dumps(
            {
                "pack_sha256": sha[:16],
                "datasets": len(pack),
                "params": sum(p.numel() for p in model.parameters()),
                "probe_utility_sum": round(float(np.sum(utils)), 8),
                "probe_finite": bool(np.isfinite(utils).all()),
                "device": args.device,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    )


if __name__ == "__main__":
    main()
