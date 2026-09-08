"""Frozen-base proxy utilities (the paired-ΔElo control variate) and the
optional fp16 KL logit cache.

u_base[d] = tier-1 soft utility of the FROZEN deployed weights on dataset d,
under the identical seed-A protocol the trainer uses. Regenerate whenever
seed A rotates. The KL cache stores the frozen model's bucket logits on each
train dataset's seed-A query rows (fp16), for the KL(p_θ ‖ p_base) anchor.

Usage: python3 -m python.elo_finetune.proxy_utilities --device cuda:0 \
           [--seed-a 1001] [--micro 16] [--kl-cache] [--pack-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .eval_protocol import ProtocolRunner
from .model_io import load_regressor
from .pack import DEFAULT_PACK_DIR, FablePack, SPLIT_NAMES, bucket_schedule, collate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed-a", type=int, default=1001)
    parser.add_argument("--micro", type=int, default=16)
    parser.add_argument("--kl-cache", action="store_true")
    args = parser.parse_args()

    pack = FablePack(args.pack_dir / "fable_real_v1.pack", preload=True)
    device = torch.device(args.device)
    model = load_regressor(device=device)
    runner = ProtocolRunner(model, device)

    u_base = np.full(len(pack), np.nan, dtype=np.float64)
    kl_logits: dict[int, np.ndarray] = {}
    started = time.time()
    done = 0
    for split in SPLIT_NAMES:
        plans = bucket_schedule(pack, split, run_seed=args.seed_a, epoch=0,
                                micro_size=args.micro)
        for plan in plans:
            batch = collate(pack, plan, seed_a=args.seed_a)
            out = runner.tier1(batch, with_grad=False)
            u_base[plan.dataset_indices] = out.utilities_f64()
            if args.kl_cache and split == "train":
                logits = out.logits.detach().half().cpu().numpy()
                for b, i in enumerate(plan.dataset_indices):
                    kl_logits[i] = logits[:, b, :]
            done += len(plan.dataset_indices)
        print(f"{split}: {done}/{len(pack)} datasets, {time.time() - started:.0f}s")

    assert np.isfinite(u_base).all(), "u_base has gaps"
    pack_sha = (args.pack_dir / "fable_real_v1.pack.sha256").read_text().split()[0]
    np.savez(
        args.pack_dir / "fable_base_proxy_v1.npz",
        u_base=u_base,
        seed_a=np.int64(args.seed_a),
        pack_sha256=np.array(pack_sha),
    )
    summary = {
        "u_base_mean": float(u_base.mean()),
        "u_base_median": float(np.median(u_base)),
        "seconds": round(time.time() - started, 1),
    }

    if args.kl_cache:
        D = len(pack)
        counts = np.zeros(D, dtype=np.int64)
        for i, arr in kl_logits.items():
            counts[i] = arr.size
        offsets = np.concatenate([[0], np.cumsum(counts)])
        flat = np.empty(int(offsets[-1]), dtype=np.float16)
        for i, arr in kl_logits.items():
            flat[offsets[i] : offsets[i + 1]] = arr.ravel()
        flat.tofile(args.pack_dir / "kl_base_logits_v1.bin")
        np.save(args.pack_dir / "kl_base_offsets_v1.npy", offsets)
        (args.pack_dir / "kl_base_meta_v1.json").write_text(
            json.dumps({"seed_a": args.seed_a, "pack_sha256": pack_sha,
                        "n_buckets": 5000}) + "\n"
        )
        summary["kl_cache_gb"] = round(flat.nbytes / 1e9, 2)

    print(json.dumps(summary))


if __name__ == "__main__":
    main()
