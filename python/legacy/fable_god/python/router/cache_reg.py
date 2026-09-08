"""Cache the two frozen REGRESSION experts' features over packed regression tasks, so
the regression fusion trunk trains fast (zero expert forward per step). Regression
experts are single-infer (no batched path), so this shards across the 8 A100s.

Per task saves BOTH experts' 512-d hidden + per-row mean + quantiles + confidence +
a generalizable dataset descriptor + the target y.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from python.fusion import experts as E
from python.fusion.data import (iter_packed_shards, load_packed_shard, group_by_shape,
                                gate_meta_vector)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-shards", type=int, default=300)
    ap.add_argument("--out", default=os.path.expanduser("~/fable_god/cache/fusion/reg_packed"))
    args = ap.parse_args()
    dev = torch.device("cuda"); os.makedirs(args.out, exist_ok=True)

    pfn = E.TabPFNv3Reg(dev); icl = E.TabICLReg(dev)
    shards = list(iter_packed_shards("regression_packed", args.split))[:args.max_shards]
    shards = shards[args.gpu::args.nshards]
    ok = skip = 0
    for si, sh in enumerate(shards):
        name = os.path.basename(sh).replace(".npz", "")
        fp = os.path.join(args.out, f"{args.split}_{name}.pt")
        if os.path.exists(fp):
            continue
        try:
            groups = group_by_shape(load_packed_shard(sh))     # standardize ctx + n_test=128
            entries = []
            for grp in groups:
                for t in grp:
                    op, oi = pfn.infer(t), icl.infer(t)
                    entries.append({
                        "hash": t.hash,
                        "hidden": torch.stack([op.hidden, oi.hidden], 1).half().cpu(),   # (M,2,512)
                        "means": torch.stack([op.mean, oi.mean], -1).cpu(),              # (M,2)
                        "quant": torch.stack([op.quantiles, oi.quantiles], 1).cpu(),     # (M,2,Q)
                        "conf": torch.stack([op.conf, oi.conf], 1).cpu(),               # (M,2,3)
                        "gate_meta": torch.from_numpy(gate_meta_vector(t)),
                        "y": torch.from_numpy(t.yte.astype("float32")),                 # (M,)
                    })
                    ok += 1
            if entries:
                torch.save(entries, fp)
        except Exception as e:
            skip += 1
            if skip <= 5:
                print(f"skip {name}: {type(e).__name__}: {e}", flush=True)
        if (si + 1) % 5 == 0:
            print(f"gpu{args.gpu} shard {si+1}/{len(shards)} | {ok} tasks cached", flush=True)
    print(f"gpu{args.gpu} DONE ({args.split}): {ok} tasks / {skip} skipped -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
