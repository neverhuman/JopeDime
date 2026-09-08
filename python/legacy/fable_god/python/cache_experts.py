"""Cache TabPFN-v3 + TabICL internal FEATURES on the RIGHT data: the real hyperion
binary datasets that the Elo actually grades (router cache hashes -> raw tables at
~/quant/binary/<hash>). One .pt per dataset holds BOTH experts' 512-d hidden reps +
logits + conf + labels, so the single fused model (frozen experts + trainable head)
fine-tunes FAST on just these combined features. Shardable across the 8 A100s.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from python.fusion import experts as E
from python.fusion.data import load_dataset, gate_meta_vector

sys.path.insert(0, os.path.expanduser("~/hyperion/eval_harness"))
import vendored_elo as v  # noqa: E402

QBIN = os.path.expanduser("~/quant/binary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=os.path.expanduser("~/fable_god/cache/router/real_bin"))
    args = ap.parse_args()
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)

    D = v.load_cache("bin")
    hashes = list(D["hashes"])[args.gpu::args.nshards]     # GPU shard
    pfn = E.TabPFNv3ClsBinary(dev); icl = E.TabICLCls(dev)
    ok = skip = 0
    for h in hashes:
        fp = os.path.join(args.out, f"{h}.pt")
        if os.path.exists(fp):
            ok += 1; continue
        d = os.path.join(QBIN, h)
        try:
            ds = load_dataset(d)
            op, oi = pfn.infer(ds), icl.infer(ds)
            ent = {
                "hash": h,
                "hidden": torch.stack([op.hidden, oi.hidden], 1).half().cpu(),   # (M,2,512)
                "logits": torch.stack([op.logits, oi.logits], 1).cpu(),          # (M,2,2)
                "conf": torch.stack([op.conf, oi.conf], 1).cpu(),                # (M,2,3)
                "gate_meta": torch.from_numpy(gate_meta_vector(ds)),
                "y": torch.from_numpy(ds.yte.astype("int64")),
            }
            torch.save(ent, fp); ok += 1
        except Exception as e:
            skip += 1
            if skip <= 5:
                print(f"skip {h}: {type(e).__name__}: {e}", flush=True)
        if (ok + skip) % 40 == 0:
            print(f"gpu{args.gpu}: {ok+skip}/{len(hashes)} ({ok} ok)", flush=True)
    print(f"gpu{args.gpu} DONE: {ok} cached / {skip} skipped -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
