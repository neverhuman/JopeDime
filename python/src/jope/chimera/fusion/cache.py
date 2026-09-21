"""Precompute frozen-expert outputs once, so the fusion head trains with zero
expert forwards (the 'learn fused weights fast' enabler). One .pt per dataset."""

from __future__ import annotations

import os
import time

import torch

from .data import gate_meta_vector, load_dataset


def build_cache(dirs, experts, out_dir, device="cuda", log_every=25, max_test=None):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for i, d in enumerate(dirs):
        h = os.path.basename(d.rstrip("/"))
        fp = os.path.join(out_dir, f"{h}.pt")
        if os.path.exists(fp):
            continue
        try:
            ds = load_dataset(d)
            if ds.n_features < 1 or ds.Xte.shape[0] < 2 or ds.n_train < 8:
                raise ValueError("too small")
            if max_test and ds.Xte.shape[0] > max_test:  # cap test rows for speed
                ds.Xte = ds.Xte[:max_test]
                ds.yte = ds.yte[:max_test]
            outs = [ex.infer(ds) for ex in experts]
        except Exception as e:
            print(f"skip {h}: {type(e).__name__}: {e}", flush=True)
            continue
        entry = {
            "hash": h,
            "task": ds.task,
            "hidden": torch.stack([o.hidden for o in outs], 1),  # (M,E,512)
            "conf": torch.stack([o.conf for o in outs], 1),  # (M,E,Cc)
            "gate_meta": torch.from_numpy(gate_meta_vector(ds)),  # (Dm,)
            "y": torch.from_numpy(ds.yte.astype("float32")),  # (M,)
        }
        if outs[0].logits is not None:  # classification: (M,E,K)
            entry["logits"] = torch.stack([o.logits for o in outs], 1)
        if outs[0].mean is not None:  # regression
            entry["means"] = torch.stack([o.mean for o in outs], -1)  # (M,E)
            entry["quant"] = torch.stack([o.quantiles for o in outs], 1)
        torch.save(entry, fp)
        if (i + 1) % log_every == 0:
            print(f"cached {i + 1}/{len(dirs)} ({(time.time() - t0) / (i + 1):.2f}s/ds)", flush=True)
    print(f"cache done: {len(os.listdir(out_dir))} entries in {out_dir}")


def load_cache(out_dir, device="cpu"):
    entries = []
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".pt"):
            e = torch.load(os.path.join(out_dir, f), map_location=device, weights_only=False)
            entries.append(e)
    return entries


# ---- batched packed cache: order-of-magnitude faster over the whole corpus ----
def build_cache_packed(
    bases,
    experts,
    out_dir,
    split="train",
    max_shards=None,
    max_tasks=None,
    log_every=20,
    shard_index=0,
    nshards=1,
    col_target_k=0,
):
    """Iterate packed shards, batch same-shape tasks through the frozen experts,
    write one .pt per input shard (many tasks). bases: list like ['real_packed',
    'synthetic_packed']."""
    from .data import gate_meta_vector, group_by_shape, iter_packed_shards, load_packed_shard

    os.makedirs(out_dir, exist_ok=True)
    shards = []
    for b in bases:
        shards += list(iter_packed_shards(b, split))
    if max_shards:
        shards = shards[:max_shards]
    shards = shards[shard_index::nshards]  # GPU sharding
    t0 = time.time()
    ntask = 0
    for si, sh in enumerate(shards):
        name = os.path.basename(sh).replace(".npz", "")
        fp = os.path.join(out_dir, f"{name}.pt")
        if os.path.exists(fp):
            continue
        try:
            tasks = load_packed_shard(sh)
            if col_target_k:  # near-infinite real tasks
                from .data import column_as_target_tasks

                aug = []
                for t in tasks:
                    aug.extend(column_as_target_tasks(t, k=col_target_k, seed=si))
                tasks = tasks + aug
            groups = group_by_shape(tasks)
            # cap batch by feature-count to bound attention memory (avoid OOM on
            # wide tables); throughput stays high on typical narrow tasks.
            subgroups = []
            for grp in groups:
                mb = max(4, min(64, 4000 // max(grp[0].n_features, 1)))
                for s in range(0, len(grp), mb):
                    subgroups.append(grp[s : s + mb])
            entries = []
            for grp in subgroups:
                outs = [ex.infer_batch(grp) for ex in experts]  # list[E] of list[B]
                for j, tk in enumerate(grp):
                    o = [outs[e][j] for e in range(len(experts))]
                    ent = {
                        "hash": tk.hash,
                        "task": tk.task,
                        "hidden": torch.stack([x.hidden for x in o], 1),  # (M,E,512)
                        "conf": torch.stack([x.conf for x in o], 1),
                        "gate_meta": torch.from_numpy(gate_meta_vector(tk)),
                        "y": torch.from_numpy(tk.yte.astype("float32")),
                    }
                    if o[0].logits is not None:
                        ent["logits"] = torch.stack([x.logits for x in o], 1)
                    if o[0].mean is not None:
                        ent["means"] = torch.stack([x.mean for x in o], -1)
                        ent["quant"] = torch.stack([x.quantiles for x in o], 1)
                    entries.append(ent)
                    ntask += 1
            torch.save(entries, fp)
        except Exception as e:
            print(f"skip shard {name}: {type(e).__name__}: {e}", flush=True)
            continue
        if (si + 1) % log_every == 0:
            print(
                f"shard {si + 1}/{len(shards)} | {ntask} tasks | {ntask / (time.time() - t0):.0f} tasks/s",
                flush=True,
            )
        if max_tasks and ntask >= max_tasks:
            break
    print(f"packed cache done: {ntask} tasks in {out_dir} ({time.time() - t0:.0f}s)")


def load_cache_packed(out_dir, device="cpu", limit=None, files=None):
    """Load per-shard .pt files (each a list of task entries) into a flat list.
    `files`: explicit list of paths (for per-rank sharded loading — avoids every
    DDP rank holding the whole cache in RAM)."""
    if files is None:
        files = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if f.endswith(".pt")]
    entries = []
    for fp in files:
        obj = torch.load(fp, map_location=device, weights_only=False)
        entries.extend(obj if isinstance(obj, list) else [obj])
        if limit and len(entries) >= limit:
            break
    return entries


def list_cache_files(out_dir):
    return [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if f.endswith(".pt")]
