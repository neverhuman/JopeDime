#!/usr/bin/env python3
"""Encode the un-cached multiclass backlog, amortizing the expert load.

Multiclass has 1,250 source datasets and ~270 encoded, and it sits at 45
datasets with observations -- far below the 500-dataset floor that
`test_exact_train.py` requires before it will train on the task at all. So the
model has never learned multiclass, and every multiclass wave we evaluate buys
almost no learning. Draining this backlog is what changes that.

Why a batch loop rather than N invocations of `test_exact_expansion.py cache`:
`load_expert` is memoized PER PROCESS (encode_batch.py:10), and the expert is
233 MB. A single dataset measured 945 s end to end, and that is overwhelmingly
the one-time load, not the 0.6 MB of data. Paying it 980 times is the whole
cost; paying it once per worker is not.

Deliberately local. The cluster path needs a bundle activation that is blocked
behind a separate evidence chain, and the local path is already proven -- it
produced synth_0049e4a0e270dbfa. Lift should not wait on a deploy gate.

Loop health outranks drain speed. The evaluation queue recovered from zero
twice today; --workers stays modest and the caller lowers it if queued depth
falls.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

SOURCE_ROOT = Path("/home/ubuntu/remote_super/quant/multi_label")
CACHE_ROOT = Path("/home/ubuntu/jope-flywheel/encodings/multiclass-expanded")
# The only dataset over 10 MB in the corpus (24.5 MB against a 0.6 MB median).
# It runs last and alone so it cannot stall a worker that would otherwise
# finish dozens of ordinary datasets.
OUTLIER = "0f7f4024a9b28e9c"
# What materialize_cache must leave behind. Anything short of this is a broken
# cache, and a broken cache is worse than an absent one because the catalog
# will happily adopt it.
REQUIRED = (
    "cache-manifest.v1.json",
    "signature_manifest.json",
    "hyperion3train128dim.vs",
    "train3.vs",
    "train_target.vs",
    "train.csv",
    "test.csv",
)


def load_expansion():
    """Import the expansion module, which owns the encoder guards."""

    spec = importlib.util.spec_from_file_location(
        "test_exact_expansion", "/home/ubuntu/jope-dime/scripts/test_exact_expansion.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, "/home/ubuntu/jope-dime/scripts")
    spec.loader.exec_module(module)
    return module


def pending(expansion) -> list[str]:
    """Un-cached multiclass dataset ids, outlier last."""

    cached = {p.name for p in CACHE_ROOT.iterdir() if p.is_dir()} if CACHE_ROOT.is_dir() else set()
    ids = []
    for path in sorted(SOURCE_ROOT.iterdir()):
        if not path.is_dir() or path.name in cached:
            continue
        if not (path / "train.csv").is_file() or not (path / "meta.json").is_file():
            continue
        ids.append(path.name)
    ids.sort(key=lambda name: (name == OUTLIER, name))
    return ids


def encode_one(expansion, dataset_id: str) -> tuple[bool, float, str]:
    """Encode one dataset through the sanctioned materialize path."""

    started = time.time()
    source = SOURCE_ROOT / dataset_id
    try:
        meta = json.loads((source / "meta.json").read_text())
        entry = {
            "task": "multiclass",
            "dataset_id": dataset_id,
            "dataset_hash": dataset_id,
            "source": str(source),
            "source_family": expansion.source_family(meta, dataset_id),
            "train_sha256": expansion.sha256_file(source / "train.csv"),
            "test_sha256": expansion.sha256_file(source / "test.csv"),
        }
        expansion.materialize_cache(entry, encoder=expansion.existing_hyperion_encoder)
    except Exception as error:  # noqa: BLE001 - one bad dataset must not end the drain
        return False, time.time() - started, str(error)[:200]

    # Trust nothing: confirm the cache is complete rather than assuming a
    # clean return means a usable dataset.
    cache = CACHE_ROOT / dataset_id
    missing = [name for name in REQUIRED if not (cache / name).is_file()]
    if missing:
        return False, time.time() - started, f"incomplete cache, missing {missing}"
    return True, time.time() - started, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="stop after this many datasets")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--validate-one",
        action="store_true",
        help="encode exactly one dataset and report, for the correctness gate",
    )
    args = parser.parse_args()

    expansion = load_expansion()
    # Prove the encoder resolves BEFORE doing 980 datasets' worth of work; the
    # guards fail in milliseconds naming the path rather than after ~854s of a
    # misleading device message.
    expansion.ensure_foundation_expert_importable()
    expansion.widen_signature_timeout()

    todo = pending(expansion)
    if args.shards > 1:
        todo = [name for index, name in enumerate(todo) if index % args.shards == args.shard]
    if args.validate_one:
        todo = todo[:1]
    elif args.limit:
        todo = todo[: args.limit]

    print(
        f"[drain] shard {args.shard}/{args.shards}: {len(todo)} datasets "
        f"(pid {os.getpid()}); expert loads once for all of them",
        flush=True,
    )
    done = failed = 0
    elapsed_total = 0.0
    for index, dataset_id in enumerate(todo, start=1):
        ok, elapsed, error = encode_one(expansion, dataset_id)
        elapsed_total += elapsed
        if ok:
            done += 1
            # The marginal cost is the number that sizes the rest of the work;
            # the first dataset carries the expert load and the rest do not.
            print(
                f"[drain] {index}/{len(todo)} {dataset_id} ok {elapsed:.1f}s "
                f"(mean {elapsed_total / index:.1f}s)",
                flush=True,
            )
        else:
            failed += 1
            print(f"[drain] {index}/{len(todo)} {dataset_id} FAILED {elapsed:.1f}s: {error}", flush=True)
    print(f"[drain] shard {args.shard}: {done} encoded, {failed} failed", flush=True)
    sys.exit(1 if failed and not done else 0)


if __name__ == "__main__":
    main()
