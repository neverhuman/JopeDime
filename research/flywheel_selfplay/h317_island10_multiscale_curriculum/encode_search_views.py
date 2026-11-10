#!/usr/bin/env python3
"""Encode SEARCH-row view sets (the canonical `kind="search"` recipe of crop_shuffle_synthetic_views) for an explicit
cid list, and compare two view directories array-by-array. A thin driver: it changes no recipe code.

  encode   --cids-file F --view-dir D [--threads N] [--limit K --sample-seed S]
           every cid must be a TRAIN context of the current GP pilot (`common_h317.pilot_shard(rows, folds)`); writes
           D/<cid>.npz exactly as `crop_shuffle_synthetic_views.encode_context(enc, cid, row, "search")`; resumable;
           DOPE sha checked before/after; receipt D/encode_receipt_<host>_<n>.json.
  compare  --a DIR --b DIR [--cids-file F]   bit-identity of every array of every common (or listed) context.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402
import crop_shuffle_synthetic_views as V  # noqa: E402


def _cids(path: str) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def cmd_encode(args) -> None:
    rows = C.train_rows()
    pilot = set(C.pilot_shard(rows, C.family_folds(rows)))
    by = {r["context_sha256"]: r for r in rows}
    cids = _cids(args.cids_file)
    bad = [c for c in cids if c not in by or c not in pilot]
    if bad or len(set(cids)) != len(cids):
        raise SystemExit(
            f"{len(bad)} cids are not TRAIN pilot contexts (or duplicates)"
        )
    if args.limit:
        rng = np.random.default_rng(
            C.stream_seed("H317-search-xcheck", args.sample_seed)
        )
        cids = sorted(rng.choice(sorted(cids), size=args.limit, replace=False).tolist())
    out = Path(args.view_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    enc = V.DopeViewEncoder(threads=args.threads)
    done = skipped = 0
    for cid in cids:
        p = out / f"{cid}.npz"
        if V._view_file_ok(p):
            skipped += 1
            continue
        arrays = V.encode_context(enc, cid, by[cid], "search")
        V._save_npz(p, arrays)
        done += 1
    after = enc.dope_sha()
    rec = {
        "schema": "jope.h317.search-view-encode/v1",
        "host": os.uname().nodename,
        "recipe": V.RECIPE_VERSION,
        "kind": "search",
        "cids": cids,
        "encoded": done,
        "skipped_existing": skipped,
        "threads": args.threads,
        "elapsed_s": round(time.time() - t0, 1),
        "dope_sha256_before": enc.dope_sha_before,
        "dope_sha256_after": after,
        "dope_unchanged": after == enc.dope_sha_before == C.DOPE_SHA256,
        "guard": C.guard_report(),
    }
    C.write_json(out / f"encode_receipt_{os.uname().nodename}_{len(cids)}.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "cids"}))
    if not rec["dope_unchanged"]:
        raise SystemExit("DOPE hash changed")


def cmd_compare(args) -> None:
    a, b = Path(args.a), Path(args.b)
    names = (
        [f"{c}.npz" for c in _cids(args.cids_file)]
        if args.cids_file
        else sorted(p.name for p in a.glob("*.npz") if (b / p.name).exists())
    )
    n = exact = 0
    diffs = []
    for name in names:
        with (
            np.load(a / name, allow_pickle=False) as x,
            np.load(b / name, allow_pickle=False) as y,
        ):
            n += 1
            same = set(x.files) == set(y.files) and all(
                np.array_equal(x[k], y[k]) for k in x.files
            )
            exact += int(same)
            if not same:
                diffs.append(name)
    rep = {
        "a": str(a),
        "b": str(b),
        "contexts": n,
        "bit_identical": exact,
        "differing": diffs[:5],
        "all_exact": n > 0 and exact == n,
    }
    print(json.dumps(rep))
    if not rep["all_exact"]:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--cids-file", required=True)
    e.add_argument("--view-dir", required=True)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--sample-seed", default="0")
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--cids-file", default="")
    args = ap.parse_args()
    {"encode": cmd_encode, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
