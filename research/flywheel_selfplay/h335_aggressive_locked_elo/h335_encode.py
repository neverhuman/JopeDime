#!/usr/bin/env python3
"""Encode the SEARCH-row view sets of the H335 Phase T panel P201b with the FROZEN H317 recipe
(`crop_shuffle_synthetic_views.encode_context(enc, cid, row, "search")`, the same calls as H317 encode_search_views.py),
into $H317_DATA/views-search/<cid>.npz. Resumable; DOPE sha checked before/after; receipt written next to the views.
Run on xbabe2 inside `H335_VIEWS_RW=1 sandbox_h335.sh python3 h335_encode.py [--threads N] [--limit K]`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h335_core as K  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import crop_shuffle_synthetic_views as V  # noqa: E402

C = K.C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = C.train_rows()
    by = {r["context_sha256"]: r for r in rows}
    cids = K.panel_p201b(rows)
    if args.limit:
        cids = cids[: args.limit]
    out = V.VIEW_SEARCH_DIR
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    enc = V.DopeViewEncoder(threads=args.threads)
    done = skipped = 0
    for cid in cids:
        p = out / f"{cid}.npz"
        if V._view_file_ok(p):
            skipped += 1
            continue
        V._save_npz(p, V.encode_context(enc, cid, by[cid], "search"))
        done += 1
    after = enc.dope_sha()
    rec = {"schema": "jope.h335.search-view-encode/v1", "host": os.uname().nodename, "recipe": V.RECIPE_VERSION,
           "kind": "search", "panel": "P201b", "cids": cids, "encoded": done, "skipped_existing": skipped,
           "threads": args.threads, "elapsed_s": round(time.time() - t0, 1),
           "dope_sha256_before": enc.dope_sha_before, "dope_sha256_after": after,
           "dope_unchanged": after == enc.dope_sha_before == C.DOPE_SHA256, "guard": C.guard_report()}
    C.write_json(out / f"encode_receipt_{os.uname().nodename}_{len(cids)}.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "cids"}))
    if not rec["dope_unchanged"]:
        raise SystemExit("DOPE hash changed")


if __name__ == "__main__":
    main()
