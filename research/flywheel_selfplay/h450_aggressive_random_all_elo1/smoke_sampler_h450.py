#!/usr/bin/env python3
"""H450 pub-r4 CPU sampler smoke: draw real-table worker items (no model, no GPU) and report the exposure ledger,
the injected-column share (must be <= 10 % on every episode), dev91's share, row duplication and throughput.

  smoke_sampler_h450.py --bank BANK --bank-sha256 HEX --items 2000 --workers 8 --out SMOKE.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import secrets
import time
from pathlib import Path

import numpy as np

import train_aggressive_random_h450 as T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--bank-sha256", required=True)
    ap.add_argument("--items", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=140_000)
    ap.add_argument("--row-tokens", type=int, default=4_096)
    ap.add_argument("--bmax", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists (create-once)")
    t0 = time.time()
    bank = T.CombinedF64Bank(Path(a.bank), expect_sha256=a.bank_sha256)
    load_s = time.time() - t0
    led = T.Ledger(bank)
    nonce = hashlib.blake2b(secrets.token_bytes(32), digest_size=32).digest()
    ctx = mp.get_context("fork")
    q = ctx.Queue(maxsize=64)
    ev = ctx.Event()
    procs = [ctx.Process(target=T._episode_worker_loop, args=(w, nonce, bank, a.tokens, a.row_tokens, a.bmax, q, ev), daemon=True)
             for w in range(a.workers)]
    for p in procs:
        p.start()
    t1 = time.time()
    max_inj_frac, viol, dup_rows, shapes_bad, n_batched, n_eps = 0.0, 0, 0, 0, 0, 0
    widths, rows = [], []
    for i in range(a.items):
        xb, yb, n_ctx, infos, fps = q.get(timeout=300)
        led.add_real(infos)
        led.add_fps(fps, synthetic=False)
        for b_, fp in enumerate(fps):
            if T.dataset_fingerprint(xb[b_], yb[b_], n_ctx) != fp:
                raise SystemExit("fingerprint does not reproduce in the main process")
        n_batched += int(xb.shape[0] > 1 or len(infos) > 1)
        for b, (t, tm, k, F, n, sp) in enumerate(infos):
            n_eps += 1
            frac = k / F
            max_inj_frac = max(max_inj_frac, frac)
            viol += int(k > int(np.floor(0.10 * F)))
            shapes_bad += int(xb.shape[1] != n or xb.shape[2] != F)
            widths.append(F)
            rows.append(n)
            x = xb[b]
            u = np.unique(np.round(np.nan_to_num(x, nan=1.2345e-300), 12), axis=0)
            dup_rows += int(u.shape[0] < x.shape[0])
    wall = time.time() - t1
    ev.set()
    for p in procs:
        p.terminate()
    s = led.summary()
    doc = {"schema": "jope.h450.smoke/v1", "items": a.items, "episodes": n_eps, "batched_items": n_batched,
           "bank_sha256": bank.sha256, "bank_tables": bank.n_tables, "bank_load_s": round(load_s, 2),
           "workers": a.workers, "wall_s": round(wall, 2), "episodes_per_s": round(n_eps / wall, 2),
           "max_injected_fraction": max_inj_frac, "injection_rule_violations": viol, "shape_mismatches": shapes_bad,
           "episodes_with_duplicate_rows": dup_rows, "width_quantiles": np.percentile(widths, [5, 25, 50, 75, 95]).tolist(),
           "row_quantiles": np.percentile(rows, [5, 25, 50, 75, 95]).tolist(), "exposure": s}
    with open(out, "x") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(json.dumps({k: doc[k] for k in ("episodes", "episodes_per_s", "max_injected_fraction", "injection_rule_violations",
                                          "shape_mismatches", "episodes_with_duplicate_rows")}),
          json.dumps({k: s[k] for k in ("dev91_share_of_real_episodes", "injected_column_share", "target_modes", "target_spaces",
                                        "unique_datasets_seen", "datasets_seen", "duplicate_real_datasets",
                                        "distinct_source_tables_seen")}))


if __name__ == "__main__":
    main()
