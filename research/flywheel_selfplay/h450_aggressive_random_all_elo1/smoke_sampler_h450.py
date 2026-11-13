#!/usr/bin/env python3
"""H450 pub-r4 CPU sampler smoke: draw real-table worker items (no model, no GPU) and report the exposure ledger,
the injected-column share (at most floor(inj_max_frac * F) per episode, default 10 %), dev91's share, row duplication and throughput.
pub-r12: with the trainer's AUG flags it also counts episodes that miss the per-episode drop or injection.
pub-r14: the max-entropy flags (--inj-max-frac, --rows-random, --target-random, --feat-affine-noise,
--uniform-by-dataset) pass through to the trainer AUG; the injection checks use the trainer cap floor(inj_max_frac * F).

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
    ap.add_argument("--feature-drop-every-batch", type=int, choices=(0, 1), default=0)
    ap.add_argument("--feature-drop-max-frac", type=float, default=0.35)
    ap.add_argument("--synth-inj-every-batch", type=int, choices=(0, 1), default=0)
    ap.add_argument("--inj-max-frac", type=float, default=T.INJ_MAX_FRAC)
    for fl in ("--rows-random", "--target-random", "--feat-affine-noise", "--uniform-by-dataset"):
        ap.add_argument(fl, type=int, choices=(0, 1), default=0)
    a = ap.parse_args()
    T.AUG.update(feature_drop_every=bool(a.feature_drop_every_batch), feature_drop_max_frac=float(a.feature_drop_max_frac),
                 synth_inj_every=bool(a.synth_inj_every_batch), inj_max_frac=float(a.inj_max_frac),
                 rows_random=bool(a.rows_random), target_random=bool(a.target_random),
                 feat_affine_noise=bool(a.feat_affine_noise), uniform_by_dataset=bool(a.uniform_by_dataset))   # before the fork, as in the trainer
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
    drop_miss, inj_miss, drop_over_unb, drop_over_bat, drop_fracs = 0, 0, 0, 0, []
    for i in range(a.items):
        xb, yb, n_ctx, infos, fps = q.get(timeout=300)
        led.add_real(infos)
        led.add_fps(fps, synthetic=False)
        for b_, fp in enumerate(fps):
            if T.dataset_fingerprint(xb[b_], yb[b_], n_ctx) != fp:
                raise SystemExit("fingerprint does not reproduce in the main process")
        n_batched += int(xb.shape[0] > 1 or len(infos) > 1)
        batched = len(infos) > 1 or xb.shape[0] > 1
        for b, (t, tm, k, F, n, sp, d, p) in enumerate(infos):
            n_eps += 1
            if p >= 2:   # pub-r12 per-episode audit: >= 1 real column removed, >= 1 injected when the cap allows it
                drop_miss += int(d < 1)
                drop_fracs.append(d / p)
                # tables wider than WIDE_GPU always lose columns to the width cap; batch mates lose what F_t needs
                over = p <= T.WIDE_GPU and d > max(1, int(np.floor(a.feature_drop_max_frac * p)))
                drop_over_bat += int(over and batched)
                drop_over_unb += int(over and not batched)
            if T._inj_cap(F) >= 1:
                inj_miss += int(k < 1)
            frac = k / F
            max_inj_frac = max(max_inj_frac, frac)
            viol += int(k > T._inj_cap(F))
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
           "row_quantiles": np.percentile(rows, [5, 25, 50, 75, 95]).tolist(), "exposure": s,
           "augmentation": dict(T.AUG), "episodes_p2_without_drop": drop_miss, "episodes_f10_without_injection": inj_miss,
           "drop_over_max_frac_unbatched": drop_over_unb, "drop_over_max_frac_batch_mates": drop_over_bat,
           "drop_frac_quantiles": np.percentile(drop_fracs, [5, 25, 50, 75, 95]).tolist() if drop_fracs else None}
    with open(out, "x") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(json.dumps({k: doc[k] for k in ("episodes", "episodes_per_s", "max_injected_fraction", "injection_rule_violations",
                                          "shape_mismatches", "episodes_with_duplicate_rows", "augmentation",
                                          "episodes_p2_without_drop", "episodes_f10_without_injection",
                                          "drop_over_max_frac_unbatched", "drop_over_max_frac_batch_mates", "drop_frac_quantiles")}),
          json.dumps({k: s[k] for k in ("dev91_share_of_real_episodes", "injected_column_share", "target_modes", "target_spaces",
                                        "unique_datasets_seen", "datasets_seen", "duplicate_real_datasets",
                                        "distinct_source_tables_seen")}))


if __name__ == "__main__":
    main()
