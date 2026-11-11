#!/usr/bin/env python3
"""H421 step 2 - fill (PREREG section 3.2): one SEARCH evaluation of every pool candidate on every design context.

A SEARCH evaluation is the engine's 2-fold inner cross-validation on SEARCH rows; it returns each fold's
out-of-fold predictions (p0, p1), so one evaluation gives a candidate's out-of-fold SEARCH prediction for the whole
context. Results already in the run directory's memo (the stored H416 / H417 / H420 evaluations of that context) are
hits; only missing ones run. No AUDIT evaluation is requested (kind "search" only).

The context's own stored stack candidates (the columns of its stored H matrix; shas from --hsets) are read from the
memo as well: their stored SEARCH evaluations give the same 2-fold out-of-fold predictions, so the selection can
measure a candidate against what the stack of that context already holds. They are read, never re-evaluated.

Per context, create-once: <run-dir>/fill_npz/<cid>.npz
  P [n_search, m] float32 out-of-fold predictions of the m candidates that completed (fold 0 rows, then fold 1 rows)
  y [n_search] targets in the same order; n0 = rows of fold 0; b0, b1 the worker's fold baselines
  cand [m] pool ids of the columns; status [pool] 0 = completed, 1 = failed / timeout, 2 = not valid on this table
  utility, fitness, wall_s [pool] (nan where not completed); gate [pool] the search gate; head, sha [pool] the head op and program sha
  PH [n_search, h] float32, h_sha, h_fitness, h_gate, h_head: the stored stack candidates found in the memo; h_missing

  fill_h421.py --run-dir R --pool R/candidates.json --hsets R/hsets.json --cores L --shard k/n [--timeout 60]
               [--budget-s S] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NEURAL = {"GS": "H0.1-A"}   # build_rt needs one screener; the fill never calls it


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--hsets", required=True, help="json: cid -> shas of the context's stored stack candidates")
    ap.add_argument("--cores", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--timeout", type=float, default=60.0)   # the search limit (PREREG 3.2)
    ap.add_argument("--budget-s", type=float, default=0.0, help="stop before a new context once this shard's worker seconds reach it")
    ap.add_argument("--limit", type=int, default=0, help="first N contexts of the shard (timing sample)")
    ap.add_argument("--cids", default="", help="comma-separated context ids instead of the pool's design list (smoke)")
    a = ap.parse_args()

    import numpy as np

    import apex_core as A  # FIRST import of the GP stack: h335_core pins + guard

    import lang_h420 as LANG

    DH, K, C, L, G = A.DH, A.K, A.C, A.L, A.G
    DP = DH.DP
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    A.install()
    LANG.install(G)   # the extended language: LIB_EXT presets and the wide-table rule parse as in H420
    chk = LANG.selfcheck(G)
    raw = Path(a.pool).read_bytes()
    doc = json.loads(raw)
    pool = doc["pool"]
    hraw = Path(a.hsets).read_bytes()
    hsets = json.loads(hraw)
    closed = set(A.AM.cohort("DEV", "all")) | set(A.AM.cohort("CONFIRM", "all"))
    cids = [c for c in a.cids.split(",") if c] or list(doc["cids"])
    if set(cids) - set(doc["cids"]) or set(cids) & closed:
        raise SystemExit("a context outside the design list, or a DEV / CONFIRM context, was requested")
    run_dir = Path(a.run_dir)
    memo = run_dir / f"gproot_{K.GROUP}" / "memo"
    memo.mkdir(parents=True, exist_ok=True)
    worker = Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"
    nd = {s: Path(C.DATA_ROOT) / "neural" / "runs" / n for s, n in NEURAL.items()}
    rt, _scr, _man = A.build_rt(run_dir, L.parse_cpus(a.cores), worker, a.timeout, (A.SEARCH_SEED,), nd)
    out_dir = run_dir / "fill_npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    sk, sn = (int(v) for v in a.shard.split("/"))
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))[sk::sn]
    if a.limit:
        order = order[: a.limit]
    t0, done, skipped, stopped = time.time(), 0, 0, False
    per = []
    for cid in order:
        p = out_dir / f"{cid}.npz"
        if p.is_file():
            skipped += 1
            continue
        w_before = float(rt.engine.stats().get("wall_s_total", 0.0))
        if a.budget_s and w_before >= a.budget_s:
            stopped = True
            break
        ctx = rt.engine.context(cid, rt.fam_of[cid])
        cands, ids = [], []
        status = np.full(len(pool), 2, np.int8)
        for e in pool:
            gm = DP._pipe(e["dsl"])
            c = DP._cand(gm, f"h421_fill:{e['name']}", ctx.width) if gm is not None else None
            if c is not None:
                cands.append(c); ids.append(e["id"])
        res = rt.engine._evaluate(ctx, cands, "search") if cands else []
        y = np.concatenate([ctx.y0, ctx.y1]).astype(np.float64)
        util = np.full(len(pool), np.nan); fit = np.full(len(pool), np.nan); wall = np.full(len(pool), np.nan)
        gate = np.zeros(len(pool), bool)
        head = np.asarray([""] * len(pool), dtype="<U32")
        sha = np.asarray([""] * len(pool), dtype="<U64")
        for i, c in zip(ids, cands):
            head[i] = DH.DP15.head_of(c.dsl); sha[i] = c.sha
        cols, keep = [], []

        def usable(r) -> bool:
            return bool(r is not None and r.ok and len(r.p0) == len(ctx.y0) and len(r.p1) == len(ctx.y1))

        for i, r in zip(ids, res):
            wall[i] = float(r.wall_s)
            if usable(r):
                status[i] = 0; util[i] = float(r.utility); fit[i] = float(r.fitness); gate[i] = bool(r.gate)
                cols.append(np.concatenate([r.p0, r.p1]).astype(np.float32)); keep.append(i)
            else:
                status[i] = 1
        env = ctx.kind_sha("search")
        hcols, hsha, hfit, hgate, hhead, hmiss = [], [], [], [], [], 0
        for s in hsets.get(cid, []):   # stored evaluations only: a memo read, never a request
            r = rt.engine._memo.get((cid, s, "search", env))
            if usable(r):
                hcols.append(np.concatenate([r.p0, r.p1]).astype(np.float32)); hsha.append(s)
                hfit.append(float(r.fitness)); hgate.append(bool(r.gate)); hhead.append(DH.DP15.head_of(r.dsl))
            else:
                hmiss += 1
        empty = np.zeros((len(y), 0), np.float32)
        buf = io.BytesIO()
        np.savez_compressed(buf, P=np.column_stack(cols) if cols else empty, y=y,
                            n0=np.int64(len(ctx.y0)), b0=np.float64(ctx.b0), b1=np.float64(ctx.b1),
                            width=np.int64(ctx.width), cand=np.asarray(keep, np.int64), status=status, utility=util,
                            fitness=fit, wall_s=wall, gate=gate, head=head, sha=sha,
                            PH=np.column_stack(hcols) if hcols else empty, h_head=np.asarray(hhead, dtype="<U32"),
                            h_sha=np.asarray(hsha, dtype="<U64"), h_fitness=np.asarray(hfit, np.float64),
                            h_gate=np.asarray(hgate, bool), h_missing=np.int64(hmiss))
        tmp = out_dir / f".{cid}.{os.getpid()}.tmp"
        tmp.write_bytes(buf.getvalue())
        os.link(tmp, p)   # create-once
        tmp.unlink()
        done += 1
        per.append({"cid": cid, "n_search": int(len(y)), "width": int(ctx.width), "valid": len(ids), "completed": len(keep),
                    "h_found": len(hsha), "h_missing": hmiss,
                    "worker_s": float(rt.engine.stats().get("wall_s_total", 0.0)) - w_before})
    st = {"wave": "H421", "step": "fill", "shard": a.shard, "contexts_done": done, "contexts_skipped": skipped,
          "budget_stop": stopped, "wall_s": time.time() - t0, "pool_sha256": hashlib.sha256(raw).hexdigest(),
          "hsets_sha256": hashlib.sha256(hraw).hexdigest(),
          "pool": len(pool), "timeout_s": a.timeout, "engine": rt.engine.stats(), "guard": C.guard_report(),
          "audit_ledger": rt.ledger.check(), "physical_by_kind": DP.physical_by_kind(rt), "lang_selfcheck": chk,
          "manifest_sha256": A.MANIFEST_SHA256, "host": os.uname().nodename, "per_context": per}
    if any(k != "search" for k, v in st["physical_by_kind"].items() if v):
        raise AssertionError("a non-search evaluation ran in the fill")
    C.write_json(run_dir / f"status_fill_{int(t0)}_s{sk}of{sn}.json", st)
    print(json.dumps({k: st[k] for k in ("shard", "contexts_done", "contexts_skipped", "budget_stop", "wall_s", "physical_by_kind")}
                     | {"worker_s": st["engine"].get("wall_s_total"), "timeouts": st["engine"].get("timeouts")}))
    rt.engine.close()


if __name__ == "__main__":
    main()
