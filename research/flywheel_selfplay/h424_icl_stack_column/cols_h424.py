#!/usr/bin/env python3
"""H424: stack columns of the H419 in-context model, CPU inference only (no GP engine, no table is opened here).

  cols_h424.py --bank FILE --ckpt CKPT --ckpt-sha256 HEX --size M --out DIR [--threads 1] [--shard k/n] [--budget-s S]

For every context of the bank (bank_h424.py), cheapest first (rows x features), and for the "ridgeres" readout of the
model (icl_diag.run_variant at its
published bytes: ridge on the context rows plus the model's prediction of the ridge residual),
with the checkpoint's EMA weights:
  q    [n_search]  5-fold out-of-fold prediction of the SEARCH rows: each fold of the stored fold labels is the query
                   in turn, the other four folds are the context (the stored candidates' out-of-fold matrix is built
                   on the same folds);
  bag  [n_audit]   mean of those five fold-context models' predictions of the AUDIT rows (the stored "bag" column);
  ref  [n_audit]   prediction of the AUDIT rows with all SEARCH rows as the context (the stored "refit" column).
The model sees SEARCH targets of context rows only. AUDIT targets are not in the bank; query targets are passed as
zeros and do not enter a prediction. One create-once OUT/<cid>.npz per context (a restart continues); the run stops
before a new context once --budget-s seconds have passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import icl_diag as DG  # noqa: E402
from icl_model import ICL  # noqa: E402

VARIANTS = ("ridgeres",)   # "full" (the plain readout) is not computed: CPU inference is too slow for two readouts
SEED = 424_700


def columns(model, S: torch.Tensor, lab: torch.Tensor, XA: torch.Tensor, i: int) -> dict:
    """S [n,F+1] SEARCH rows (target last), lab [n] folds 0..4, XA [na,F] AUDIT features."""
    n, na = S.shape[0], XA.shape[0]
    X, y = S[:, :-1], S[:, -1]
    out = {v: {"q": torch.zeros(n, dtype=torch.float64), "bag": torch.zeros(na, dtype=torch.float64)} for v in VARIANTS}
    zeros = lambda k: torch.zeros(k, dtype=y.dtype)
    for k in range(5):
        ci, qi = torch.nonzero(lab != k).squeeze(1), torch.nonzero(lab == k).squeeze(1)
        x = torch.cat([X[ci], X[qi], XA])
        yy = torch.cat([y[ci], zeros(len(qi) + na)])
        for v in VARIANTS:
            p = DG.run_variant(model, v, x, yy, int(ci.numel()), SEED + 10 * i + k)
            out[v]["q"][qi] = p[:len(qi)]
            out[v]["bag"] += p[len(qi):] / 5.0
    x, yy = torch.cat([X, XA]), torch.cat([y, zeros(na)])
    for v in VARIANTS:
        out[v]["ref"] = DG.run_variant(model, v, x, yy, n, SEED + 10 * i + 9)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-sha256", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--budget-s", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="k/n: this process takes every n-th context of the cheapest-first order")
    a = ap.parse_args()
    if a.threads > 16:
        raise SystemExit("at most 16 threads")
    t0, c0 = time.time(), time.process_time()
    torch.set_num_threads(a.threads)
    random.seed(0)
    torch.manual_seed(0)
    if hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest() != a.ckpt_sha256:
        raise SystemExit("the checkpoint is not the one named by --ckpt-sha256")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ICL(a.size)
    model.load_state_dict(ck["ema"])
    model.eval()
    z = np.load(a.bank, allow_pickle=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ss, sa = z["s_shape"].astype(np.int64), z["a_shape"].astype(np.int64)
    s_off = np.concatenate([[0], np.cumsum(ss[:, 0] * ss[:, 1])])
    a_off = np.concatenate([[0], np.cumsum(sa[:, 0] * sa[:, 1])])
    l_off = np.concatenate([[0], np.cumsum(ss[:, 0])])
    cells = (ss[:, 0] + sa[:, 0]) * (ss[:, 1] - 1)
    order = sorted(range(len(z["cids"])), key=lambda j: (int(cells[j]), str(z["cids"][j])))   # cheapest first
    k, n = (int(v) for v in a.shard.split("/"))
    order = order[k::n]
    if a.limit:
        order = order[:a.limit]
    done = skipped = 0
    stop = False
    T = len(order)
    for i in order:
        (r, c), (ra, ca) = ss[i].tolist(), sa[i].tolist()
        S = torch.from_numpy(z["search"][s_off[i]:s_off[i + 1]].reshape(r, c).copy())
        XA = torch.from_numpy(z["audit"][a_off[i]:a_off[i + 1]].reshape(ra, ca).copy())
        lab = torch.from_numpy(z["lab"][l_off[i]:l_off[i + 1]].astype(np.int64))
        cid = str(z["cids"][i])
        p = out / f"{cid}.npz"
        if p.exists():
            skipped += 1
            continue
        if a.budget_s and time.time() - t0 > a.budget_s:
            stop = True
            break
        t1 = time.time()
        cols = columns(model, S, lab, XA, i)
        ok = all(bool(torch.isfinite(cols[v][k]).all()) for v in VARIANTS for k in ("q", "bag", "ref"))
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, ok=ok, wall_s=time.time() - t1, n_search=r, n_audit=ra, features=c - 1,
                 **{f"{k}_{v}": cols[v][k].numpy() for v in VARIANTS for k in ("q", "bag", "ref")})
        tmp.rename(p)
        done += 1
        print(json.dumps({"i": i, "cid": cid[:12], "rows": r, "F": c - 1, "na": ra, "ok": ok,
                          "s": round(time.time() - t1, 1)}), flush=True)
    rec = {"schema": "jope.h424.cols/v1", "bank": a.bank, "bank_sha256": hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(),
           "ckpt_sha256": a.ckpt_sha256, "ckpt_state": ck.get("state"), "size": a.size, "weights": "ema",
           "threads": a.threads, "device": "cpu", "cuda_visible": bool(torch.cuda.is_available()),
           "contexts": int(T), "written": done, "already_there": skipped, "budget_stop": stop,
           "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
    rec["shard"] = a.shard
    (out / f"cols_status_{int(t0)}_s{k}of{n}.json").write_text(json.dumps(rec, sort_keys=True))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
