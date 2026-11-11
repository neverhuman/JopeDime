#!/usr/bin/env python3
"""H430: stack columns of the exact-key encoders, CPU only (no GP engine, no model checkpoint).

  cols_h430.py --bank FILE --out DIR [--shard k/n] [--budget-s S] [--ctx-s 900]

For every context of the bank (bank_h424.py layout; cheapest first by rows x features) and every column of
keys_core.NAMES, with the stored 5-fold labels of the SEARCH rows:
  q_<name>    [n_search]  out-of-fold prediction: each fold is the query in turn, the other four are the fit rows
  bag_<name>  [n_audit]   mean of those five fold models' predictions of the AUDIT rows
  ref_<name>  [n_audit]   prediction of the AUDIT rows from all SEARCH rows
Only SEARCH targets are used. AUDIT targets are not in the bank. One create-once OUT/<cid>.npz per context (a restart
continues) plus OUT/<cid>.json with what the fit rows contained (best key per fold, its leave-one-out u). The run
starts no new context after --budget-s seconds; a context that exceeds --ctx-s seconds is written with ok = False.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import keys_core as K  # noqa: E402

SEED = 430_000


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def load_bank(path: str):
    b = np.load(path, allow_pickle=False)
    # amendment 2: every array is read from the file once. Indexing the NpzFile inside the loop re-read the whole
    # array for every context, and for a float64 bank each context then kept its own full copy alive.
    search, audit, y64, labs = b["search"], b["audit"], b["y64"], b["lab"]
    s_shape, a_shape = b["s_shape"], b["a_shape"]
    so = ao = lo = 0
    for i, cid in enumerate(b["cids"].tolist()):
        n, w = (int(v) for v in s_shape[i])
        na, f = (int(v) for v in a_shape[i])
        S = np.array(search[so:so + n * w], np.float64).reshape(n, w)
        XA = np.array(audit[ao:ao + na * f], np.float64).reshape(na, f)
        yield {"i": i, "cid": cid, "X": S[:, :-1], "y": np.array(y64[lo:lo + n], np.float64),
               "lab": np.array(labs[lo:lo + n], np.int64), "XA": XA}
        so += n * w; ao += na * f; lo += n


def columns(c: dict) -> tuple:
    X, y, lab, XA = c["X"], c["y"], c["lab"], c["XA"]
    n, na = len(y), len(XA)
    seed = SEED + int(hashlib.sha256(c["cid"].encode()).hexdigest()[:6], 16)
    out = {f"{p}_{k}": np.zeros(n if p == "q" else na) for k in K.NAMES for p in ("q", "bag")}
    info = {"folds": []}
    for k in range(5):
        fi, qi = np.nonzero(lab != k)[0], np.nonzero(lab == k)[0]
        pr, inf = K.heads(X[fi], y[fi], np.vstack([X[qi], XA]), seed + k)
        for name in K.NAMES:
            out[f"q_{name}"][qi] = pr[name][:len(qi)]
            out[f"bag_{name}"] += pr[name][len(qi):] / 5.0
        info["folds"].append(inf)
    pr, inf = K.heads(X, y, XA, seed + 9)
    for name in K.NAMES:
        out[f"ref_{name}"] = pr[name]
    info["refit"] = inf
    sd = float(y.std())
    info["oof_u"] = {name: float(-np.log(max(float(np.sqrt(np.mean((out[f"q_{name}"] - y) ** 2))), 1e-300) / sd))
                     for name in K.NAMES}
    info["direct_folds"] = int(sum(bool(f["direct"]) for f in info["folds"]))
    return out, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--budget-s", type=float, default=1e12)
    ap.add_argument("--ctx-s", type=int, default=900)
    a = ap.parse_args()
    k, m = (int(v) for v in a.shard.split("/"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctxs = sorted(load_bank(a.bank), key=lambda c: (c["X"].shape[0] * c["X"].shape[1], c["cid"]))
    mine = [c for j, c in enumerate(ctxs) if j % m == k]
    t0, done, skipped, failed = time.time(), 0, 0, 0
    signal.signal(signal.SIGALRM, _alarm)
    for c in mine:
        p = out / f"{c['cid']}.npz"
        if p.exists():
            skipped += 1
            continue
        if time.time() - t0 > a.budget_s:
            break
        t1 = time.time()
        signal.alarm(a.ctx_s)
        try:
            cols, info = columns(c)
            ok = True
        except _Timeout:
            cols, info, ok = {}, {"error": "timeout"}, False
        except Exception as e:   # a failed context is recorded and scores the baseline in every arm
            cols, info, ok = {}, {"error": f"{type(e).__name__}: {e}"}, False
        finally:
            signal.alarm(0)
        info.update({"cid": c["cid"], "n_search": int(len(c["y"])), "n_audit": int(len(c["XA"])),
                     "width": int(c["X"].shape[1]), "wall_s": time.time() - t1, "ok": ok})
        tmp = out / f".{c['cid']}.{k}.tmp.npz"
        np.savez(tmp, ok=np.array(ok), wall_s=np.array(time.time() - t1), **cols)
        (out / f"{c['cid']}.json").write_text(json.dumps(info, sort_keys=True))
        tmp.rename(p)
        done += 1
        failed += not ok
    st = {"shard": a.shard, "contexts": len(mine), "written": done, "already_there": skipped, "failed": int(failed),
          "wall_s": time.time() - t0, "bank_sha256": hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(),
          "code_sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ("keys_core.py", "cols_h430.py")}}
    (out / f"cols_status_{k}of{m}_{int(t0)}.json").write_text(json.dumps(st, sort_keys=True))
    print(json.dumps(st))


if __name__ == "__main__":
    main()
