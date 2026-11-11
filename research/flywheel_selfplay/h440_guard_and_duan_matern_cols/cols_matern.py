#!/usr/bin/env python3
"""Wave B (`guard_and_duan_matern_cols`): Sharded CPU column builder in H424/H430 bank format.

  cols_matern.py --bank FILE --out DIR [--shard k/n] [--budget-s S] [--ctx-s 900]

Emits create-once OUT/<cid>.npz with `q_<name>`, `bag_<name>`, `ref_<name>` for all 9 columns in
`guard_matern_core.NAMES` (7 primary W2-W4 columns + 2 plain controls). Uses SEARCH targets only;
AUDIT targets are not in the bank. Reads bank arrays once to keep peak memory small.
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
import guard_matern_core as GM  # noqa: E402

SEED = 434_000


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def load_bank_shard(path: str, k: int, m: int) -> list[dict]:
    with np.load(path, allow_pickle=False) as b:
        s_shape = b["s_shape"]
        a_shape = b["a_shape"]
        cids = b["cids"].tolist()
        meta = []
        so = ao = lo = 0
        for i, cid in enumerate(cids):
            n, w = (int(v) for v in s_shape[i])
            na, f = (int(v) for v in a_shape[i])
            meta.append((n * max(w - 1, 1), str(cid), i, n, w, na, f, so, ao, lo))
            so += n * w
            ao += na * f
            lo += n
        meta.sort(key=lambda t: (t[0], t[1]))
        picked = [t for j, t in enumerate(meta) if j % m == k]
        if not picked:
            return []
        search, audit, y64, labs = b["search"], b["audit"], b["y64"], b["lab"]
        mine = []
        for _, cid, i, n, w, na, f, so, ao, lo in picked:
            S = np.array(search[so:so + n * w], np.float64).reshape(n, w)
            X = np.array(S[:, :-1], dtype=np.float64, copy=True)
            XA = np.array(audit[ao:ao + na * f], np.float64, copy=True).reshape(na, f)
            mine.append({
                "i": i,
                "cid": cid,
                "X": X,
                "y": np.array(y64[lo:lo + n], np.float64, copy=True),
                "lab": np.array(labs[lo:lo + n], np.int64, copy=True),
                "XA": XA,
            })
        return mine


def columns(c: dict) -> tuple[dict[str, np.ndarray], dict]:
    X, y, lab, XA = c["X"], c["y"], c["lab"], c["XA"]
    n, na = len(y), len(XA)
    seed = (SEED + int(hashlib.sha256(c["cid"].encode()).hexdigest()[:6], 16)) & 0x7FFFFFFF
    out = {f"{p}_{k}": np.zeros(n if p == "q" else na, dtype=np.float64) for k in GM.NAMES for p in ("q", "bag")}
    info = {"folds": []}
    for k in range(5):
        fi, qi = np.nonzero(lab != k)[0], np.nonzero(lab == k)[0]
        pr, inf = GM.heads(X[fi], y[fi], np.vstack([X[qi], XA]), seed + k * 101)
        for name in GM.NAMES:
            out[f"q_{name}"][qi] = pr[name][:len(qi)]
            out[f"bag_{name}"] += pr[name][len(qi):] / 5.0
        info["folds"].append(inf)
    pr_ref, inf_ref = GM.heads(X, y, XA, seed + 909)
    for name in GM.NAMES:
        out[f"ref_{name}"] = pr_ref[name]
    info["refit"] = inf_ref
    sd = max(float(y.std()), 1e-12)
    info["oof_u"] = {
        name: float(-np.log(max(float(np.sqrt(np.mean((out[f"q_{name}"] - y) ** 2))) / sd, 1e-12)))
        for name in GM.NAMES
    }
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
    mine = load_bank_shard(a.bank, k, m)
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
        except Exception as e:
            cols, info, ok = {}, {"error": f"{type(e).__name__}: {e}"}, False
        finally:
            signal.alarm(0)
        info.update({
            "cid": c["cid"],
            "n_search": int(len(c["y"])),
            "n_audit": int(len(c["XA"])),
            "width": int(c["X"].shape[1]),
            "wall_s": time.time() - t1,
            "ok": ok,
        })
        tmp = out / f".{c['cid']}.{k}.tmp.npz"
        np.savez(tmp, ok=np.array(ok), wall_s=np.array(time.time() - t1), **cols)
        (out / f"{c['cid']}.json").write_text(json.dumps(info, sort_keys=True))
        tmp.rename(p)
        done += 1
        failed += not ok
    st = {
        "shard": a.shard,
        "contexts": len(mine),
        "written": done,
        "already_there": skipped,
        "failed": int(failed),
        "wall_s": time.time() - t0,
        "bank_sha256": hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(),
        "code_sha256": {
            f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
            for f in ("guard_matern_core.py", "cols_matern.py")
        },
    }
    (out / f"cols_status_{k}of{m}_{int(t0)}.json").write_text(json.dumps(st, sort_keys=True))
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
