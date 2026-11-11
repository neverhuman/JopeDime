#!/usr/bin/env python3
"""H429: H424's stack column for wide tables, rebuilt with the readout H428 confirmed. CPU inference only.

  cols_h429.py --bank FILE --ckpt CKPT --ckpt-sha256 HEX --size M --out DIR [--shard k/n] [--budget-s S]

For every context of the H424 inference bank with more than WIDE = 30 features, and with the checkpoint's EMA weights,
the three column parts H424 built (q = 5-fold out-of-fold SEARCH prediction on the stored folds; bag = mean of the
five fold-context predictions of the AUDIT rows; ref = prediction of the AUDIT rows from all SEARCH rows), with one
change in every pass: the model no longer sees all features.
  ridge  as in H424: icl_resid.residual_targets on ALL features, with H424's seed of that pass (so the ridge part and
         its residual targets are the ones H424 used);
  model  predicts that residual from each of K = 8 seeded random subsets of 30 features; the 8 predictions are
         averaged and added to the ridge (H428's variant T1).
Narrower tables are not touched (they keep H424's stored column). The model sees SEARCH targets of context rows only;
AUDIT targets are not in the bank. One create-once OUT/<cid>.npz per context in H424's column format; a restart
continues; no new context is started after --budget-s seconds.
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
import icl_resid as RES  # noqa: E402
from icl_model import ICL  # noqa: E402

WIDE, SUB, K = 30, 30, 8
SEED_H424 = 424_700   # cols_h424.SEED: the per-pass seeds of H424's column


def subsets(cid: str, Fn: int):
    out = []
    for j in range(K):
        s = int.from_bytes(hashlib.sha256(f"H429-subset\0{cid}\0{j}".encode()).digest()[:8], "big")
        out.append(torch.from_numpy(np.sort(np.random.default_rng(s).choice(Fn, SUB, replace=False))))
    return out


@torch.no_grad()
def t1(model, x, yy, n_ctx: int, seed: int, subs) -> torch.Tensor:
    yr, used, bq = RES.residual_targets(x[None], yy[None], n_ctx, torch.Generator().manual_seed(seed))
    acc = torch.zeros(x.shape[0] - n_ctx, dtype=torch.float64)
    for S in subs:
        acc += DG.predict(model, x[:, S], yr[0, :n_ctx], n_ctx) / K
    return acc + bq[0].cpu() if bool(used[0]) else acc


def columns(model, S: torch.Tensor, lab: torch.Tensor, XA: torch.Tensor, i: int, cid: str) -> dict:
    n, na = S.shape[0], XA.shape[0]
    X, y = S[:, :-1], S[:, -1]
    subs = subsets(cid, X.shape[1])
    q, bag = torch.zeros(n, dtype=torch.float64), torch.zeros(na, dtype=torch.float64)
    zeros = lambda k: torch.zeros(k, dtype=y.dtype)
    for k in range(5):
        ci, qi = torch.nonzero(lab != k).squeeze(1), torch.nonzero(lab == k).squeeze(1)
        x = torch.cat([X[ci], X[qi], XA])
        yy = torch.cat([y[ci], zeros(len(qi) + na)])
        p = t1(model, x, yy, int(ci.numel()), SEED_H424 + 10 * i + k, subs)
        q[qi] = p[:len(qi)]
        bag += p[len(qi):] / 5.0
    ref = t1(model, torch.cat([X, XA]), torch.cat([y, zeros(na)]), n, SEED_H424 + 10 * i + 9, subs)
    return {"q": q, "bag": bag, "ref": ref}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-sha256", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--budget-s", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0, c0 = time.time(), time.process_time()
    torch.set_num_threads(1)
    random.seed(0)
    torch.manual_seed(0)
    if hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest() != a.ckpt_sha256:
        raise SystemExit("the checkpoint is not the one named by --ckpt-sha256")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ICL(a.size)
    model.load_state_dict(ck["ema"])
    model.eval()
    z = np.load(a.bank, allow_pickle=False)
    if set(z["cohort"].tolist()) != {"DEV"}:
        raise SystemExit("the bank is not the H424 DEV bank")
    ss, sa = z["s_shape"].astype(np.int64), z["a_shape"].astype(np.int64)
    s_off = np.concatenate([[0], np.cumsum(ss[:, 0] * ss[:, 1])])
    a_off = np.concatenate([[0], np.cumsum(sa[:, 0] * sa[:, 1])])
    l_off = np.concatenate([[0], np.cumsum(ss[:, 0])])
    wide = [j for j in range(len(z["cids"])) if ss[j, 1] - 1 > WIDE]
    order = sorted(wide, key=lambda j: (int(ss[j, 0] + sa[j, 0]), str(z["cids"][j])))
    k, n = (int(v) for v in a.shard.split("/"))
    order = order[k::n]
    if a.limit:
        order = order[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    stop = False
    for i in order:
        cid = str(z["cids"][i])
        p = out / f"{cid}.npz"
        if p.exists():
            skipped += 1
            continue
        if a.budget_s and time.time() - t0 > a.budget_s:
            stop = True
            break
        (r, c), (ra, ca) = ss[i].tolist(), sa[i].tolist()
        S = torch.from_numpy(z["search"][s_off[i]:s_off[i + 1]].reshape(r, c).copy())
        XA = torch.from_numpy(z["audit"][a_off[i]:a_off[i + 1]].reshape(ra, ca).copy())
        lab = torch.from_numpy(z["lab"][l_off[i]:l_off[i + 1]].astype(np.int64))
        t1_ = time.time()
        cols = columns(model, S, lab, XA, i, cid)
        ok = all(bool(torch.isfinite(v).all()) for v in cols.values())
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, ok=ok, wall_s=time.time() - t1_, n_search=r, n_audit=ra, features=c - 1, variant="T1",
                 q_ridgeres=cols["q"].numpy(), bag_ridgeres=cols["bag"].numpy(), ref_ridgeres=cols["ref"].numpy())
        tmp.rename(p)
        done += 1
        print(json.dumps({"i": i, "cid": cid[:12], "rows": r, "F": c - 1, "na": ra, "ok": ok, "s": round(time.time() - t1_, 1)}), flush=True)
    st = {"schema": "jope.h429.cols/v1", "bank_sha256": hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(),
          "ckpt_sha256": a.ckpt_sha256, "ckpt_state": ck.get("state"), "size": a.size, "weights": "ema", "threads": 1,
          "device": "cpu", "cuda_visible": bool(torch.cuda.is_available()), "shard": a.shard, "wide_contexts": len(wide),
          "assigned": len(order), "written": done, "already_there": skipped, "budget_stop": stop,
          "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
    (out / f"cols_status_{int(t0)}_s{k}of{n}.json").write_text(json.dumps(st, sort_keys=True))
    print(json.dumps(st))


if __name__ == "__main__":
    main()
