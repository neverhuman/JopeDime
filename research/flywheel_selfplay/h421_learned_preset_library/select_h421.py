#!/usr/bin/env python3
"""H421 steps 3-4 - selection and the TRAIN-side gate (PREREG section 3.3-3.4, addendum 1). Offline: numpy only, no
worker run, design contexts only, SEARCH out-of-fold predictions only.

Per design context c the fill gives 2-fold out-of-fold SEARCH predictions of
  H_c   the context's stored stack candidates (what the stack of c already holds), and
  the pool candidates (library presets of B* and harvested programs).
A preset gets stack weight only if it enters the context's stack candidate set, so the deploy's own rule
(deploy_h415.hf_candidates: HF_TOP 48, HF_PER_HEAD 2, HF_CAP 64) is applied to H_c plus the presets under test:
  top block   the 48 best search fitnesses; a preset enters it when its fitness beats the 48th of H_c;
  additions   of the rest, heads that are not in the top block, at most 2 programs per head (best fitness first),
              then the 16 best by fitness. Stored additions and presets compete for these slots.
The presets under test are B*'s LIB_EXT presets plus the candidate set S (completed, search gate passed).

u_c(S) = -ln(RMSE / RMSE of the fold-mean predictor) of the Caruana blend (meta_h416.kcar, 100 steps) over the
         resulting candidate set of c.                                      Objective: family mean of u_c(S).
LIB_PORT  greedy forward selection of K = 20 harvested candidates for that objective (ties: lower pool id).
LIB_RAND  20 harvested candidates drawn at random (seed 421101).
LIB_COV   greedy for coverage: the family mean of the best single column's utility.
Gate      two-fold family cross-fit: select on one half, score u_c(LIB_PORT_half) - u_c(LIB_RAND) on the other,
          both directions; passes when the family mean is above 0 with one-sided sign-flip p < 0.05.

  select_h421.py --fill-dir R/fill_npz --pool R/candidates.json --out-dir DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

K_SELECT = 20
STEPS = 100
TOP_BLOCK, PER_HEAD, CAP = 48, 2, 64   # deploy_h415.HF_TOP, HF_PER_HEAD, HF_CAP
RAND_SEED = 421101
FLIP_SEED, N_FLIPS = 421102, 100_000


def gram_kcar(G: np.ndarray, c: np.ndarray, steps: int = STEPS) -> np.ndarray:
    """meta_h416.kcar on Gram form: G = Q'Q, c = Q'y. Step t picks argmin_j ||(s + q_j) / t - y||, i.e.
    argmin_j G_jj + 2 s.q_j - 2 t c_j; ties -> the lowest index, as numpy argmin there."""
    cnt = np.zeros(len(c))
    sq = np.zeros(len(c))
    d = np.diag(G).copy()
    for t in range(1, steps + 1):
        j = int(np.argmin(d + 2.0 * sq - 2.0 * t * c))
        cnt[j] += 1
        sq += G[:, j]
    return cnt


class Ctx:
    def __init__(self, path: Path, pool):
        z = np.load(path, allow_pickle=False)
        y = z["y"].astype(np.float64)
        n, n0 = len(y), int(z["n0"])
        self.base2 = (n0 * float(z["b0"]) ** 2 + (n - n0) * float(z["b1"]) ** 2) / n
        hf, hg, hh = z["h_fitness"], z["h_gate"], z["h_head"]
        order = sorted(np.flatnonzero(hg), key=lambda j: (hf[j], j))
        top = order[:TOP_BLOCK]
        self.thr = hf[top[-1]] if len(top) >= TOP_BLOCK else math.inf      # the 48th-best fitness of H_c
        self.top_cols = [int(j) for j in top]
        self.top_heads = {str(hh[j]) for j in top}
        self.h_adds = [(float(hf[j]), str(hh[j]), int(j)) for j in order[TOP_BLOCK:]]   # stored additions
        A = np.column_stack([z["PH"].astype(np.float64), z["P"].astype(np.float64)])
        self.n, self.nh = n, z["PH"].shape[1]
        self.G, self.c, self.yy = A.T @ A, A.T @ y, float(y @ y)
        self.col = {int(i): self.nh + k for k, i in enumerate(z["cand"])}   # pool id -> column
        fit, gate, head, sha = z["fitness"], z["gate"], z["head"], z["sha"]
        held = {str(x) for x in z["h_sha"]}   # a preset that is already a stored stack candidate changes nothing
        self.cand = {i: (float(fit[i]), str(head[i])) for i in self.col if gate[i] and str(sha[i]) not in held}
        self.ext = [e["id"] for e in pool if e["source"] == "lib_ext" and e["id"] in self.cand]
        self.elig = set(self.cand)
        self.status = z["status"]
        self.h_missing = int(z["h_missing"])
        self._memo, self._u = {}, {}

    def cols_with(self, sel):
        """The stack candidate columns of this context when the presets LIB_EXT + sel are added to the library."""
        key = tuple(sorted(i for i in sel if i in self.cand))
        got = self._memo.get(key)
        if got is not None:
            return got
        cols, adds = list(self.top_cols), list(self.h_adds)
        for i in list(self.ext) + list(key):
            f, h = self.cand[i]
            if f < self.thr:
                cols.append(self.col[i])            # top block
            elif h not in self.top_heads:
                adds.append((f, h, self.col[i]))    # competes for an addition slot
        per, kept = {}, []
        for f, h, j in sorted(adds, key=lambda t: (t[0], t[2])):
            if per.get(h, 0) < PER_HEAD:
                per[h] = per.get(h, 0) + 1
                kept.append(j)
        cols += kept[: max(0, CAP - TOP_BLOCK)]
        if len(self._memo) < 4096:
            self._memo[key] = cols
        return cols

    def u_blend(self, cols) -> float:
        key = tuple(cols)
        got = self._u.get(key)
        if got is not None:
            return got
        cols = np.asarray(cols, np.int64)
        if len(cols) == 0:
            return 0.0
        G, c = self.G[np.ix_(cols, cols)], self.c[cols]
        cnt = gram_kcar(G, c)
        w = cnt / STEPS
        mse = (w @ G @ w - 2.0 * w @ c + self.yy) / self.n
        u = -0.5 * math.log(max(mse, 1e-300) / self.base2)
        if len(self._u) < 20000:
            self._u[key] = u
        return u

    def u_best_single(self, cols) -> float:
        cols = np.asarray(cols, np.int64)
        if len(cols) == 0:
            return 0.0
        mse = (np.diag(self.G)[cols] - 2.0 * self.c[cols] + self.yy) / self.n
        return -0.5 * math.log(max(float(mse.min()), 1e-300) / self.base2)


def greedy(ctxs, cand_ids, k: int, coverage: bool = False):
    """Forward selection over the contexts in ctxs (dict cid -> Ctx). Returns (chosen ids, objective trace)."""
    f = (lambda c, cols: c.u_best_single(cols)) if coverage else (lambda c, cols: c.u_blend(cols))
    cur = {cid: f(c, c.cols_with([])) for cid, c in ctxs.items()}
    chosen, trace = [], [float(np.mean(list(cur.values())))]
    for _ in range(k):
        best, best_obj, best_vals = None, -math.inf, None
        for i in cand_ids:
            if i in chosen:
                continue
            vals = {cid: (f(c, c.cols_with(chosen + [i])) if i in c.elig else cur[cid]) for cid, c in ctxs.items()}
            obj = float(np.mean(list(vals.values())))
            if obj > best_obj + 1e-15:
                best, best_obj, best_vals = i, obj, vals
        if best is None:
            break
        chosen.append(best); cur = best_vals; trace.append(best_obj)
    return chosen, trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fill-dir", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-missing", type=int, default=0, help="design contexts allowed to lack a fill record")
    a = ap.parse_args()
    import apex_core_meta as AM
    import meta_h416 as M

    raw = Path(a.pool).read_bytes()
    doc = json.loads(raw)
    pool, fam_of = doc["pool"], doc["family_of"]
    closed = set(AM.cohort("DEV", "all")) | set(AM.cohort("CONFIRM", "all"))
    if set(doc["cids"]) & closed:
        raise SystemExit("a DEV or CONFIRM context is among the design contexts")
    harvest = [e["id"] for e in pool if e["source"] == "harvest"]
    ctxs, absent = {}, []
    for cid in doc["cids"]:
        p = Path(a.fill_dir) / f"{cid}.npz"
        if p.is_file():
            ctxs[cid] = Ctx(p, pool)
        else:
            absent.append(cid)
    if len(absent) > a.allow_missing:
        raise SystemExit(f"{len(absent)} design contexts have no fill record")
    # self-test: the Gram form equals meta_h416.kcar on the stored predictions of 5 contexts
    for cid in list(ctxs)[:5]:
        z = np.load(Path(a.fill_dir) / f"{cid}.npz", allow_pickle=False)
        Q = np.column_stack([z["PH"].astype(np.float64), z["P"].astype(np.float64)])
        if Q.shape[1]:   # compare the blends, not the counts: duplicate columns may break ties differently
            yv = z["y"].astype(np.float64)
            r1 = float(np.sqrt(np.mean((Q @ (M.kcar(Q, yv, STEPS) / STEPS) - yv) ** 2)))
            r2 = float(np.sqrt(np.mean((Q @ (gram_kcar(Q.T @ Q, Q.T @ yv) / STEPS) - yv) ** 2)))
            if abs(r1 - r2) > 1e-9 * max(r1, 1e-300):
                raise SystemExit(f"{cid}: the Gram Caruana differs from meta_h416.kcar ({r1} vs {r2})")
    fams = sorted({fam_of[c] for c in ctxs})
    if len(fams) != len(ctxs):
        raise SystemExit("a family has more than one design context; the family mean must be coded for that case")
    half = {f: int(hashlib.sha256(("H421-gate\0" + f).encode()).hexdigest(), 16) % 2 for f in fams}
    rng = np.random.default_rng(RAND_SEED)
    rand = sorted(int(i) for i in rng.choice(harvest, size=K_SELECT, replace=False))
    diffs = {}
    cross = []
    for h in (0, 1):
        fit = {c: x for c, x in ctxs.items() if half[fam_of[c]] == h}
        hold = {c: x for c, x in ctxs.items() if half[fam_of[c]] != h}
        sel, tr = greedy(fit, harvest, K_SELECT)
        for c, x in hold.items():
            diffs[c] = x.u_blend(x.cols_with(sel)) - x.u_blend(x.cols_with(rand))
        cross.append({"fit_half": h, "fit_contexts": len(fit), "held_out": len(hold), "selected": sel,
                      "fit_objective": [tr[0], tr[-1]],
                      "held_out_mean_port": float(np.mean([x.u_blend(x.cols_with(sel)) for x in hold.values()])),
                      "held_out_mean_rand": float(np.mean([x.u_blend(x.cols_with(rand)) for x in hold.values()])),
                      "held_out_mean_base": float(np.mean([x.u_blend(x.cols_with([])) for x in hold.values()]))})
    d = np.array([diffs[c] for c in sorted(diffs)])
    flips = np.random.default_rng(FLIP_SEED).choice([-1.0, 1.0], size=(N_FLIPS, len(d)))
    p = float((np.sum((flips @ d) / len(d) >= d.mean() - 1e-15) + 1) / (N_FLIPS + 1))
    gate = {"n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)), "win": int((d > 0).sum()),
            "tie": int((d == 0).sum()), "loss": int((d < 0).sum()), "signflip_p_one_sided": p,
            "pass": bool(d.mean() > 0 and p < 0.05), "cross_fit": cross}
    port, tr_port = greedy(ctxs, harvest, K_SELECT)
    cov, tr_cov = greedy(ctxs, harvest, K_SELECT, coverage=True)
    by_id = {e["id"]: e for e in pool}
    u = lambda sel: float(np.mean([x.u_blend(x.cols_with(sel)) for x in ctxs.values()]))
    pairs = sum(len(x.col) for x in ctxs.values())
    out = {"schema": "jope.h421.selection/v1", "pool_sha256": hashlib.sha256(raw).hexdigest(),
           "design_contexts": len(ctxs), "contexts_without_fill": absent, "families": len(fams),
           "harvest_candidates": len(harvest), "k": K_SELECT,
           "fill": {"completed_pairs": pairs, "usable_pairs": sum(len(x.elig) for x in ctxs.values()),
                    "failed_pairs": int(sum((x.status == 1).sum() for x in ctxs.values())),
                    "not_valid_pairs": int(sum((x.status == 2).sum() for x in ctxs.values())),
                    "stored_candidates_missing": int(sum(x.h_missing for x in ctxs.values()))},
           "objective": {"base": u([]), "port": u(port), "rand": u(rand), "cov": u(cov),
                         "trace_port": tr_port, "trace_cov_best_single": tr_cov},
           "gate": gate,
           "sets": {k: [{"id": i, "name": by_id[i]["name"], "rank": by_id[i].get("rank"),
                         "n_families": by_id[i]["n_families"], "dsl": by_id[i]["dsl"],
                         "usable_in": sum(1 for x in ctxs.values() if i in x.elig),
                         "enters_alone_in": sum(1 for x in ctxs.values() if i in x.elig
                                                and x.col[i] in x.cols_with([i]))} for i in v]
                    for k, v in (("LIB_PORT", port), ("LIB_RAND", rand), ("LIB_COV", cov))}}
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    for k, pre in (("LIB_PORT", "port"), ("LIB_RAND", "rand"), ("LIB_COV", "cov")):
        b = json.dumps([[f"{pre}_{e['name']}", e["dsl"]] for e in out["sets"][k]], indent=1).encode()
        with open(od / f"{k}.json", "xb") as f:
            f.write(b)
        out[f"{k}_sha256"] = hashlib.sha256(b).hexdigest()
    with open(od / "selection.json", "x") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps({"gate": {k: gate[k] for k in ("n", "mean", "median", "win", "tie", "loss", "signflip_p_one_sided", "pass")},
                      "objective": {k: out["objective"][k] for k in ("base", "port", "rand", "cov")}, "fill": out["fill"],
                      "overlap_port_cov": len(set(port) & set(cov)), "overlap_port_rand": len(set(port) & set(rand))}))


if __name__ == "__main__":
    main()
