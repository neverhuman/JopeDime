"""H94 — W6 prospective test of H92's HEDGE_CTX on NEW programs (CPU, native training-CV fits).

H92 found, on cached development pools, that hedging W1's context model (keep the program-only prior's top-2, add the
context model's top-2) clears the common gate (+0.036 U4, +50.5 Elo). That was adaptive development on the same
families. H94 tests the rule on programs that were never evaluated before.

Fixed before any H94 outcome:
* Roster: the frozen H67 roster (50 DL-Train contexts / 44 families; every family inside one W1 fold), seeds 701-703.
  Per context and seed, 16 NEW proposals (not in the pool) from the pinned live proposer (H81's generator, own stream).
* Scorers: W1-recipe refits per W1 outer fold, both trained identically except for features: PRIOR = program-only
  (prog412, random_state 700+k), CTX = context-aware (prog412 + lm48_z + dope629, random_state 800+k; the H81 refit
  models.pkl, sha256 c8bfbe94...). A context is scored only by the model of the fold that excluded its whole family.
  Using two refits makes the comparison internally consistent; W1's lost fitted models are not needed.
* Arms (after the four hash-ordered anchors, whose pool utilities are known history; each arm reveals four new programs
  by native original-cut training CV; ties by program SHA): PRIOR_top4, CTX_top4, HEDGE_CTX (PRIOR top-2 + CTX top-2
  among the rest). Endpoint: retained-best utility over anchors + the arm's four new evaluations. An execution error makes
  that arm INCOMPLETE; incomplete context-seeds leave the paired panel and are counted.
* Primary: HEDGE_CTX - PRIOR_top4 with the common gate (one-sided .05 LCB > +.001, median >= 0, <= 10% of families
  losing > .01). Secondary: CTX_top4 - PRIOR_top4, HEDGE_CTX - CTX_top4. Family Elo (PRIOR_top4 = 1500); family-bootstrap
  median and mean 95% intervals (10,000 draws, seed 20261031).
* Consistency guard (before any proposal): on the roster's admitted cached pools, the refits must reproduce a positive
  family-mean HEDGE_CTX - PRIOR (H92's direction). Otherwise the run stops and reports.
* Budget: at most 1,200 native fits (W6 ceiling; the union of the three arms is <= 8 fits per context-seed), deadline
  2026-10-31T16:00Z, xbabe2 only (x86-64 evaluator), nice 15, load-gated; zero GPU.
No DL-Val, outer-test, sealed or holdout row is read; DOPE is not loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
H81_PATH = HERE.parent / "h81_w4_prospective_2plus2" / "run_prospective.py"
H81_SHA = "9ad50be577a30c616f7668d52f621d5411e70019ba9f1d1960eb8a45a448c999"  # merged H81 runner (helpers only: pins, inputs, native fit, ledger)
CTX_MODELS_SHA = "c8bfbe94f74d6ffe3f40b9848e43e22e832ada8672828d31f8ada647ee3096a1"
ARMS = ("PRIOR_top4", "CTX_top4", "HEDGE_CTX")
ANCHOR = "PRIOR_top4"
PRIMARY = ("HEDGE_CTX", "PRIOR_top4")
SECONDARY = (("CTX_top4", "PRIOR_top4"), ("HEDGE_CTX", "CTX_top4"))
FIT_CAP = 1200
DEADLINE = "2026-10-31T16:00:00Z"
MIN_PROPOSALS = 4


def load_h81():
    src = H81_PATH.read_bytes()
    if hashlib.sha256(src).hexdigest() != H81_SHA:
        raise SystemExit("H81 runner bytes differ from the merged source")
    spec = importlib.util.spec_from_file_location("h81_run", H81_PATH)
    m = importlib.util.module_from_spec(spec); sys.modules["h81_run"] = m; spec.loader.exec_module(m)
    return m


def picks(arm: str, mu_prog: np.ndarray, mu_ctx: np.ndarray, ids: list[str], ranked) -> list[int]:
    if arm == "PRIOR_top4":
        return ranked(mu_prog, ids)[:4]
    if arm == "CTX_top4":
        return ranked(mu_ctx, ids)[:4]
    if arm == "HEDGE_CTX":
        keep = ranked(mu_prog, ids)[:2]
        return keep + [i for i in ranked(mu_ctx, ids) if i not in keep][:2]
    raise ValueError(arm)


def fit_prog_models(h81, out: Path, threads: int) -> dict:
    """W1 recipe, program-only variant: same slot sampling stream as the context refit (default_rng(700) over
    selected.json train order), random_state 700+k."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    train, fold_by_cid, _, _ = h81.load_inputs()
    rng = np.random.default_rng(700)
    xp, y, fk = [], [], []
    for r in train:
        c = r["context_sha256"]; p = h81.pool(c)
        take = min(192, len(p["u"]) - 4)
        samp = 4 + rng.choice(len(p["u"]) - 4, size=take, replace=False)
        xp.append(p["program"][samp]); y.append((p["u"][samp] - p["u"].mean()).astype(np.float32)); fk.append(np.full(take, fold_by_cid[c]))
    X, Y, FK = np.concatenate(xp), np.concatenate(y), np.concatenate(fk)
    models = {}
    with threadpool_limits(limits=threads):
        for k in range(5):
            models[k] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
                                                      random_state=700 + k).fit(X[FK != k], Y[FK != k])
            print("prog model", k, flush=True)
    with open(out / "models_prog.pkl", "wb") as f:
        pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
    return models


def cached_guard(h81, prog: dict, ctx: dict) -> dict:
    """Refits on the roster's admitted cached pools: does HEDGE_CTX beat PRIOR on average (H92's direction)?"""
    train, fold_by_cid, ctxf, acq = h81.load_inputs()
    idx = {r["context_sha256"]: i for i, r in enumerate(train)}
    fam = {r["context_sha256"]: r["family_id"] for r in train}
    per = defaultdict(lambda: defaultdict(list))
    for c in acq:
        p = h81.pool(c); k = fold_by_cid[c]
        un = np.arange(4, len(p["u"])); ids = [p["sha"][i] for i in un]
        mp = prog[k].predict(p["program"][un])
        mc = ctx[k].predict(np.concatenate([p["program"][un], np.repeat(ctxf[idx[c]][None], len(un), 0)], axis=1))
        for arm in ARMS:
            sel = picks(arm, mp, mc, ids, h81.ranked)
            per[fam[c]][arm].append(float(max(np.max(p["u"][:4]), np.max(p["u"][un[sel]]))))
    fm = {a: mean(mean(v[a]) for v in per.values()) for a in ARMS}
    return dict(contexts=len(acq), families=len(per), cached_family_mean_U4=fm,
                hedge_minus_prior=fm["HEDGE_CTX"] - fm["PRIOR_top4"], passed=bool(fm["HEDGE_CTX"] - fm["PRIOR_top4"] > 0))


def run(h81, out: Path, run_id: str, prog: dict, ctx: dict, workers: int):
    sys.path.insert(0, str(h81.MICRO / "python"))
    from rust_evaluator import featurize_dsl_412, propose_flywheel_candidates
    train, fold_by_cid, ctxf, acq = h81.load_inputs()
    idx = {r["context_sha256"]: i for i, r in enumerate(train)}
    fam = {r["context_sha256"]: r["family_id"] for r in train}
    champ = defaultdict(list)
    for r in json.loads((h81.INPUT / "champions.json").read_text()):
        champ[r["context_sha256"]].append({"dsl": r["dsl"], "program_sha256": r["program_sha256"]})
    bank = [r for r in json.loads((Path.home() / "jope-data/h58/cache/champions_featurized.json").read_text())
            if r["dl_split"] == "DL-Train" and r["context_sha256"] in idx]
    lm = np.load(h81.W1 / "cache/train_only_context_features.npz", allow_pickle=False)["lm48_z"].astype(np.float64)
    ledger, store = h81.Ledger(out / "ledger.json", cap=FIT_CAP, deadline=DEADLINE), h81.FitStore(out / "fits.jsonl", run_id)
    units, shortfalls = [], []
    for c in acq:
        p = h81.pool(c); k = fold_by_cid[c]; known = set(p["sha"])
        for seed in h81.SEEDS:
            props, seen, batch = [], set(known), 0
            while len(props) < h81.N_PROPOSALS and batch < 50:
                for q in propose_flywheel_candidates(champ[c], lm[idx[c]], bank, fam[c], n_proposals=16,
                                                     seed=h81.hseed("h94-stream", c, seed, batch) % (2 ** 31)):
                    if q["program_sha256"] not in seen and len(props) < h81.N_PROPOSALS:
                        seen.add(q["program_sha256"]); props.append(q)
                batch += 1
            if len(props) < MIN_PROPOSALS:
                shortfalls.append(dict(context=c, seed=seed, proposals=len(props))); continue
            t0 = time.process_time()
            feats = np.stack([np.asarray(featurize_dsl_412(q["dsl"])[0], dtype=np.float32) for q in props])
            mp = prog[k].predict(feats)
            mc = ctx[k].predict(np.concatenate([feats, np.repeat(ctxf[idx[c]][None], len(props), 0)], axis=1))
            ids = [q["program_sha256"] for q in props]
            sel = {a: picks(a, mp, mc, ids, h81.ranked) for a in ARMS}
            units.append(dict(context=c, seed=seed, props=props, ids=ids, sel=sel, anchors_best=float(np.max(p["u"][:4])),
                              pool_best_fitness=float(np.min(p["fitness"][p["gate"]])) if p["gate"].any() else math.inf,
                              selection_cpu_s=time.process_time() - t0))
    res = store.run([(u["context"], u["props"][i]["dsl"]) for u in units for a in ARMS for i in u["sel"][a]], out / "scratch", ledger, workers)
    rows, hwm = [], {}
    for u in units:
        for a in ARMS:
            fits = [res[(u["context"], u["ids"][i])] for i in u["sel"][a]]
            vals = [h81.utility(f["status"], f["gate"], f["fitness"]) for f in fits]
            ok = all(v is not None for v in vals)
            rows.append(dict(context=u["context"], family=fam[u["context"]], seed=u["seed"], arm=a,
                             status="completed" if ok else "incomplete", retained_best=max([u["anchors_best"]] + vals) if ok else None,
                             new_best=max(vals) if ok else None, revealed=[u["ids"][i] for i in u["sel"][a]], revealed_utility=vals,
                             native_wall_s=sum(f["wall_s"] for f in fits), native_cpu_s=sum(f["cpu_s"] for f in fits),
                             selection_cpu_s=u["selection_cpu_s"]))
            for i, f in zip(u["sel"][a], fits):
                if f["status"] == "completed" and f["gate"] and f["fitness"] < u["pool_best_fitness"] - h81.MARGIN:
                    hwm.setdefault((u["context"], u["ids"][i]), dict(context_sha256=u["context"], family_id=fam[u["context"]],
                                                                    program_sha256=u["ids"][i], dsl=u["props"][i]["dsl"],
                                                                    fitness_original_cut=f["fitness"], pool_best_fitness=u["pool_best_fitness"],
                                                                    found_by=a, seed=u["seed"], status="apparent (single original cut)"))
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    (out / "apparent_hwm.json").write_text(json.dumps(list(hwm.values()), indent=1))
    (out / "shortfalls.json").write_text(json.dumps(shortfalls, indent=1))
    return rows, shortfalls, list(hwm.values()), ledger.state


def analyze(h81, rows: list[dict], elo=None) -> dict:
    fam = {r["context"]: r["family"] for r in rows}
    by = defaultdict(lambda: defaultdict(list)); bad = set()
    for r in rows:
        if r["status"] != "completed":
            bad.add((r["context"], r["seed"]))
    for r in rows:
        if (r["context"], r["seed"]) not in bad:
            by[r["context"]][r["arm"]].append(r["retained_best"])
    ctx_means = {c: {a: mean(v[a]) for a in ARMS} for c, v in by.items() if all(v.get(a) for a in ARMS)}
    families = sorted({fam[c] for c in ctx_means})
    fv = {f: {a: mean(ctx_means[c][a] for c in ctx_means if fam[c] == f) for a in ARMS} for f in families}
    draws = np.random.default_rng(h81.BOOTSTRAP_SEED).integers(0, len(families), size=(h81.BOOTSTRAP_DRAWS, len(families)))

    def con(a, b):
        v = np.array([fv[f][a] - fv[f][b] for f in families]); c = h81.contrast(v, draws)
        med = np.median(v[draws], axis=1)
        c["median_ci95"] = [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]
        c["exact_tie_fraction"] = float(np.mean(np.abs(v) <= h81.TIE))
        return c
    out = dict(contexts=len(ctx_means), families=len(families), incomplete_context_seeds=len(bad),
               family_mean_retained_best={a: mean(fv[f][a] for f in families) for a in ARMS},
               primary=dict(contrast=f"{PRIMARY[0]} - {PRIMARY[1]}", **con(*PRIMARY)),
               secondary={f"{a} - {b}": con(a, b) for a, b in SECONDARY},
               elapsed_per_arm={a: dict(native_wall_s=sum(r["native_wall_s"] for r in rows if r["arm"] == a),
                                        native_cpu_s=sum(r["native_cpu_s"] for r in rows if r["arm"] == a),
                                        selection_cpu_s=sum(r["selection_cpu_s"] for r in rows if r["arm"] == a)) for a in ARMS})
    if elo is not None:
        recs = [elo.DatasetEvalRecord(c, fam[c], "regression", v) for c, v in sorted(ctx_means.items())]
        out["family_elo"] = elo.compute_family_grouped_bradley_terry_elo(recs, anchor_contender=ANCHOR, anchor_elo=1500.0,
                                                                         tie_margin=h81.TIE, prior=0.01, n_iters=400)
    out["verdict"] = ("ADVANCE_PROSPECTIVE (primary passes the common gate)" if out["primary"]["common_gate_passed"]
                      else "NO_ADVANCE (primary gate not met)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--ctx-models", type=Path, required=True, help="H81 refit models.pkl (sha256 c8bfbe94...)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    h81 = load_h81()
    h81.verify_pins()
    if h81.sha256_file(a.ctx_models) != CTX_MODELS_SHA:
        raise SystemExit("context refit models.pkl hash mismatch")
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.ctx_models, "rb") as f:
        ctx = pickle.load(f)
    prog_path = a.out / "models_prog.pkl"
    if prog_path.exists():
        with open(prog_path, "rb") as f:
            prog = pickle.load(f)
    else:
        prog = fit_prog_models(h81, a.out, a.threads)
    guard = cached_guard(h81, prog, ctx)
    guard["models_prog_sha256"] = h81.sha256_file(prog_path)
    (a.out / "guard.json").write_text(json.dumps(guard, indent=1))
    print("guard", json.dumps(guard), flush=True)
    if not guard["passed"]:
        raise SystemExit("consistency guard failed: refits do not reproduce H92's cached direction; no proposals scored")
    rows, shortfalls, hwm, ledger = run(h81, a.out, a.run_id, prog, ctx, a.workers)
    res = analyze(h81, rows, h81.load_elo())
    res.update(run_id=a.run_id, guard=guard, shortfalls=len(shortfalls), apparent_hwm=len(hwm), ledger=ledger,
               source_sha256=h81.sha256_file(Path(__file__)), h81_source_sha256=h81.sha256_file(H81_PATH),
               ctx_models_sha256=CTX_MODELS_SHA, gpu_seconds=0)
    (a.out / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("contexts", "families", "family_mean_retained_best", "primary", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
