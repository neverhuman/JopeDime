"""H112 — W4 active reveal (two-plus-two) re-tested with the H110 proposal-distribution scorer as the prior: CPU harness.

W4 asks whether choosing the second pair from the first pair's outcomes beats a fixed top four at four total fits.
H103 tested it with the cached-pool program prior and found NO_ADVANCE: that prior had no signal among new proposals
(it tied random4), so the update had nothing to refine. H110 then found a scorer trained on native reveals of proposer
programs that beats random4 by +269 Elo. H112 re-runs W4 with that scorer as the prior. The choice follows H103/H110
outcomes on the same roster and is disclosed as adaptive; no H112 outcome has been seen.

Preregistered here (PREREG.md is the prose copy; the frozen source is authoritative):
* Memory (training data for the prior): the H106 memory (H94 + H103 reveal logs, recovered by stream replay through the
  merged H106 module) plus every H110 native fit (fits.jsonl sha256 239b9d26..., all 16 proposals of each roster
  context x seeds 801-803, recovered by replaying H110's stream). Rows de-duplicated per (context, program).
* Prior: the H106 learner recipe (features [prog412 - champion, prog412], target = within-context utility percentile,
  HistGBM 200 / .05 / 15 leaves / l2 1, random_state 1120+k), one model per W1 outer fold k trained on contexts outside
  fold k. Utility scale: mu = b_k * score, with b_k the OLS slope of clipped (+-1) within-context-centred utility on
  within-context-centred out-of-fold score inside the training folds (inner folds j != k). No context intercept: the
  update below is level-free.
* Arms (zero history, four reveals each, ties by program SHA): s_top4 (comparator, Elo anchor); s_2plus2 (reveal the
  top two; residuals r = clip(u - mu, +-1) centred across the pair; RBF-GP posterior mean on proj16 with the merged
  H81 posterior_mean and lengthscale rule, noise ratio 1; reveal the top two of the rest); s_2plus2_swap (placebo: the
  two centred residuals exchanged); random4 (h112-random).
* Programs: H67 roster x fresh seeds 901-903, 16 new proposals each (stream "h112-stream", admitted pool excluded).
  All 16 are evaluated natively, so every arm is scored on a fully labelled batch; the first pair is revealed before
  the adaptive second pair is chosen, and every pick is fixed before the remaining fits run.
* Endpoint: best utility among an arm's four picks. Seeds averaged within context, contexts within family.
* Primary: s_2plus2 - s_top4 under the common gate (one-sided .05 LCB > +.001, median >= 0, <= 10% of families losing
  > .01), 10,000-draw family bootstrap (seed 20261031). ADVANCE also requires s_2plus2 - s_2plus2_swap > 0 in family
  mean. Secondary: s_top4 - random4 (H110 A replication on fresh seeds). Family Elo, s_top4 = 1500. Completeness gate.
* Budget: 2,400 native fits, xbabe2, nice 15, load-gated; deadline 2026-11-01T18:00Z. Zero GPU.
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
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
H106_PATH = HERE.parent / "h106_w9_proposal_scorer_replay" / "run_h106.py"
H106_SHA = "13730ebfdf1a451157799a7d4a639ccae61ed88361044006adb0ff548978f7da"
H110_FITS_SHA = "239b9d265e2cfe30ba48f42085f12009c497e04ba5eb0a695a2fe2689e67c00f"
H110_SEEDS, H110_STREAM = (801, 802, 803), "h106-stream"
ARMS = ("s_top4", "s_2plus2", "s_2plus2_swap", "random4")
ANCHOR = "s_top4"
PRIMARY = ("s_2plus2", "s_top4")
SECONDARY = (("s_2plus2", "s_2plus2_swap"), ("s_top4", "random4"))
EVAL_SEEDS = (901, 902, 903)
STREAM = "h112-stream"
CLIP = 1.0
FIT_CAP = 2400
DEADLINE = "2026-11-01T18:00:00Z"


def load_h106():
    if hashlib.sha256(H106_PATH.read_bytes()).hexdigest() != H106_SHA:
        raise SystemExit("H106 runner bytes differ from the merged source")
    spec = importlib.util.spec_from_file_location("h106_run", H106_PATH)
    m = importlib.util.module_from_spec(spec); sys.modules["h106_run"] = m; spec.loader.exec_module(m)
    return m


def h110_memory(h81, prop, h110_dir: Path) -> list[dict]:
    path = h110_dir / "fits.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != H110_FITS_SHA:
        raise SystemExit("H110 fits.jsonl differs from the published run")
    fits = {(r["context"], r["program_sha256"]): r for r in map(json.loads, path.read_text().splitlines())}
    rows, missing = {}, 0
    for c in prop.roster:
        for seed in H110_SEEDS:
            for q in prop.batch(c, seed, H110_STREAM):
                f = fits.get((c, q["program_sha256"]))
                if f is None:
                    missing += 1; continue
                u = h81.utility(f["status"], f["gate"], f["fitness"])
                if u is not None:
                    rows.setdefault((c, q["program_sha256"]), dict(source="H110", context=c, family=prop.fam[c], fold=prop.fold[c],
                                                                   program_sha256=q["program_sha256"], dsl=q["dsl"], utility=u))
    if missing:
        raise SystemExit(f"{missing} replayed H110 proposals have no logged fit")
    return list(rows.values())


def full_memory(h106, h81, prop, h94_dir, h103_dir, h110_dir) -> list[dict]:
    out = {}
    for r in h106.build_memory(prop, {"h94": h94_dir, "h103": h103_dir}) + h110_memory(h81, prop, h110_dir):
        key = (r["context"], r["program_sha256"])
        if key in out and abs(out[key]["utility"] - r["utility"]) > 1e-9:
            raise SystemExit(f"inconsistent utilities for one program in one context: {key}")
        out.setdefault(key, dict(r, source="MEMORY"))
    return sorted(out.values(), key=lambda r: (r["context"], r["program_sha256"]))


def fit_gbm(h106, prop, rows: list[dict], seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    by = defaultdict(list)
    for r in rows:
        by[r["context"]].append(r)
    X, Y = [], []
    for c, g in sorted(by.items()):
        X.append(prop.x(c, [r["dsl"] for r in g])); Y.append(h106.percentile_target(np.array([r["utility"] for r in g])))
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                         random_state=seed).fit(np.concatenate(X), np.concatenate(Y))


def centred(values: np.ndarray, groups: list[str]) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    g = np.asarray(groups)
    for c in set(groups):
        m = g == c
        out[m] -= out[m].mean()
    return out


def ols_slope(score: np.ndarray, utility: np.ndarray, groups: list[str]) -> float:
    x, y = centred(score, groups), np.clip(centred(utility, groups), -CLIP, CLIP)
    den = float(np.dot(x, x))
    return float(np.dot(x, y) / den) if den > 0 else 0.0


def fit_priors(h106, prop, memory: list[dict], threads: int = 8) -> dict:
    """Per outer fold k: the scorer on folds != k and an out-of-fold utility-scale slope from inner folds j != k."""
    from threadpoolctl import threadpool_limits
    models, slopes, receipt = {}, {}, {}
    with threadpool_limits(limits=threads):
        for k in range(5):
            train = [r for r in memory if r["fold"] != k]
            models[k] = fit_gbm(h106, prop, train, 1120 + k)
            sc, ut, gr = [], [], []
            for j in sorted({r["fold"] for r in train}):
                inner = fit_gbm(h106, prop, [r for r in train if r["fold"] != j], 1130 + 10 * k + j)
                held = [r for r in train if r["fold"] == j]
                by = defaultdict(list)
                for r in held:
                    by[r["context"]].append(r)
                for c, g in by.items():
                    sc.extend(inner.predict(prop.x(c, [r["dsl"] for r in g]))); ut.extend(r["utility"] for r in g); gr.extend([c] * len(g))
            slopes[k] = ols_slope(np.array(sc), np.array(ut), gr)
            receipt[f"fold{k}"] = dict(train_rows=len(train), train_contexts=len({r["context"] for r in train}), slope=slopes[k])
    return dict(models=models, slopes=slopes, receipt=receipt)


def second_pair(h81, mu: np.ndarray, x: np.ndarray, ids: list[str], first: list[int], first_u: list[float], scale: float,
                swap: bool) -> list[int]:
    r = np.clip(np.asarray(first_u, dtype=np.float64) - mu[first], -CLIP, CLIP)
    r = r - r.mean()
    if swap:
        r = r[::-1].copy()
    return h81.ranked(h81.posterior_mean(mu, x, first, list(r), scale), ids, set(first))[:2]


def select(h81, mu, x, ids, scale, rng_seed, reveal) -> dict:
    """`reveal(indices) -> utilities` is the only outcome access; it is called on the first pair only."""
    top = h81.ranked(mu, ids)
    picks = {"s_top4": top[:4], "random4": [int(i) for i in np.random.default_rng(rng_seed).choice(len(ids), 4, replace=False)]}
    fu = reveal(top[:2])
    for arm, swap in (("s_2plus2", False), ("s_2plus2_swap", True)):
        picks[arm] = None if any(v is None for v in fu) else top[:2] + second_pair(h81, mu, x, ids, top[:2], fu, scale, swap)
    return picks


def evaluate(h106, h81, prop, priors, contexts, seeds, out: Path, run_id: str, workers: int, tag: str) -> list[dict]:
    sys.path.insert(0, str(h81.MICRO / "python"))
    from rust_evaluator import get_w1_sign_matrix
    W = get_w1_sign_matrix().astype(np.float64)
    ledger, store = h81.Ledger(out / "ledger.json", cap=FIT_CAP, deadline=DEADLINE), h81.FitStore(out / "fits.jsonl", run_id)
    rows, shortfalls = [], []
    for c in contexts:
        k, p = prop.fold[c], h81.pool(c)
        scale = h81.lengthscale((p["program"].astype(np.float64) @ W) / math.sqrt(412))
        for seed in seeds:
            props = prop.batch(c, seed, STREAM)
            if len(props) < h106.N_PROPOSALS:
                shortfalls.append(dict(context=c, seed=seed, proposals=len(props))); continue
            ids, dsls = [q["program_sha256"] for q in props], [q["dsl"] for q in props]
            feats = prop.x(c, dsls)
            mu = priors["slopes"][k] * priors["models"][k].predict(feats)
            x = (feats[:, 412:].astype(np.float64) @ W) / math.sqrt(412)

            def reveal(sel):
                res = store.run([(c, dsls[i]) for i in sel], out / "scratch", ledger, workers)
                return [h81.utility(*(res[(c, ids[i])][f] for f in ("status", "gate", "fitness"))) for i in sel]

            picks = select(h81, mu, x, ids, scale, h81.hseed("h112-random", c, seed), reveal)
            res = store.run([(c, d) for d in dsls], out / "scratch", ledger, workers)
            u = [h81.utility(*(res[(c, s)][f] for f in ("status", "gate", "fitness"))) for s in ids]
            complete = all(v is not None for v in u)
            for a in ARMS:
                sel = picks[a]
                ok = complete and sel is not None and len(set(sel)) == 4
                rows.append(dict(tag=tag, context=c, family=prop.fam[c], seed=seed, arm=a, status="completed" if ok else "incomplete",
                                 best_of_four=max(u[i] for i in sel) if ok else None, revealed=None if sel is None else [ids[i] for i in sel]))
            rows.append(dict(tag=tag, context=c, family=prop.fam[c], seed=seed, arm="ORACLE4", status="completed" if complete else "incomplete",
                             best_of_four=max(u) if complete else None, revealed=None))  # ceiling, never a contender
    with open(out / f"rows_{tag.lower()}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    (out / f"shortfalls_{tag.lower()}.json").write_text(json.dumps(shortfalls, indent=1))
    return rows


def analyze(h81, rows: list[dict], roster: list[str], shortfalls: list, elo=None) -> dict:
    rows = [r for r in rows if r["tag"] == "H112"]
    if not rows:
        raise SystemExit("no H112 rows: dry-run rows are never analyzed")
    got = defaultdict(list)
    for r in rows:
        if r["status"] == "completed" and r["arm"] in ARMS:
            got[(r["context"], r["arm"])].append(r["seed"])
    gaps = [(c, a) for c in roster for a in ARMS if sorted(got.get((c, a), [])) != sorted(EVAL_SEEDS)]
    extra = {r["context"] for r in rows} - set(roster)
    fam = {r["context"]: r["family"] for r in rows}
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["status"] == "completed" and r["arm"] in ARMS:
            by[r["context"]][r["arm"]].append(r["best_of_four"])
    cm = {c: {a: mean(v[a]) for a in ARMS} for c, v in by.items() if all(v.get(a) for a in ARMS)}
    families = sorted({fam[c] for c in cm})
    fv = {f: {a: mean(cm[c][a] for c in cm if fam[c] == f) for a in ARMS} for f in families}
    draws = np.random.default_rng(h81.BOOTSTRAP_SEED).integers(0, len(families), size=(h81.BOOTSTRAP_DRAWS, len(families)))

    def con(a, b):
        v = np.array([fv[f][a] - fv[f][b] for f in families]); c = h81.contrast(v, draws)
        med = np.median(v[draws], axis=1)
        c["median_ci95"] = [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]
        return c
    out = dict(contexts=len(cm), families=len(families), roster_contexts=len(roster), panel_gaps=len(gaps),
               off_roster_contexts=len(extra), proposal_shortfalls=len(shortfalls),
               family_mean_best_of_four={a: mean(fv[f][a] for f in families) for a in ARMS} if families else {},
               primary=dict(contrast=f"{PRIMARY[0]} - {PRIMARY[1]}", **con(*PRIMARY)) if families else {},
               secondary={f"{a} - {b}": con(a, b) for a, b in SECONDARY} if families else {})
    if elo is not None and cm:
        recs = [elo.DatasetEvalRecord(c, fam[c], "regression", v) for c, v in sorted(cm.items())]
        out["family_elo"] = elo.compute_family_grouped_bradley_terry_elo(recs, anchor_contender=ANCHOR, anchor_elo=1500.0,
                                                                         tie_margin=h81.TIE, prior=0.01, n_iters=400)
    if gaps or extra or shortfalls or len(cm) != len(roster):
        out["verdict"] = f"INCOMPLETE (panel gaps {len(gaps)}, off-roster {len(extra)}, shortfalls {len(shortfalls)}); no verdict"
        return out
    placebo = out["secondary"]["s_2plus2 - s_2plus2_swap"]["family_mean"] > 0
    out["verdict"] = ("ADVANCE (primary passes the common gate and beats the swap placebo)"
                      if out["primary"]["common_gate_passed"] and placebo else "NO_ADVANCE")
    return out


def fold_manifest(prop, memory: list[dict]) -> dict:
    per = defaultdict(lambda: dict(contexts=[], families=set(), memory_rows=0))
    for c in prop.roster:
        per[prop.fold[c]]["contexts"].append(c); per[prop.fold[c]]["families"].add(prop.fam[c])
    for r in memory:
        per[r["fold"]]["memory_rows"] += 1
    return dict(schema="jope.h112.fold-manifest/v1", source="W1 fold manifest (sha256 dd211134...) over the frozen H67 roster",
                folds={str(k): dict(contexts=sorted(v["contexts"]), families=sorted(v["families"]), memory_rows=v["memory_rows"])
                       for k, v in sorted(per.items())},
                families_spanning_folds=sum(1 for f in {prop.fam[c] for c in prop.roster}
                                            if len({prop.fold[c] for c in prop.roster if prop.fam[c] == f}) > 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prep", "dry-run", "run", "analyze"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--h94-run", type=Path, required=True)
    ap.add_argument("--h103-run", type=Path, required=True)
    ap.add_argument("--h110-run", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    h106 = load_h106(); h81 = h106.load_h81(); h81.verify_pins()
    prop = h106.Proposer(h81)
    a.out.mkdir(parents=True, exist_ok=True)
    receipt = dict(run_id=a.run_id, stage=a.stage, source_sha256=h81.sha256_file(Path(__file__)), h106_source_sha256=H106_SHA,
                   h110_fits_sha256=H110_FITS_SHA, gpu_seconds=0)
    if a.stage == "analyze":
        rows = [json.loads(l) for l in (a.out / "rows_h112.jsonl").read_text().splitlines() if l.strip()]
        receipt.update(analyze(h81, rows, prop.roster, json.loads((a.out / "shortfalls_h112.json").read_text()), h81.load_elo()))
        (a.out / "metrics.json").write_text(json.dumps(receipt, indent=1)); print(receipt["verdict"]); return
    memory = full_memory(h106, h81, prop, a.h94_run, a.h103_run, a.h110_run)
    with open(a.out / "memory.jsonl", "w") as f:
        for r in memory:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    manifest = fold_manifest(prop, memory)
    (a.out / "fold_manifest.json").write_text(json.dumps(manifest, indent=1))
    priors = fit_priors(h106, prop, memory)
    with open(a.out / "priors.pkl", "wb") as f:
        pickle.dump(dict(models=priors["models"], slopes=priors["slopes"]), f, protocol=pickle.HIGHEST_PROTOCOL)
    receipt.update(memory_rows=len(memory), memory_sha256=h81.sha256_file(a.out / "memory.jsonl"),
                   fold_manifest_sha256=h81.sha256_file(a.out / "fold_manifest.json"),
                   families_spanning_folds=manifest["families_spanning_folds"], priors=priors["receipt"],
                   priors_sha256=h81.sha256_file(a.out / "priors.pkl"))
    if a.stage == "dry-run":
        ctx = h106.dry_contexts(prop)
        rows = evaluate(h106, h81, prop, priors, ctx, (901,), a.out, a.run_id, a.workers, "DRY_RUN")
        receipt.update(dry_run_contexts=ctx, dry_run_rows=len(rows), dry_run_complete=sum(r["status"] == "completed" for r in rows),
                       ledger=h81.Ledger(a.out / "ledger.json", cap=FIT_CAP, deadline=DEADLINE).state,
                       note="DRY_RUN rows are integrity evidence only; never part of an H112 result")
    elif a.stage == "run":
        rows = evaluate(h106, h81, prop, priors, prop.roster, EVAL_SEEDS, a.out, a.run_id, a.workers, "H112")
        receipt.update(rows=len(rows), ledger=h81.Ledger(a.out / "ledger.json", cap=FIT_CAP, deadline=DEADLINE).state)
    (a.out / f"{a.stage}_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items() if k != "priors"}, indent=1))


if __name__ == "__main__":
    main()
