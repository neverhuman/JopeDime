"""H106 — proposal-distribution scorer and W9 memory-source replay (non-neural learner class): CPU harness.

H94 and H103 found that W1-recipe priors (program-only and context-aware), which rank admitted pool programs well,
do not beat random selection among the live proposer's NEW programs. H106 trains a scorer on the proposal
distribution itself and asks two questions:

  A. Does a scorer trained on logged native reveals of proposer programs beat random4 on fresh proposals?
  B. W9 (causal closeout, non-neural learner class): does memory acquired by a guide (the program prior) make a better
     common learner than uniformly acquired memory, at matched rows and identical learner recipe?

Preregistered here (PREREG.md is the prose copy; the frozen source is authoritative):
* Memories: native reveals logged by H94 (opus-h94-20261031a rows.jsonl, sha256 18564293...) and H103
  (opus-h103-20261031a rows_w4.jsonl, sha256 2e075757...), all on the frozen H67 roster (50 contexts / 44 families).
  Program DSLs are recovered by replaying each run's deterministic proposer stream; every logged program must be
  recovered or the build stops. Sources: UNIFORM = H103 random4; PRIOR = H103 prog_top4 + H94 PRIOR_top4;
  ADAPTIVE = H103 prog_2plus2 + prog_2plus2_swap; CONTEXT = H94 CTX_top4 + HEDGE_CTX; ALL = their union.
  Rows are de-duplicated per (context, program).
* Learner (identical for every memory): features = [prog412(p) - prog412(context champion), prog412(p)]; target =
  the program's within-context utility percentile among that memory's rows for the context, centred at zero;
  HistGradientBoostingRegressor(max_iter 200, lr .05, leaves 15, l2 1, random_state 1060+k), one model per W1 outer
  fold k trained only on contexts outside fold k. Matched rows: UNIFORM, PRIOR, ADAPTIVE and CONTEXT are each
  subsampled (sha256-keyed order) to the smallest of the four row counts available for that fold.
* Prospective endpoint: roster contexts x fresh seeds 801-803, 16 NEW proposals per context-seed (stream
  "h106-stream", excluding the admitted pool). ALL 16 are evaluated by native original-cut training CV, so every arm
  is computed on a fully labelled batch (no restricted-subset comparison). Each context is scored by the fold model
  that excluded its whole family. Arm value = best utility among its top 4 (zero history).
* Arms: S_ALL, S_UNIFORM, S_PRIOR, S_ADAPTIVE, S_CONTEXT (scorer top 4), prog_top4 (H94 program-only refit),
  random4 (h106-random), ORACLE4 (best possible four; descriptive ceiling only).
* Primaries (Bonferroni m = 2, one-sided .025 each, common gate): A = S_ALL - random4; B = S_PRIOR - S_UNIFORM.
  Secondary: S_ADAPTIVE - S_UNIFORM, S_CONTEXT - S_UNIFORM, S_ALL - prog_top4, prog_top4 - random4 (H103 replication
  on a fresh stream). Family Elo with random4 = 1500. Completeness gate as in H96.
* Budget: 2,400 native fits (150 context-seeds x 16), xbabe2, nice 15, load-gated; deadline 2026-11-01T12:00Z.
No DL-Val, outer-test, sealed or holdout row is read; DOPE is not loaded; zero GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
H81_PATH = HERE.parent / "h81_w4_prospective_2plus2" / "run_prospective.py"
H81_SHA = "9ad50be577a30c616f7668d52f621d5411e70019ba9f1d1960eb8a45a448c999"
PROG_MODELS_SHA = "5938a35497ccde1a1489c4d6b13d122196aed16739a3272afea06d57d5fff692"
LOGS = {"h94": ("rows.jsonl", "18564293060da510364950c041c97cd717021337f0289956045d53beae509dd1", "h94-stream"),
        "h103": ("rows_w4.jsonl", "2e0757579d5effa82efc1a628b7172880c8e03f0c66d922e5bbb7e514a9cc205", "h96-stream")}
SOURCES = {"UNIFORM": {("h103", "random4")},
           "PRIOR": {("h103", "prog_top4"), ("h94", "PRIOR_top4")},
           "ADAPTIVE": {("h103", "prog_2plus2"), ("h103", "prog_2plus2_swap")},
           "CONTEXT": {("h94", "CTX_top4"), ("h94", "HEDGE_CTX")}}
MATCHED = ("UNIFORM", "PRIOR", "ADAPTIVE", "CONTEXT")
SCORERS = MATCHED + ("ALL",)
ARMS = tuple(f"S_{s}" for s in SCORERS) + ("prog_top4", "random4")
ANCHOR = "random4"
PRIMARY = (("S_ALL", "random4"), ("S_PRIOR", "S_UNIFORM"))
SECONDARY = (("S_ADAPTIVE", "S_UNIFORM"), ("S_CONTEXT", "S_UNIFORM"), ("S_ALL", "prog_top4"), ("prog_top4", "random4"))
EVAL_SEEDS = (801, 802, 803)
STREAM = "h106-stream"
N_PROPOSALS = 16
FIT_CAP = 2400
DEADLINE = "2026-11-01T12:00:00Z"


def load_h81():
    if hashlib.sha256(H81_PATH.read_bytes()).hexdigest() != H81_SHA:
        raise SystemExit("H81 runner bytes differ from the merged source")
    spec = importlib.util.spec_from_file_location("h81_run", H81_PATH)
    m = importlib.util.module_from_spec(spec); sys.modules["h81_run"] = m; spec.loader.exec_module(m)
    return m


def hk(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


class Proposer:
    """Replays the live proposer exactly as H94/H96 did (first 16 unique programs not in the admitted pool)."""

    def __init__(self, h81):
        sys.path.insert(0, str(h81.MICRO / "python"))
        from rust_evaluator import featurize_dsl_412, propose_flywheel_candidates
        self.h81, self.feat, self.propose = h81, featurize_dsl_412, propose_flywheel_candidates
        train, self.fold, _, self.roster = h81.load_inputs()
        self.idx = {r["context_sha256"]: i for i, r in enumerate(train)}
        self.fam = {r["context_sha256"]: r["family_id"] for r in train}
        self.champ = defaultdict(list)
        for r in json.loads((h81.INPUT / "champions.json").read_text()):
            self.champ[r["context_sha256"]].append({"dsl": r["dsl"], "program_sha256": r["program_sha256"]})
        self.bank = [r for r in json.loads((Path.home() / "jope-data/h58/cache/champions_featurized.json").read_text())
                     if r["dl_split"] == "DL-Train" and r["context_sha256"] in self.idx]
        self.lm = np.load(h81.W1 / "cache/train_only_context_features.npz", allow_pickle=False)["lm48_z"].astype(np.float64)

    def batch(self, c: str, seed: int, stream: str) -> list[dict]:
        known, props, batch = set(self.h81.pool(c)["sha"]), [], 0
        seen = set(known)
        while len(props) < N_PROPOSALS and batch < 50:
            for q in self.propose(self.champ[c], self.lm[self.idx[c]], self.bank, self.fam[c], n_proposals=16,
                                  seed=self.h81.hseed(stream, c, seed, batch) % (2 ** 31)):
                if q["program_sha256"] not in seen and len(props) < N_PROPOSALS:
                    seen.add(q["program_sha256"]); props.append(q)
            batch += 1
        return props

    def x(self, c: str, dsls: list[str]) -> np.ndarray:
        champ = np.asarray(self.feat(self.champ[c][0]["dsl"])[0], dtype=np.float32)
        f = np.stack([np.asarray(self.feat(d)[0], dtype=np.float32) for d in dsls])
        return np.concatenate([f - champ[None], f], axis=1)


def percentile_target(u: np.ndarray) -> np.ndarray:
    """Within-context utility percentile, centred at zero; ties share their mean rank."""
    order = np.argsort(np.argsort(u, kind="stable"), kind="stable").astype(np.float64)
    for v in np.unique(u):
        m = u == v
        order[m] = order[m].mean()
    return (order / max(len(u) - 1, 1)) - 0.5 if len(u) > 1 else np.zeros(1)


def build_memory(prop: Proposer, logs_dir: dict[str, Path]) -> list[dict]:
    rows, missing = {}, []
    for log, (name, sha, stream) in LOGS.items():
        path = logs_dir[log] / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"{log} reveal log differs from the published run")
        cache = {}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") != "completed":
                continue
            key = (r["context"], r["seed"])
            if key not in cache:
                cache[key] = {q["program_sha256"]: q["dsl"] for q in prop.batch(r["context"], r["seed"], stream)}
            for s, u in zip(r["revealed"], r["revealed_utility"]):
                if s not in cache[key]:
                    missing.append((log, r["context"][:12], r["seed"], s[:12])); continue
                for src, members in SOURCES.items():
                    if (log, r["arm"]) in members:
                        rows.setdefault((src, r["context"], s), dict(source=src, context=r["context"], family=prop.fam[r["context"]],
                                                                    fold=prop.fold[r["context"]], program_sha256=s,
                                                                    dsl=cache[key][s], utility=float(u)))
    if missing:
        raise SystemExit(f"{len(missing)} logged programs not recovered by stream replay, e.g. {missing[:3]}")
    return sorted(rows.values(), key=lambda r: (r["source"], r["context"], r["program_sha256"]))


def training_rows(memory: list[dict], source: str, k: int) -> list[dict]:
    """Rows of `source` outside fold k; matched sources are cut to the smallest matched count (hash order)."""
    def pick(src):
        rs = [r for r in memory if r["source"] == src and r["fold"] != k] if src != "ALL" else \
             list({(r["context"], r["program_sha256"]): r for r in memory if r["fold"] != k}.values())
        return sorted(rs, key=lambda r: hk("h106-mem", src, r["context"], r["program_sha256"]))
    if source == "ALL":
        return pick("ALL")
    n = min(len(pick(s)) for s in MATCHED)
    return pick(source)[:n]


def fit_scorers(prop: Proposer, memory: list[dict], out: Path, threads: int = 8) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    models, receipt = {}, {}
    with threadpool_limits(limits=threads):
        for src in SCORERS:
            for k in range(5):
                rs = training_rows(memory, src, k)
                by = defaultdict(list)
                for r in rs:
                    by[r["context"]].append(r)
                X, Y = [], []
                for c, g in sorted(by.items()):
                    X.append(prop.x(c, [r["dsl"] for r in g])); Y.append(percentile_target(np.array([r["utility"] for r in g])))
                m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                                  random_state=1060 + k).fit(np.concatenate(X), np.concatenate(Y))
                models[(src, k)] = m
                receipt[f"{src}/fold{k}"] = dict(rows=len(rs), contexts=len(by))
    with open(out / "scorers.pkl", "wb") as f:
        pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dict(models=models, receipt=receipt)


def arm_picks(scores: dict[str, np.ndarray], ids: list[str], ranked, rng_seed: int) -> dict[str, list[int]]:
    picks = {a: ranked(scores[a], ids)[:4] for a in ARMS if a != "random4"}
    picks["random4"] = [int(i) for i in np.random.default_rng(rng_seed).choice(len(ids), 4, replace=False)]
    return picks


def evaluate(h81, prop: Proposer, scorers: dict, prog: dict, contexts: list[str], seeds, out: Path, run_id: str,
             workers: int, tag: str) -> list[dict]:
    ledger, store = h81.Ledger(out / "ledger.json", cap=FIT_CAP, deadline=DEADLINE), h81.FitStore(out / "fits.jsonl", run_id)
    rows, shortfalls = [], []
    for c in contexts:
        k, offset = prop.fold[c], float(np.mean(h81.pool(c)["u"]))
        for seed in seeds:
            props = prop.batch(c, seed, STREAM)
            if len(props) < N_PROPOSALS:
                shortfalls.append(dict(context=c, seed=seed, proposals=len(props))); continue
            ids, dsls = [q["program_sha256"] for q in props], [q["dsl"] for q in props]
            x = prop.x(c, dsls)
            scores = {f"S_{s}": scorers[(s, k)].predict(x) for s in SCORERS}
            scores["prog_top4"] = prog[k].predict(x[:, 412:]) + offset
            picks = arm_picks(scores, ids, h81.ranked, h81.hseed("h106-random", c, seed))  # fixed before any outcome
            res = store.run([(c, d) for d in dsls], out / "scratch", ledger, workers)
            u = [h81.utility(*(res[(c, s)][f] for f in ("status", "gate", "fitness"))) for s in ids]
            complete = all(v is not None for v in u)
            for a, sel in list(picks.items()) + [("ORACLE4", None)]:
                vals = sorted([v for v in u if v is not None], reverse=True)[:4] if sel is None else [u[i] for i in sel]
                ok = complete and len(vals) == 4
                rows.append(dict(tag=tag, context=c, family=prop.fam[c], seed=seed, arm=a, status="completed" if ok else "incomplete",
                                 best_of_four=max(vals) if ok else None, revealed=None if sel is None else [ids[i] for i in sel]))
    with open(out / f"rows_{tag.lower()}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    (out / f"shortfalls_{tag.lower()}.json").write_text(json.dumps(shortfalls, indent=1))
    return rows


def analyze(h81, rows: list[dict], roster: list[str], shortfalls: list, elo=None) -> dict:
    rows = [r for r in rows if r["tag"] == "H106" and r["arm"] in ARMS + ("ORACLE4",)]
    if not rows:
        raise SystemExit("no H106 rows: dry-run rows are never analyzed")
    want = sorted(EVAL_SEEDS)
    got = defaultdict(list)
    for r in rows:
        if r["status"] == "completed":
            got[(r["context"], r["arm"])].append(r["seed"])
    gaps = [(c, a) for c in roster for a in ARMS if sorted(got.get((c, a), [])) != want]
    extra = {r["context"] for r in rows} - set(roster)
    fam = {r["context"]: r["family"] for r in rows}
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["status"] == "completed":
            by[r["context"]][r["arm"]].append(r["best_of_four"])
    cm = {c: {a: mean(v[a]) for a in ARMS + ("ORACLE4",)} for c, v in by.items() if all(v.get(a) for a in ARMS + ("ORACLE4",))}
    families = sorted({fam[c] for c in cm})
    fv = {f: {a: mean(cm[c][a] for c in cm if fam[c] == f) for a in ARMS + ("ORACLE4",)} for f in families}
    draws = np.random.default_rng(h81.BOOTSTRAP_SEED).integers(0, len(families), size=(h81.BOOTSTRAP_DRAWS, len(families)))

    def con(a, b, adjusted):
        v = np.array([fv[f][a] - fv[f][b] for f in families]); c = h81.contrast(v, draws)
        if adjusted:  # Bonferroni m = 2: one-sided .025 replaces the .05 LCB in the gate
            lcb = float(np.percentile(v[draws].mean(axis=1), 2.5))
            c.update(adjusted_one_sided_lcb=lcb, common_gate_passed=bool(lcb > h81.GATE_LCB and c["family_median"] >= 0
                                                                        and c["families_losing_more_than_0_01_fraction"] <= 0.10))
        med = np.median(v[draws], axis=1)
        c["median_ci95"] = [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]
        return c
    out = dict(contexts=len(cm), families=len(families), roster_contexts=len(roster), panel_gaps=len(gaps),
               off_roster_contexts=len(extra), proposal_shortfalls=len(shortfalls),
               family_mean_best_of_four={a: mean(fv[f][a] for f in families) for a in ARMS + ("ORACLE4",)},
               primary={f"{a} - {b}": con(a, b, True) for a, b in PRIMARY},
               secondary={f"{a} - {b}": con(a, b, False) for a, b in SECONDARY})
    if elo is not None and cm:
        recs = [elo.DatasetEvalRecord(c, fam[c], "regression", {a: v[a] for a in ARMS}) for c, v in sorted(cm.items())]
        out["family_elo"] = elo.compute_family_grouped_bradley_terry_elo(recs, anchor_contender=ANCHOR, anchor_elo=1500.0,
                                                                         tie_margin=h81.TIE, prior=0.01, n_iters=400)
    if gaps or extra or shortfalls or len(cm) != len(roster):
        out["verdict"] = f"INCOMPLETE (panel gaps {len(gaps)}, off-roster {len(extra)}, shortfalls {len(shortfalls)}); no verdict"
        return out
    passed = {k: v["common_gate_passed"] for k, v in out["primary"].items()}
    out["verdict"] = dict(A_scorer_vs_random="PASS" if passed["S_ALL - random4"] else "FAIL",
                          B_guided_vs_uniform_memory="PASS" if passed["S_PRIOR - S_UNIFORM"] else "FAIL")
    return out


def dry_contexts(prop: Proposer, n: int = 3) -> list[str]:
    rf = {prop.fam[c] for c in prop.roster}
    pool = [c for c in prop.idx if prop.fam[c] not in rf]
    return sorted(pool, key=lambda c: hk("h106-dry", c))[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prep", "dry-run", "run", "analyze"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--h94-run", type=Path, required=True)
    ap.add_argument("--h103-run", type=Path, required=True)
    ap.add_argument("--prog-models", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    h81 = load_h81(); h81.verify_pins()
    if h81.sha256_file(a.prog_models) != PROG_MODELS_SHA:
        raise SystemExit("program-only models.pkl hash mismatch")
    a.out.mkdir(parents=True, exist_ok=True)
    prop = Proposer(h81)
    receipt = dict(run_id=a.run_id, stage=a.stage, source_sha256=h81.sha256_file(Path(__file__)), h81_source_sha256=H81_SHA,
                   prog_models_sha256=PROG_MODELS_SHA, reveal_logs={k: v[1] for k, v in LOGS.items()}, gpu_seconds=0)
    if a.stage == "analyze":
        rows = [json.loads(l) for l in (a.out / "rows_h106.jsonl").read_text().splitlines() if l.strip()]
        receipt.update(analyze(h81, rows, prop.roster, json.loads((a.out / "shortfalls_h106.json").read_text()), h81.load_elo()))
        (a.out / "metrics.json").write_text(json.dumps(receipt, indent=1)); print(receipt["verdict"]); return
    memory = build_memory(prop, {"h94": a.h94_run, "h103": a.h103_run})
    with open(a.out / "memory.jsonl", "w") as f:
        for r in memory:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    counts = defaultdict(int)
    for r in memory:
        counts[r["source"]] += 1
    receipt.update(memory_rows=dict(counts), memory_contexts=len({r["context"] for r in memory}),
                   memory_sha256=h81.sha256_file(a.out / "memory.jsonl"))
    fitted = fit_scorers(prop, memory, a.out)
    receipt.update(scorer_training=fitted["receipt"], scorers_sha256=h81.sha256_file(a.out / "scorers.pkl"))
    with open(a.prog_models, "rb") as f:
        prog = pickle.load(f)
    if a.stage == "dry-run":
        ctx = dry_contexts(prop)
        rows = evaluate(h81, prop, fitted["models"], prog, ctx, (801,), a.out, a.run_id, a.workers, "DRY_RUN")
        receipt.update(dry_run_contexts=ctx, dry_run_rows=len(rows), dry_run_complete=sum(r["status"] == "completed" for r in rows),
                       ledger=h81.Ledger(a.out / "ledger.json", cap=FIT_CAP, deadline=DEADLINE).state,
                       note="DRY_RUN rows are integrity evidence only; never part of an H106 result")
    elif a.stage == "run":
        rows = evaluate(h81, prop, fitted["models"], prog, prop.roster, EVAL_SEEDS, a.out, a.run_id, a.workers, "H106")
        receipt.update(rows=len(rows), ledger=h81.Ledger(a.out / "ledger.json", cap=FIT_CAP, deadline=DEADLINE).state)
    (a.out / f"{a.stage}_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items() if k != "scorer_training"}, indent=1))


if __name__ == "__main__":
    main()
