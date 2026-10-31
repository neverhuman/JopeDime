"""H96 — W4 active reveal (two-plus-two) on NEW programs with the program-only prior: CPU harness and dry run.

W4 asks whether choosing observations adaptively beats a fixed history at four total fits. H81 built the prospective
new-program stratum but was blocked before scoring: its W1 context-aware comparator could not be reproduced. H94 then
showed that, on new programs, the program-only W1-recipe refit is the stronger prior (the context model lost by
-51.9 Elo). H96 therefore uses the program-only refit as the W4 comparator. That choice was informed by H94's outcome
on the same roster: it is disclosed as adaptive, and no W4 arm outcome on the roster has been seen.

Preregistered here (PREREG.md is the prose copy):
* Roster: the frozen H67 roster (50 DL-Train contexts / 44 families, every family inside one W1 fold) x seeds
  701-703. Per context and seed, 16 NEW proposals (never in the admitted pool) from the pinned live proposer, stream
  "h96-stream" (H81 and H94 used other streams).
* Prior: mu = f_k(prog412) + mean admitted-pool utility of the context, with f_k the H94 program-only refit for the
  W1 outer fold that excluded the context's whole family (models_prog.pkl, sha256 5938a354...). The pool mean is
  admitted history, so it moves the prior onto the utility scale without using any new program's outcome.
* CPU arms, zero history, four reveals each (the merged H81/H79 rule, imported, never copied): prog_top4
  (comparator, Elo anchor); prog_2plus2 (reveal the top two, update an RBF GP posterior mean on proj16 with their
  residuals u - mu, reveal the top two of the rest); prog_2plus2_swap (placebo: the two residuals swapped); random4.
* GPU arm (queued, not run by this harness): r882_top4, scored by the r882 checkpoint copy with DOPE c51a56d0 frozen.
  r882 requires four revealed anchors, so it sees the context's four hash-ordered pool anchors. That is more history
  than the CPU arms have, so r882_top4 is descriptive continuity only, never the W4 comparator. See GPU_QUEUE.md.
* Endpoint: the best utility among the four new reveals (zero-history W4 endpoint); execution errors make an arm
  INCOMPLETE. Seeds are averaged within context and contexts within family.
* Primary: prog_2plus2 - prog_top4 under the common gate (one-sided .05 LCB > +.001, median >= 0, <= 10% of
  families losing > .01), paired family bootstrap (10,000 draws, seed 20261031). Secondary: prog_2plus2 -
  prog_2plus2_swap, prog_top4 - random4. Family Elo with prog_top4 = 1500. Matched elapsed time and fit counts per arm.
* Prior-signal guard (before any proposal is fitted): on the roster's admitted cached pools, zero-history prog_top4
  must beat random4 in family-mean best-of-four utility. Otherwise the run stops.
* Budget: at most 1,800 native fits (W4 ceiling 2,400 cumulative; H81 spent 0), xbabe2 only, nice 15, load-gated.
* Dry run: three DL-Train contexts from families outside the roster, seed 701, full pipeline with native fits. Its
  rows are tagged DRY_RUN and are never part of a W4 result.
No DL-Val, outer-test, sealed or holdout row is read. DOPE is not loaded by this harness. Zero GPU.
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
H81_SHA = "9ad50be577a30c616f7668d52f621d5411e70019ba9f1d1960eb8a45a448c999"
PROG_MODELS_SHA = "5938a35497ccde1a1489c4d6b13d122196aed16739a3272afea06d57d5fff692"
ARMS = ("random4", "prog_top4", "prog_2plus2", "prog_2plus2_swap")
H81_NAME = {"random4": "random4", "prog_top4": "gbm_top4", "prog_2plus2": "gbm_2plus2", "prog_2plus2_swap": "gbm_2plus2_swap"}
ANCHOR = "prog_top4"
PRIMARY = ("prog_2plus2", "prog_top4")
SECONDARY = (("prog_2plus2", "prog_2plus2_swap"), ("prog_top4", "random4"))
FIT_CAP = 1800
DEADLINE = "2026-11-01T12:00:00Z"
STREAM = "h96-stream"
N_DRY = 3


def load_h81():
    if hashlib.sha256(H81_PATH.read_bytes()).hexdigest() != H81_SHA:
        raise SystemExit("H81 runner bytes differ from the merged source")
    spec = importlib.util.spec_from_file_location("h81_run", H81_PATH)
    m = importlib.util.module_from_spec(spec); sys.modules["h81_run"] = m; spec.loader.exec_module(m)
    return m


def dry_contexts(train: list[dict], roster: list[str], n: int = N_DRY) -> list[str]:
    """DL-Train contexts whose family is not in the roster, in sha256-keyed order."""
    fam = {r["context_sha256"]: r["family_id"] for r in train}
    rf = {fam[c] for c in roster}
    pool = [r["context_sha256"] for r in train if r["family_id"] not in rf]
    return sorted(pool, key=lambda c: hashlib.sha256(f"h96-dry|{c}".encode()).hexdigest())[:n]


def select(h81, mu: np.ndarray, x: np.ndarray, ids: list[str], scale: float, rng_seed: int, reveal) -> dict:
    """All four arms. `reveal(indices) -> utilities` is the only outcome access, called after the first stage."""
    first = {a: h81.first_stage(H81_NAME[a], mu, ids, rng_seed) for a in ARMS}
    picks = {a: list(first[a]) for a in ARMS}
    for a in ("prog_2plus2", "prog_2plus2_swap"):
        fu = reveal(first[a])
        if any(v is None for v in fu):
            picks[a] = None  # an execution error in the first pair blocks the update: arm INCOMPLETE
            continue
        picks[a] = first[a] + h81.second_stage(H81_NAME[a], mu, x, ids, first[a], fu, scale)
    return picks


def guard(h81, prog: dict) -> dict:
    """Zero-history best-of-four on cached roster pools: does the prior beat random4?"""
    sys.path.insert(0, str(h81.MICRO / "python"))
    train, fold_by_cid, _, acq = h81.load_inputs()
    fam = {r["context_sha256"]: r["family_id"] for r in train}
    per = defaultdict(lambda: defaultdict(list))
    for c in acq:
        p = h81.pool(c); k = fold_by_cid[c]
        un = np.arange(4, len(p["u"])); ids = [p["sha"][i] for i in un]
        mu = prog[k].predict(p["program"][un])
        for seed in h81.SEEDS:
            top = h81.ranked(mu, ids)[:4]
            rnd = h81.first_stage("random4", mu, ids, h81.hseed("h96-guard-random", c, seed))
            per[fam[c]]["prog_top4"].append(float(np.max(p["u"][un[top]])))
            per[fam[c]]["random4"].append(float(np.max(p["u"][un[rnd]])))
    fm = {a: mean(mean(v[a]) for v in per.values()) for a in ("prog_top4", "random4")}
    return dict(contexts=len(acq), families=len(per), cached_family_mean_best_of_four=fm,
                prior_minus_random=fm["prog_top4"] - fm["random4"], passed=bool(fm["prog_top4"] > fm["random4"]))


def run(h81, out: Path, run_id: str, prog: dict, contexts: list[str], seeds, workers: int, tag: str) -> list[dict]:
    sys.path.insert(0, str(h81.MICRO / "python"))
    from rust_evaluator import featurize_dsl_412, get_w1_sign_matrix, propose_flywheel_candidates
    train, fold_by_cid, _, _ = h81.load_inputs()
    idx = {r["context_sha256"]: i for i, r in enumerate(train)}
    fam = {r["context_sha256"]: r["family_id"] for r in train}
    W = get_w1_sign_matrix().astype(np.float64)
    champ = defaultdict(list)
    for r in json.loads((h81.INPUT / "champions.json").read_text()):
        champ[r["context_sha256"]].append({"dsl": r["dsl"], "program_sha256": r["program_sha256"]})
    bank = [r for r in json.loads((Path.home() / "jope-data/h58/cache/champions_featurized.json").read_text())
            if r["dl_split"] == "DL-Train" and r["context_sha256"] in idx]
    lm = np.load(h81.W1 / "cache/train_only_context_features.npz", allow_pickle=False)["lm48_z"].astype(np.float64)
    ledger, store = h81.Ledger(out / "ledger.json", cap=FIT_CAP, deadline=DEADLINE), h81.FitStore(out / "fits.jsonl", run_id)
    rows, shortfalls = [], []
    for c in contexts:
        p = h81.pool(c); k = fold_by_cid[c]; known = set(p["sha"])
        scale = h81.lengthscale((p["program"].astype(np.float64) @ W) / math.sqrt(412))
        offset = float(np.mean(p["u"]))
        for seed in seeds:
            props, seen, batch = [], set(known), 0
            while len(props) < h81.N_PROPOSALS and batch < 50:
                for q in propose_flywheel_candidates(champ[c], lm[idx[c]], bank, fam[c], n_proposals=16,
                                                     seed=h81.hseed(STREAM, c, seed, batch) % (2 ** 31)):
                    if q["program_sha256"] not in seen and len(props) < h81.N_PROPOSALS:
                        seen.add(q["program_sha256"]); props.append(q)
                batch += 1
            if len(props) < 8:
                shortfalls.append(dict(context=c, seed=seed, proposals=len(props))); continue
            t0 = time.process_time()
            feats = np.stack([np.asarray(featurize_dsl_412(q["dsl"])[0], dtype=np.float32) for q in props])
            mu = prog[k].predict(feats).astype(np.float64) + offset
            x = (feats.astype(np.float64) @ W) / math.sqrt(412)
            ids = [q["program_sha256"] for q in props]
            score_cpu = time.process_time() - t0

            def reveal(sel):
                res = store.run([(c, props[i]["dsl"]) for i in sel], out / "scratch", ledger, workers)
                return [h81.utility(*(res[(c, ids[i])][k2] for k2 in ("status", "gate", "fitness"))) for i in sel]

            picks = select(h81, mu, x, ids, scale, h81.hseed("h96-random", c, seed), reveal)
            done = store.run([(c, props[i]["dsl"]) for a in ARMS if picks[a] for i in picks[a]], out / "scratch", ledger, workers)
            for a in ARMS:
                sel = picks[a] or []
                fits = [done[(c, ids[i])] for i in sel]
                vals = [h81.utility(f["status"], f["gate"], f["fitness"]) for f in fits]
                ok = len(sel) == 4 and all(v is not None for v in vals)
                rows.append(dict(tag=tag, context=c, family=fam[c], seed=seed, arm=a, status="completed" if ok else "incomplete",
                                 best_of_four=max(vals) if ok else None, revealed=[ids[i] for i in sel], revealed_utility=vals,
                                 native_wall_s=sum(f["wall_s"] for f in fits), native_cpu_s=sum(f["cpu_s"] for f in fits),
                                 selection_cpu_s=score_cpu, anchors_best=float(np.max(p["u"][:4]))))
    with open(out / f"rows_{tag.lower()}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    (out / f"shortfalls_{tag.lower()}.json").write_text(json.dumps(shortfalls, indent=1))
    return rows


def analyze(h81, rows: list[dict], elo=None, *, roster: list[str], shortfalls: list[dict]) -> dict:
    rows = [r for r in rows if r["tag"] == "W4"]
    if not rows:
        raise SystemExit("no W4 rows: dry-run rows are never analyzed")
    # Completeness gate: every roster context x arm must hold exactly one completed row per preregistered seed.
    seeds = sorted(h81.SEEDS)
    got = defaultdict(list)
    for r in rows:
        if r["status"] == "completed":
            got[(r["context"], r["arm"])].append(r["seed"])
    gaps = [f"{c[:12]}/{a}:{sorted(got.get((c, a), []))}" for c in roster for a in ARMS if sorted(got.get((c, a), [])) != seeds]
    extra = sorted({r["context"] for r in rows} - set(roster))
    bad = {(r["context"], r["seed"]) for r in rows if r["status"] != "completed"}
    fam = {r["context"]: r["family"] for r in rows}
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if (r["context"], r["seed"]) not in bad:
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
    out = dict(contexts=len(cm), families=len(families), incomplete_context_seeds=len(bad),
               family_mean_best_of_four={a: mean(fv[f][a] for f in families) for a in ARMS},
               primary=dict(contrast=f"{PRIMARY[0]} - {PRIMARY[1]}", **con(*PRIMARY)),
               secondary={f"{a} - {b}": con(a, b) for a, b in SECONDARY},
               elapsed_per_arm={a: dict(native_wall_s=sum(r["native_wall_s"] for r in rows if r["arm"] == a),
                                        native_cpu_s=sum(r["native_cpu_s"] for r in rows if r["arm"] == a)) for a in ARMS})
    if elo is not None:
        recs = [elo.DatasetEvalRecord(c, fam[c], "regression", v) for c, v in sorted(cm.items())]
        out["family_elo"] = elo.compute_family_grouped_bradley_terry_elo(recs, anchor_contender=ANCHOR, anchor_elo=1500.0,
                                                                         tie_margin=h81.TIE, prior=0.01, n_iters=400)
    out.update(roster_contexts=len(roster), panel_gaps=len(gaps), panel_gap_examples=gaps[:5],
               off_roster_contexts=len(extra), proposal_shortfalls=len(shortfalls))
    if gaps or extra or shortfalls or out["incomplete_context_seeds"] or out["contexts"] != len(roster):
        out["verdict"] = (f"INCOMPLETE (panel gaps {len(gaps)}, off-roster {len(extra)}, shortfalls {len(shortfalls)}, "
                          f"incomplete context-seeds {out['incomplete_context_seeds']}); no verdict")
        return out
    out["verdict"] = ("ADVANCE (primary passes the common gate)" if out["primary"]["common_gate_passed"]
                      else "NO_ADVANCE (primary gate not met)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("dry-run", "run", "analyze"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--prog-models", type=Path, required=True, help="H94 models_prog.pkl (sha256 5938a354...)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    h81 = load_h81()
    h81.verify_pins()
    if h81.sha256_file(a.prog_models) != PROG_MODELS_SHA:
        raise SystemExit("program-only models.pkl hash mismatch")
    with open(a.prog_models, "rb") as f:
        prog = pickle.load(f)
    a.out.mkdir(parents=True, exist_ok=True)
    train, _, _, roster = h81.load_inputs()
    receipt = dict(run_id=a.run_id, stage=a.stage, source_sha256=h81.sha256_file(Path(__file__)), h81_source_sha256=H81_SHA,
                   prog_models_sha256=PROG_MODELS_SHA, gpu_seconds=0)
    if a.stage == "analyze":
        rows = [json.loads(l) for l in (a.out / "rows_w4.jsonl").read_text().splitlines() if l.strip()]
        shortfalls = json.loads((a.out / "shortfalls_w4.json").read_text())
        receipt.update(analyze(h81, rows, h81.load_elo(), roster=roster, shortfalls=shortfalls))
        (a.out / "metrics.json").write_text(json.dumps(receipt, indent=1)); print(receipt["verdict"]); return
    g = guard(h81, prog)
    receipt["guard"] = g
    print("guard", json.dumps(g), flush=True)
    if not g["passed"]:
        (a.out / f"{a.stage}_receipt.json").write_text(json.dumps(receipt, indent=1))
        raise SystemExit("prior-signal guard failed: no proposals fitted")
    if a.stage == "dry-run":
        ctx = dry_contexts(train, roster)
        if set(ctx) & set(roster):
            raise SystemExit("dry-run context overlaps the roster")
        rows = run(h81, a.out, a.run_id, prog, ctx, (701,), a.workers, "DRY_RUN")
        receipt.update(dry_run_contexts=ctx, dry_run_rows=len(rows), dry_run_complete=sum(r["status"] == "completed" for r in rows),
                       ledger=h81.Ledger(a.out / "ledger.json", cap=FIT_CAP, deadline=DEADLINE).state,
                       note="DRY_RUN rows are integrity evidence only; never part of a W4 result")
    else:
        rows = run(h81, a.out, a.run_id, prog, roster, h81.SEEDS, a.workers, "W4")
        receipt.update(rows=len(rows), ledger=h81.Ledger(a.out / "ledger.json", cap=FIT_CAP, deadline=DEADLINE).state)
    (a.out / f"{a.stage}_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items() if k != "guard"}, indent=1))


if __name__ == "__main__":
    main()
