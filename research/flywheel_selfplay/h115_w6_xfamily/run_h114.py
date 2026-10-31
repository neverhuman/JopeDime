"""H114 — W6 cross-family outcome transfer on NEW families (prospective, CPU): preregistration harness.

W6 asks whether outcomes observed on other families improve proposal selection on a new family beyond the
program-only prior and current retrieval. H83/H92 found context gains only on cached pools; H94 found them reversed on
new programs. H110 then showed that a scorer trained on native reveals of proposer programs beats random4 on held
families of the H67 roster. H114 tests transfer where W6 requires it: on DL-Train families that are not in the roster
at all, with a context-conditioned scorer against the generic scorer, current retrieval, the program prior, a
context-shuffle placebo and random selection.

Preregistered here (PREREG.md is the prose copy; the frozen source is authoritative):
* Donor memory: every logged native outcome on the H67 roster: the H106 memory (H94 + H103 reveal logs, recovered by
  stream replay through the merged H106 module) plus all H110 fits (fits.jsonl sha256 239b9d26...). Rows are
  de-duplicated per (context, program); conflicting utilities stop the build.
* Panel: 38 DL-Train contexts, one per family, from families outside the roster (so every donor is from another family;
  the W1 manifest records 0 cross-family table and dataset aliases). Families are taken in sha256("h114-panel|family")
  order and the context in sha256("h114-context|context") order. Contexts used by earlier dry runs (H96, H106, H112)
  are excluded. One seed (1001), 16 NEW proposals per context (stream "h114-stream", admitted pool excluded); all 16
  are evaluated natively, so every arm is scored on a fully labelled batch: 608 native fits. With H94's 556 and this
  prep's 16-fit dry run the W6 total is 1,180, inside its 1,200 ceiling.
* Arms (top 4 of 16 by score, ties by program SHA; zero history):
    X_SCORER   GBM on [prog412 - champion, prog412, lm48_z(context)] -> within-context utility percentile, trained on
               all donor rows (random_state 1131): conditional transfer.
    G_SCORER   the same without lm48_z (random_state 1130): the generic proposal prior (H110 recipe).
    X_SHUF     X_SCORER with each donor context's lm48_z replaced by another donor context's (sha256-keyed permutation):
               context-shuffle placebo.
    RETR       current raw-lm12 retrieval (H83 rule): -min proj16 distance to the champions of the 4 donor contexts
               nearest by lm48_z[:12].
    PROG_PRIOR the W1-recipe program-only refit (H94 models_prog.pkl) for the W1 fold of the context.
    random4    seeded uniform four (h114-random).
* Endpoint: best utility among the arm's four picks. One context per family, so family = context.
* Primaries (Bonferroni m = 2, one-sided .025 LCB > +.001, family median >= 0, <= 10% of families losing > .01;
  10,000-draw family bootstrap, seed 20261031): X_SCORER - G_SCORER and X_SCORER - RETR.
* ADVANCE requires both primaries AND X_SCORER - X_SHUF > 0 (family mean) AND, on the unseen-template panel,
  X_SCORER - G_SCORER >= 0 (family mean). Unseen-template panel: the batch's proposals whose template (DSL with
  numeric literals replaced) appears in no donor row; every arm picks its top 4 among them; contexts with < 6 such
  proposals are excluded; fewer than 10 qualifying families makes the template check UNIDENTIFIED, which withholds
  ADVANCE ("coverage too sparse"). Otherwise NO_ADVANCE (W6 kill rule).
* Secondary: G_SCORER - random4 (H110 transfer to new families), PROG_PRIOR - random4, RETR - random4; exact-program
  overlap census. Family Elo with G_SCORER = 1500. Completeness gate: all 38 contexts x 6 arms complete.
* Budget: 608 native fits, xbabe2, nice 15, load-gated; deadline 2026-11-01T18:00Z. Zero GPU.
No DL-Val, outer-test, sealed or holdout row is read; DOPE is not loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
H106_PATH = HERE.parent / "h106_w9_proposal_scorer_replay" / "run_h106.py"
H106_SHA = "13730ebfdf1a451157799a7d4a639ccae61ed88361044006adb0ff548978f7da"
H110_FITS_SHA = "239b9d265e2cfe30ba48f42085f12009c497e04ba5eb0a695a2fe2689e67c00f"
PROG_MODELS_SHA = "5938a35497ccde1a1489c4d6b13d122196aed16739a3272afea06d57d5fff692"
ARMS = ("X_SCORER", "G_SCORER", "X_SHUF", "RETR", "PROG_PRIOR", "random4")
ANCHOR = "G_SCORER"
PRIMARY = (("X_SCORER", "G_SCORER"), ("X_SCORER", "RETR"))
SECONDARY = (("X_SCORER", "X_SHUF"), ("G_SCORER", "random4"), ("PROG_PRIOR", "random4"), ("RETR", "random4"))
N_PANEL, SEED, STREAM, MIN_UNSEEN, MIN_UNSEEN_FAMILIES = 38, 1001, "h114-stream", 6, 10
EXCLUDED_DRY_RUN_SALTS = ("h96-dry", "h106-dry")  # H112 reused the H106 dry-run contexts
FIT_CAP, DEADLINE = 608, "2026-11-01T18:00:00Z"


def load_h106():
    if hashlib.sha256(H106_PATH.read_bytes()).hexdigest() != H106_SHA:
        raise SystemExit("H106 runner bytes differ from the merged source")
    spec = importlib.util.spec_from_file_location("h106_run", H106_PATH)
    m = importlib.util.module_from_spec(spec); sys.modules["h106_run"] = m; spec.loader.exec_module(m)
    return m


def hk(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def template(dsl: str) -> str:
    return re.sub(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", "#", dsl)


def panel(prop, n: int = N_PANEL) -> list[str]:
    roster_fam = {prop.fam[c] for c in prop.roster}
    excluded = set()
    for salt in EXCLUDED_DRY_RUN_SALTS:
        pool = [c for c in prop.idx if prop.fam[c] not in roster_fam]
        excluded |= set(sorted(pool, key=lambda c: hashlib.sha256(f"{salt}|{c}".encode()).hexdigest())[:3])
    by_fam = defaultdict(list)
    for c in prop.idx:
        if prop.fam[c] not in roster_fam and c not in excluded:
            by_fam[prop.fam[c]].append(c)
    fams = sorted(by_fam, key=lambda f: hk("h114-panel", f))[:n]
    return [sorted(by_fam[f], key=lambda c: hk("h114-context", c))[0] for f in fams]


def donor_memory(h106, h81, prop, h94_dir: Path, h103_dir: Path, h110_dir: Path) -> list[dict]:
    path = h110_dir / "fits.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != H110_FITS_SHA:
        raise SystemExit("H110 fits.jsonl differs from the published run")
    fits = {(r["context"], r["program_sha256"]): r for r in map(json.loads, path.read_text().splitlines())}
    rows = h106.build_memory(prop, {"h94": h94_dir, "h103": h103_dir})
    for c in prop.roster:
        for seed in (801, 802, 803):
            for q in prop.batch(c, seed, "h106-stream"):
                f = fits.get((c, q["program_sha256"]))
                if f is None:
                    raise SystemExit("replayed H110 proposal without a logged fit")
                u = h81.utility(f["status"], f["gate"], f["fitness"])
                if u is not None:
                    rows.append(dict(source="H110", context=c, family=prop.fam[c], fold=prop.fold[c],
                                     program_sha256=q["program_sha256"], dsl=q["dsl"], utility=u))
    out = {}
    for r in rows:
        key = (r["context"], r["program_sha256"])
        if key in out and abs(out[key]["utility"] - r["utility"]) > 1e-9:
            raise SystemExit(f"inconsistent utilities for one program in one context: {key}")
        out.setdefault(key, dict(r, source="DONOR"))
    return sorted(out.values(), key=lambda r: (r["context"], r["program_sha256"]))


def context_permutation(contexts: list[str]) -> dict[str, str]:
    order = sorted(contexts)
    perm = sorted(order, key=lambda c: hk("h114-shuffle", c))
    return dict(zip(order, perm))


def fit_scorers(h106, prop, lm: np.ndarray, memory: list[dict], threads: int = 8) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    by = defaultdict(list)
    for r in memory:
        by[r["context"]].append(r)
    perm = context_permutation(sorted(by))
    X, Y, Z, ZS = [], [], [], []
    for c, g in sorted(by.items()):
        X.append(prop.x(c, [r["dsl"] for r in g])); Y.append(h106.percentile_target(np.array([r["utility"] for r in g])))
        Z.append(np.repeat(lm[prop.idx[c]][None], len(g), 0)); ZS.append(np.repeat(lm[prop.idx[perm[c]]][None], len(g), 0))
    X, Y, Z, ZS = map(np.concatenate, (X, Y, Z, ZS))
    gbm = lambda seed: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)
    with threadpool_limits(limits=threads):
        return dict(G_SCORER=gbm(1130).fit(X, Y), X_SCORER=gbm(1131).fit(np.hstack([X, Z]), Y),
                    X_SHUF=gbm(1131).fit(np.hstack([X, ZS]), Y), donor_contexts=sorted(by), rows=len(Y),
                    shuffled_contexts=sum(1 for a, b in perm.items() if a != b))


def retrieval_scores(prop, lm, W, donors: list[str], c: str, raw: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(lm[[prop.idx[x] for x in donors], :12] - lm[prop.idx[c], :12], axis=1)
    near = [donors[i] for i in np.argsort(d, kind="stable")[:4]]
    champs = np.stack([np.asarray(prop.feat(ch["dsl"])[0], dtype=np.float64) for x in near for ch in prop.champ[x]])
    cp, xp = champs @ W / math.sqrt(412), raw.astype(np.float64) @ W / math.sqrt(412)
    return -np.min(np.linalg.norm(xp[:, None, :] - cp[None, :, :], axis=2), axis=1)


def picks_for(scores: dict, ids: list[str], ranked, rng_seed: int, subset=None) -> dict:
    pool = list(range(len(ids))) if subset is None else list(subset)
    out = {}
    for a in ARMS:
        if a == "random4":
            out[a] = [pool[i] for i in np.random.default_rng(rng_seed).choice(len(pool), 4, replace=False)]
        else:
            out[a] = ranked(np.asarray(scores[a]), ids, set(range(len(ids))) - set(pool))[:4]
    return out


def evaluate(h106, h81, prop, lm, scorers, prog, memory, contexts, out: Path, run_id: str, workers: int, tag: str) -> list[dict]:
    sys.path.insert(0, str(h81.MICRO / "python"))
    from rust_evaluator import get_w1_sign_matrix
    W = get_w1_sign_matrix().astype(np.float64)
    seen_t = {template(r["dsl"]) for r in memory}
    seen_p = {r["program_sha256"] for r in memory}
    ledger, store = h81.Ledger(out / "ledger.json", cap=FIT_CAP, deadline=DEADLINE), h81.FitStore(out / "fits.jsonl", run_id)
    rows, census = [], []
    for c in contexts:
        props = prop.batch(c, SEED, STREAM)
        if len(props) < h106.N_PROPOSALS:
            rows.append(dict(tag=tag, context=c, family=prop.fam[c], arm="*", panel="all", status="shortfall")); continue
        ids, dsls = [q["program_sha256"] for q in props], [q["dsl"] for q in props]
        feats = prop.x(c, dsls)
        z = np.repeat(lm[prop.idx[c]][None], len(ids), 0)
        scores = dict(X_SCORER=scorers["X_SCORER"].predict(np.hstack([feats, z])), G_SCORER=scorers["G_SCORER"].predict(feats),
                      X_SHUF=scorers["X_SHUF"].predict(np.hstack([feats, z])),
                      RETR=retrieval_scores(prop, lm, W, scorers["donor_contexts"], c, feats[:, 412:]),
                      PROG_PRIOR=prog[prop.fold[c]].predict(feats[:, 412:]))
        unseen = [i for i, d in enumerate(dsls) if template(d) not in seen_t]
        picks = {"all": picks_for(scores, ids, h81.ranked, h81.hseed("h114-random", c, SEED))}
        if len(unseen) >= MIN_UNSEEN:
            picks["unseen"] = picks_for(scores, ids, h81.ranked, h81.hseed("h114-random-unseen", c, SEED), unseen)
        res = store.run([(c, d) for d in dsls], out / "scratch", ledger, workers)  # every pick is fixed above
        u = [h81.utility(*(res[(c, s)][f] for f in ("status", "gate", "fitness"))) for s in ids]
        ok = all(v is not None for v in u)
        census.append(dict(context=c, unseen_templates=len(unseen), exact_program_overlap=sum(s in seen_p for s in ids)))
        for pname, pk in picks.items():
            for a, sel in pk.items():
                rows.append(dict(tag=tag, context=c, family=prop.fam[c], arm=a, panel=pname, status="completed" if ok else "incomplete",
                                 best_of_four=max(u[i] for i in sel) if ok else None, revealed=[ids[i] for i in sel]))
        rows.append(dict(tag=tag, context=c, family=prop.fam[c], arm="ORACLE4", panel="all", status="completed" if ok else "incomplete",
                         best_of_four=max(u) if ok else None, revealed=None))
    with open(out / f"rows_{tag.lower()}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    (out / f"census_{tag.lower()}.json").write_text(json.dumps(census, indent=1))
    return rows


def analyze(h81, rows: list[dict], panel_contexts: list[str], elo=None) -> dict:
    rows = [r for r in rows if r["tag"] == "H114"]
    if not rows:
        raise SystemExit("no H114 rows: dry-run rows are never analyzed")

    def table(pname):
        v = defaultdict(dict)
        for r in rows:
            if r["panel"] == pname and r["status"] == "completed" and r["arm"] in ARMS:
                v[r["family"]][r["arm"]] = r["best_of_four"]
        return {f: a for f, a in v.items() if set(a) == set(ARMS)}
    fam_of = {r["context"]: r["family"] for r in rows}
    allp, unseen = table("all"), table("unseen")
    families = sorted(allp)
    draws = np.random.default_rng(h81.BOOTSTRAP_SEED).integers(0, max(len(families), 1), size=(h81.BOOTSTRAP_DRAWS, max(len(families), 1)))

    def con(a, b, adjusted):
        v = np.array([allp[f][a] - allp[f][b] for f in families]); c = h81.contrast(v, draws)
        if adjusted:
            lcb = float(np.percentile(v[draws].mean(axis=1), 2.5))
            c.update(adjusted_one_sided_lcb=lcb, common_gate_passed=bool(lcb > h81.GATE_LCB and c["family_median"] >= 0
                                                                        and c["families_losing_more_than_0_01_fraction"] <= 0.10))
        med = np.median(v[draws], axis=1)
        c["median_ci95"] = [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]
        return c
    complete = len(families) == len(panel_contexts) and {fam_of.get(c) for c in panel_contexts} == set(families)
    out = dict(families=len(families), panel_contexts=len(panel_contexts),
               family_mean_best_of_four={a: float(np.mean([allp[f][a] for f in families])) for a in ARMS} if families else {})
    if not complete:
        out["verdict"] = f"INCOMPLETE ({len(families)} of {len(panel_contexts)} panel families complete); no verdict"
        return out
    out.update(primary={f"{a} - {b}": con(a, b, True) for a, b in PRIMARY},
               secondary={f"{a} - {b}": con(a, b, False) for a, b in SECONDARY})
    uf = sorted(unseen)
    out["unseen_template_panel"] = dict(families=len(uf), x_minus_g_family_mean=float(np.mean([unseen[f]["X_SCORER"] - unseen[f]["G_SCORER"] for f in uf])) if uf else None)
    if elo is not None:
        recs = [elo.DatasetEvalRecord(f, f, "regression", allp[f]) for f in families]
        out["family_elo"] = elo.compute_family_grouped_bradley_terry_elo(recs, anchor_contender=ANCHOR, anchor_elo=1500.0,
                                                                         tie_margin=h81.TIE, prior=0.01, n_iters=400)
    p_ok = all(v["common_gate_passed"] for v in out["primary"].values())
    placebo = out["secondary"]["X_SCORER - X_SHUF"]["family_mean"] > 0
    if len(uf) < MIN_UNSEEN_FAMILIES:
        tmpl = "UNIDENTIFIED"
    else:
        tmpl = "PASS" if out["unseen_template_panel"]["x_minus_g_family_mean"] >= 0 else "FAIL"
    out["checks"] = dict(primaries=p_ok, placebo=placebo, unseen_templates=tmpl)
    out["verdict"] = "ADVANCE" if (p_ok and placebo and tmpl == "PASS") else (
        "WITHHELD (unseen-template coverage too sparse)" if (p_ok and placebo and tmpl == "UNIDENTIFIED") else "NO_ADVANCE")
    return out


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
    contexts = panel(prop)
    fm = json.loads((h81.W1 / "fold_manifest.json").read_text())
    if fm["cross_family_table_aliases"] != 0 or fm["cross_family_dataset_aliases"] != 0:
        raise SystemExit("W1 manifest records cross-family aliases; donor exclusion would need alias resolution")
    if {prop.fam[c] for c in contexts} & {prop.fam[c] for c in prop.roster} or len({prop.fam[c] for c in contexts}) != len(contexts):
        raise SystemExit("panel must hold one context per family, all outside the roster")
    receipt = dict(run_id=a.run_id, stage=a.stage, source_sha256=h81.sha256_file(Path(__file__)), h106_source_sha256=H106_SHA,
                   h110_fits_sha256=H110_FITS_SHA, prog_models_sha256=PROG_MODELS_SHA, gpu_seconds=0)
    if a.stage == "analyze":
        rows = [json.loads(l) for l in (a.out / "rows_h114.jsonl").read_text().splitlines() if l.strip()]
        receipt.update(analyze(h81, rows, contexts, h81.load_elo()))
        (a.out / "metrics.json").write_text(json.dumps(receipt, indent=1)); print(receipt["verdict"]); return
    (a.out / "panel_manifest.json").write_text(json.dumps(dict(
        schema="jope.h114.panel/v1", seed=SEED, stream=STREAM, contexts=contexts, families=[prop.fam[c] for c in contexts],
        w1_folds=[prop.fold[c] for c in contexts], roster_families_excluded=len({prop.fam[c] for c in prop.roster}),
        cross_family_table_aliases=0, cross_family_dataset_aliases=0), indent=1))
    memory = donor_memory(h106, h81, prop, a.h94_run, a.h103_run, a.h110_run)
    lm = np.load(h81.W1 / "cache/train_only_context_features.npz", allow_pickle=False)["lm48_z"].astype(np.float64)
    scorers = fit_scorers(h106, prop, lm, memory)
    with open(a.out / "scorers.pkl", "wb") as f:
        pickle.dump({k: v for k, v in scorers.items() if k in ("G_SCORER", "X_SCORER", "X_SHUF")}, f, protocol=pickle.HIGHEST_PROTOCOL)
    if h81.sha256_file(Path.home() / "jope-data/h94/opus-h94-20261031a/models_prog.pkl") != PROG_MODELS_SHA:
        raise SystemExit("program-only models.pkl hash mismatch")
    with open(Path.home() / "jope-data/h94/opus-h94-20261031a/models_prog.pkl", "rb") as f:
        prog = pickle.load(f)
    receipt.update(panel_manifest_sha256=h81.sha256_file(a.out / "panel_manifest.json"), donor_rows=scorers["rows"],
                   donor_contexts=len(scorers["donor_contexts"]), shuffled_donor_contexts=scorers["shuffled_contexts"],
                   scorers_sha256=h81.sha256_file(a.out / "scorers.pkl"))
    if a.stage == "dry-run":
        roster_fam = {prop.fam[c] for c in prop.roster}
        pool = [c for c in prop.idx if prop.fam[c] not in roster_fam]
        dry = sorted(pool, key=lambda c: hashlib.sha256(f"h106-dry|{c}".encode()).hexdigest())[:1]  # 16 fits: W6 budget
        if set(dry) & set(contexts):
            raise SystemExit("dry-run context overlaps the panel")
        rows = evaluate(h106, h81, prop, lm, scorers, prog, memory, dry, a.out, a.run_id, a.workers, "DRY_RUN")
        receipt.update(dry_run_contexts=dry, dry_run_rows=len(rows), dry_run_complete=sum(r["status"] == "completed" for r in rows),
                       ledger=h81.Ledger(a.out / "ledger.json", cap=FIT_CAP, deadline=DEADLINE).state,
                       note="DRY_RUN rows are integrity evidence only; never part of an H114 result")
    elif a.stage == "run":
        rows = evaluate(h106, h81, prop, lm, scorers, prog, memory, contexts, a.out, a.run_id, a.workers, "H114")
        receipt.update(rows=len(rows), ledger=h81.Ledger(a.out / "ledger.json", cap=FIT_CAP, deadline=DEADLINE).state)
    (a.out / f"{a.stage}_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
