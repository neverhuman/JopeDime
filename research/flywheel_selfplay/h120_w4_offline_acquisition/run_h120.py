"""H120 — W4 offline acquisition screen on fully labelled proposal batches (CPU, ZERO native fits).

W4's cumulative native-fit budget (<= 2,400) is spent: H96/H103/H112/H116 used 2,534. Two prospective two-plus-two
tests (H103, H116) were NO_ADVANCE; in H116 the update changed the revealed four in only 11 of 150 units. Before any
further budget request, H120 screens other reveal rules OFFLINE on the 300 proposal batches whose 16 proposals were
ALL evaluated by H110 (H67 roster x seeds 801-803) and H116 (seeds 901-903). Every reveal reads a cached outcome of a
program the rule asked for; no rule sees an unrevealed label.

Preregistered here (PREREG.md is the prose copy; the frozen source is authoritative):
* Prior: the merged H112 prior exactly (H112 module, sha256 de7d5882...: memory build, per-fold H106-recipe scorer,
  out-of-fold utility slope b_k); a context is scored by the model that excluded its whole family. Uncertainty: the
  standard deviation of the four inner-fold models (folds j != k, H112 seeds 1130+10k+j), times b_k.
* Rules (four reveals each, zero history, ties by program SHA):
    S_TOP4       the prior's top four (anchor);
    GP_2PLUS2    the H112 rule (merged code, unchanged);
    GP_1PLUS3    reveal the top one; clipped residual (no centring) drives the same RBF-GP posterior; reveal top 3 of rest;
    UCB_2PLUS2   first pair = top two by mu + sd; then the H112 level-free update; reveal the top two of the rest;
    random4      H112's seeded four (h112-random, so it reproduces H116); ORACLE4 = best possible four (ceiling only).
* Endpoint: best utility among the four. Seeds averaged within context (6 batches: 801-803 and 901-903), contexts
  within family (44 families).
* Primaries (Bonferroni m = 3, one-sided .05/3 LCB > +.001, median >= 0, <= 10% families losing > .01):
  GP_2PLUS2 - S_TOP4, GP_1PLUS3 - S_TOP4, UCB_2PLUS2 - S_TOP4. Family Elo with S_TOP4 = 1500.
* Decision: a rule that passes becomes a CANDIDATE for a budgeted prospective W4 test (owner budget call); if none
  passes, W4 adaptive reveal at 4-of-16 is CLOSED. This is adaptive development on batches already seen in aggregate.
* Integrity gate (before any new rule is reported): on the 150 H116 batches, S_TOP4 and GP_2PLUS2 must reproduce
  H116's published best-of-four values exactly (rows_h112.jsonl sha256 341e20a4...). Otherwise the run stops.
No native fit, DL-Val, outer-test, sealed or holdout read; DOPE not loaded; zero GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
H112_PATH = HERE.parent / "h112_w4_active_reveal_scorer" / "run_h112.py"
H112_SHA = "de7d58826a70b9e5279488594bbc7cc7e7d38224a037f56204b04781b2815579"
BATCH_SOURCES = {  # name: (seeds, stream, fits.jsonl sha256)
    "H110": ((801, 802, 803), "h106-stream", "239b9d265e2cfe30ba48f42085f12009c497e04ba5eb0a695a2fe2689e67c00f"),
    "H116": ((901, 902, 903), "h112-stream", "5dc0b404e254eb7aa2ebc1b4f7bd339e989bcb286293acd0d22578b04615cc16"),
}
DRY_SOURCE = ((1001,), "h114-stream", "08a2b66184c1ff4014bb6cbc65cf7e0594c05057e36580ea061cda56881cce72")  # H118 panel
H116_ROWS_SHA = "341e20a4e7453c554a03fc1a5257c7d3909d64eccb969b11cc05b7d55ebd8c67"
RULES = ("S_TOP4", "GP_2PLUS2", "GP_1PLUS3", "UCB_2PLUS2", "random4")
ANCHOR = "S_TOP4"
PRIMARY = (("GP_2PLUS2", "S_TOP4"), ("GP_1PLUS3", "S_TOP4"), ("UCB_2PLUS2", "S_TOP4"))
CLIP = 1.0


def load_h112():
    if hashlib.sha256(H112_PATH.read_bytes()).hexdigest() != H112_SHA:
        raise SystemExit("H112 runner bytes differ from the merged source")
    spec = importlib.util.spec_from_file_location("h112_run", H112_PATH)
    m = importlib.util.module_from_spec(spec); sys.modules["h112_run"] = m; spec.loader.exec_module(m)
    return m


def load_fits(path: Path, want: str) -> dict:
    if hashlib.sha256(path.read_bytes()).hexdigest() != want:
        raise SystemExit(f"{path} differs from the published run")
    return {(r["context"], r["program_sha256"]): r for r in map(json.loads, path.read_text().splitlines())}


def one_plus_three(h81, mu, x, ids, scale, reveal) -> list[int] | None:
    top = h81.ranked(mu, ids)
    fu = reveal(top[:1])
    if fu[0] is None:
        return None
    r = [float(np.clip(fu[0] - mu[top[0]], -CLIP, CLIP))]
    return top[:1] + h81.ranked(h81.posterior_mean(mu, x, top[:1], r, scale), ids, {top[0]})[:3]


def ucb_two_plus_two(h112, h81, mu, sd, x, ids, scale, reveal) -> list[int] | None:
    first = h81.ranked(mu + sd, ids)[:2]
    fu = reveal(first)
    if any(v is None for v in fu):
        return None
    return first + h112.second_pair(h81, mu, x, ids, first, fu, scale, False)


def screen_batch(h112, h81, mu, sd, x, ids, scale, u, rng_seed) -> dict:
    """Every rule gets a `reveal` that returns cached outcomes ONLY for the indices it asks for."""
    log = defaultdict(list)

    def reveal_for(rule):
        def reveal(sel):
            log[rule].extend(sel); return [u[i] for i in sel]
        return reveal
    base = h112.select(h81, mu, x, ids, scale, rng_seed, reveal_for("GP_2PLUS2"))
    picks = dict(S_TOP4=base["s_top4"], GP_2PLUS2=base["s_2plus2"], random4=base["random4"],
                 GP_1PLUS3=one_plus_three(h81, mu, x, ids, scale, reveal_for("GP_1PLUS3")),
                 UCB_2PLUS2=ucb_two_plus_two(h112, h81, mu, sd, x, ids, scale, reveal_for("UCB_2PLUS2")))
    for rule, sel in picks.items():
        if sel is not None and (len(set(sel)) != 4 or not set(log[rule]) <= set(sel)):
            raise SystemExit(f"{rule} revealed outside its own picks")
    out = {r: (max(u[i] for i in sel) if sel is not None and all(u[i] is not None for i in sel) else None) for r, sel in picks.items()}
    out["ORACLE4"] = max(v for v in u if v is not None)
    return out


def inner_models(h112, h106, prop, memory: list[dict]) -> dict:
    """The four inner-fold models per outer fold (H112's seeds), for the ensemble standard deviation."""
    from threadpoolctl import threadpool_limits
    out = {}
    with threadpool_limits(limits=8):
        for k in range(5):
            train = [r for r in memory if r["fold"] != k]
            out[k] = [h112.fit_gbm(h106, prop, [r for r in train if r["fold"] != j], 1130 + 10 * k + j)
                      for j in sorted({r["fold"] for r in train})]
    return out


def screen(h112, h106, h81, prop, priors, inner, contexts, source, fits) -> list[dict]:
    sys.path.insert(0, str(h81.MICRO / "python"))
    from rust_evaluator import get_w1_sign_matrix
    W = get_w1_sign_matrix().astype(np.float64)
    seeds, stream, _ = source
    rows = []
    for c in contexts:
        k, p = prop.fold[c], h81.pool(c)
        scale = h81.lengthscale((p["program"].astype(np.float64) @ W) / math.sqrt(412))
        for seed in seeds:
            props = prop.batch(c, seed, stream)
            ids, dsls = [q["program_sha256"] for q in props], [q["dsl"] for q in props]
            missing = [s for s in ids if (c, s) not in fits]
            if len(props) < h106.N_PROPOSALS or missing:
                rows.append(dict(context=c, seed=seed, status="unlabelled", missing=len(missing))); continue
            u = [h81.utility(*(fits[(c, s)][f] for f in ("status", "gate", "fitness"))) for s in ids]
            feats = prop.x(c, dsls)
            mu = priors["slopes"][k] * priors["models"][k].predict(feats)
            sd = priors["slopes"][k] * np.std(np.stack([m.predict(feats) for m in inner[k]]), axis=0)
            x = (feats[:, 412:].astype(np.float64) @ W) / math.sqrt(412)
            res = screen_batch(h112, h81, mu, sd, x, ids, scale, u, h81.hseed("h112-random", c, seed))
            rows.append(dict(context=c, family=prop.fam[c], seed=seed, status="screened", **res))
    return rows


def integrity(rows: list[dict], h116_rows: list[dict]) -> dict:
    pub = {(r["context"], r["seed"], r["arm"]): r["best_of_four"] for r in h116_rows if r["status"] == "completed"}
    mism, n = [], 0
    for r in rows:
        if r.get("seed") in BATCH_SOURCES["H116"][0] and r["status"] == "screened":
            for mine, theirs in (("S_TOP4", "s_top4"), ("GP_2PLUS2", "s_2plus2")):
                n += 1
                if r[mine] != pub.get((r["context"], r["seed"], theirs)):
                    mism.append((r["context"][:12], r["seed"], mine))
    return dict(checked=n, mismatches=len(mism), examples=mism[:3], passed=bool(n and not mism))


def analyze(h81, rows: list[dict], elo=None) -> dict:
    ok = [r for r in rows if r["status"] == "screened" and all(r[a] is not None for a in RULES)]
    fam = {r["context"]: r["family"] for r in ok}
    by = defaultdict(lambda: defaultdict(list))
    for r in ok:
        for a in RULES + ("ORACLE4",):
            by[r["context"]][a].append(r[a])
    cm = {c: {a: mean(v[a]) for a in RULES + ("ORACLE4",)} for c, v in by.items()}
    families = sorted({fam[c] for c in cm})
    fv = {f: {a: mean(cm[c][a] for c in cm if fam[c] == f) for a in RULES + ("ORACLE4",)} for f in families}
    draws = np.random.default_rng(h81.BOOTSTRAP_SEED).integers(0, len(families), size=(h81.BOOTSTRAP_DRAWS, len(families)))

    def con(a, b):
        v = np.array([fv[f][a] - fv[f][b] for f in families]); c = h81.contrast(v, draws)
        lcb = float(np.percentile(v[draws].mean(axis=1), 100 * 0.05 / len(PRIMARY)))
        c.update(adjusted_one_sided_lcb=lcb, common_gate_passed=bool(lcb > h81.GATE_LCB and c["family_median"] >= 0
                                                                    and c["families_losing_more_than_0_01_fraction"] <= 0.10))
        return c
    out = dict(batches=len(ok), unscreened=len(rows) - len(ok), contexts=len(cm), families=len(families),
               family_mean_best_of_four={a: mean(fv[f][a] for f in families) for a in RULES + ("ORACLE4",)},
               primary={f"{a} - {b}": con(a, b) for a, b in PRIMARY},
               picks_differ_from_top4={a: sum(1 for r in ok if r[a] != r["S_TOP4"]) for a in RULES if a != "S_TOP4"})
    if elo is not None:
        recs = [elo.DatasetEvalRecord(c, fam[c], "regression", {a: v[a] for a in RULES}) for c, v in sorted(cm.items())]
        out["family_elo"] = elo.compute_family_grouped_bradley_terry_elo(recs, anchor_contender=ANCHOR, anchor_elo=1500.0,
                                                                         tie_margin=h81.TIE, prior=0.01, n_iters=400)
    passed = [k for k, v in out["primary"].items() if v["common_gate_passed"]]
    out["verdict"] = (f"CANDIDATE for a budgeted prospective W4 test: {', '.join(passed)}" if passed
                      else "CLOSE W4 adaptive reveal at 4 of 16 (no rule passes the offline screen)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("integrity", "dry-run", "screen"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--h94-run", type=Path, required=True)
    ap.add_argument("--h103-run", type=Path, required=True)
    ap.add_argument("--h110-run", type=Path, required=True)
    ap.add_argument("--h112-run", type=Path, required=True, help="H116 results run (opus-h112-20261031a)")
    ap.add_argument("--h115-run", type=Path, required=True, help="H118 results run (dry-run batches only)")
    a = ap.parse_args()
    h112 = load_h112(); h106 = h112.load_h106(); h81 = h106.load_h81(); h81.verify_pins()
    prop = h106.Proposer(h81)
    a.out.mkdir(parents=True, exist_ok=True)
    memory = h112.full_memory(h106, h81, prop, a.h94_run, a.h103_run, a.h110_run)
    priors = h112.fit_priors(h106, prop, memory)
    inner = inner_models(h112, h106, prop, memory)
    receipt = dict(stage=a.stage, source_sha256=h81.sha256_file(Path(__file__)), h112_source_sha256=H112_SHA, native_fits=0,
                   gpu_seconds=0, memory_rows=len(memory), slopes=priors["slopes"])
    if a.stage == "dry-run":
        fits = load_fits(a.h115_run / "fits.jsonl", DRY_SOURCE[2])
        panel = json.loads((a.h115_run / "panel_manifest.json").read_text())["contexts"][:3]
        rows = screen(h112, h106, h81, prop, priors, inner, panel, DRY_SOURCE, fits)
        receipt.update(dry_run_contexts=panel, dry_run_rows=[{k: r[k] for k in ("seed", "status")} for r in rows],
                       note="dry-run rows: integrity of the harness only; never part of an H120 result")
    else:
        h116 = load_fits(a.h112_run / "fits.jsonl", BATCH_SOURCES["H116"][2])
        rows = screen(h112, h106, h81, prop, priors, inner, prop.roster, BATCH_SOURCES["H116"], h116)
        ref = a.h112_run / "rows_h112.jsonl"
        if h81.sha256_file(ref) != H116_ROWS_SHA:
            raise SystemExit("H116 rows differ from the published run")
        receipt["integrity"] = integrity(rows, [json.loads(l) for l in ref.read_text().splitlines()])
        if not receipt["integrity"]["passed"]:
            (a.out / f"{a.stage}_receipt.json").write_text(json.dumps(receipt, indent=1))
            raise SystemExit("integrity gate failed: the offline harness does not reproduce H116")
        if a.stage == "screen":
            h110 = load_fits(a.h110_run / "fits.jsonl", BATCH_SOURCES["H110"][2])
            rows += screen(h112, h106, h81, prop, priors, inner, prop.roster, BATCH_SOURCES["H110"], h110)
            with open(a.out / "rows_h120.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
            receipt.update(analyze(h81, rows, h81.load_elo()))
            (a.out / "metrics.json").write_text(json.dumps(receipt, indent=1))
    (a.out / f"{a.stage}_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("dry_run_rows",)}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
