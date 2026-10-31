"""H92 — W6 downside-controlled learner on the exact H83 cached harness (CPU, zero native fits).

Every H83 arm failed the common gate's 10% harm tail, and TREAT's mean gain sat in a minority of families (median
0.0000). H92 asks whether a learner that limits per-family downside clears the gate. Arms are fixed here, before any
H92 outcome, but the design is informed by H83 on the same 205 families: adaptive development, not fresh confirmation.

Harness (imported from the merged H83 module, not copied): W1 folds, 1,219 DL-Train contexts / 205 families scored out
of fold, the same four hash-ordered anchors, verified U4, the same stacked prior / rank-8 bilinear TREAT / shuffled SHUF
/ W1 CTX scorers. Integrity: before any new arm is reported, the PRIOR, TREAT, SHUF and CTX U4 values recomputed here
must equal H83's stored per-context values exactly.

Controlled arms (s = TREAT, CTX or SHUF scores; ties broken by program SHA):
  HEDGE_s  picks = the prior's top-2 unrevealed + s's top-2 among the rest   (two picks always shared with PRIOR)
  SHORT_s  picks = s's top-4 within the prior's top-16 unrevealed shortlist (no promotion of prior-implausible slots)
Primaries (Bonferroni m=4, one-sided .05): HEDGE_TREAT, SHORT_TREAT, HEDGE_CTX, SHORT_CTX, each minus PRIOR, with the
common gate (adjusted LCB > +.001, median >= 0, <= 10% of families losing > .01). Secondary: HEDGE_TREAT - HEDGE_SHUF
(placebo), HEDGE_CTX - CTX and SHORT_CTX - CTX (downside cost of control). Family Elo anchored at PRIOR; family
bootstrap median and mean 95% intervals. No DL-Val/outer/sealed/holdout rows; DOPE not loaded; no GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
H83_PATH = HERE.parent / "h83_w6_cross_family_cached" / "run_w6_cached.py"
H83_SHA = "92178e4b0ce04ab73b43eb5e1835902081d062457b628e12314549d7b2e40068"  # merged H83 runner that produced H83's results
SHORTLIST = 16
ARMS = ("PRIOR", "TREAT", "CTX", "SHUF", "HEDGE_TREAT", "SHORT_TREAT", "HEDGE_CTX", "SHORT_CTX", "HEDGE_SHUF")
PRIMARY = (("HEDGE_TREAT", "PRIOR"), ("SHORT_TREAT", "PRIOR"), ("HEDGE_CTX", "PRIOR"), ("SHORT_CTX", "PRIOR"))
SECONDARY = (("HEDGE_TREAT", "HEDGE_SHUF"), ("HEDGE_CTX", "CTX"), ("SHORT_CTX", "CTX"), ("TREAT", "PRIOR"), ("CTX", "PRIOR"))


def load_h83():
    src = H83_PATH.read_bytes()
    if hashlib.sha256(src).hexdigest() != H83_SHA:
        raise SystemExit("H83 runner bytes differ from the merged, result-producing source")
    spec = importlib.util.spec_from_file_location("h83_run", H83_PATH)
    m = importlib.util.module_from_spec(spec); sys.modules["h83_run"] = m; spec.loader.exec_module(m)
    return m


def ranked_unrevealed(score: np.ndarray, ids: list[str], pool=None) -> list[int]:
    cand = range(4, len(ids)) if pool is None else pool
    return sorted(cand, key=lambda i: (-float(score[i]), ids[i]))


def hedge_picks(prior: np.ndarray, s: np.ndarray, ids: list[str]) -> list[int]:
    keep = ranked_unrevealed(prior, ids)[:2]
    rest = [i for i in ranked_unrevealed(s, ids) if i not in keep][:2]
    return keep + rest


def short_picks(prior: np.ndarray, s: np.ndarray, ids: list[str], k: int = SHORTLIST) -> list[int]:
    shortlist = ranked_unrevealed(prior, ids)[:k]
    return ranked_unrevealed(s, ids, shortlist)[:4]


def retained_best(u: np.ndarray, picks: list[int]) -> float:
    if len(set(picks)) != 4 or any(i < 4 for i in picks):
        raise ValueError("an arm must reveal four distinct unrevealed slots")
    return float(max(np.max(u[:4]), np.max(u[picks])))


def summarize(h83, ctx: dict, fam: dict, elo=None) -> dict:
    families = sorted({fam[c] for c in ctx})
    fv = {f: {a: mean(ctx[c][a] for c in ctx if fam[c] == f) for a in ARMS} for f in families}
    draws = np.random.default_rng(h83.BOOTSTRAP_SEED).integers(0, len(families), size=(h83.BOOTSTRAP_DRAWS, len(families)))
    d = lambda a, b: np.array([fv[f][a] - fv[f][b] for f in families])

    def with_median(a, b, pct):
        v = d(a, b); c = h83.contrast(v, draws, pct); med = np.median(v[draws], axis=1)
        c["median_ci95"] = [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]
        c["exact_tie_fraction"] = float(np.mean(np.abs(v) <= h83.TIE))
        return c
    out = dict(contexts=len(ctx), families=len(families),
               family_mean_U4={a: mean(fv[f][a] for f in families) for a in ARMS},
               primary={f"{a} - {b}": with_median(a, b, 100 * 0.05 / len(PRIMARY)) for a, b in PRIMARY},
               secondary={f"{a} - {b}": with_median(a, b, 5.0) for a, b in SECONDARY})
    if elo is not None:
        recs = [elo.DatasetEvalRecord(c, fam[c], "regression", {a: v[a] for a in ARMS}) for c, v in sorted(ctx.items())]
        out["family_elo"] = elo.compute_family_grouped_bradley_terry_elo(recs, anchor_contender="PRIOR", anchor_elo=1500.0,
                                                                         tie_margin=h83.TIE, prior=0.01, n_iters=400)
    passed = [k for k, v in out["primary"].items() if v["common_gate_passed"]]
    out["primaries_passing"] = passed
    out["verdict"] = (f"ADVANCE_CACHED_SCREEN ({', '.join(passed)})" if passed else "NO_ADVANCE (no primary passes the common gate)")
    return out


def run(h83, out: Path, h83_per_context: Path) -> tuple[dict, dict, dict]:
    """Mirror of h83.run()'s scoring loop (same objects, same order); returns per-context U4 for all H92 arms."""
    from sklearn.neighbors import NearestNeighbors  # noqa: F401  (h83 import side-effects identical)
    cids, fam, fold, z, lm, pools = h83.load()
    idx = {c: i for i, c in enumerate(cids)}
    proj = z["proj16"].astype(np.float64)
    prior_by_fold = -z["cal_cost_prog412_by_fold"].astype(np.float64)
    ctx_by_fold = -z["cal_cost_prog412_ctx_by_fold"].astype(np.float64)
    per_ctx, receipts = {}, []
    for k in range(5):
        fit = [c for c in cids if fold[c] != k]
        held = [c for c in cids if fold[c] == k]
        rows_z, rows_x, rows_r, rows_g = [], [], [], []
        for c in fit:
            p = pools[c]
            sl = sorted(range(4, p["hi"] - p["lo"]), key=lambda i: h83.hkey("h83-slot", c, p["sha"][i]))[:h83.FIT_SLOTS]
            g = np.array([p["lo"] + i for i in sl])
            rows_x.append(proj[g]); rows_r.append(p["u"][sl] - prior_by_fold[k][g])
            rows_z.append(np.repeat(lm[idx[c]][None], len(sl), 0)); rows_g.append(np.full(len(sl), idx[c]))
        X, R, G, Z = np.concatenate(rows_x), np.concatenate(rows_r), np.concatenate(rows_g), np.concatenate(rows_z)
        zm, zs = Z.mean(0), Z.std(0) + 1e-9
        xm, xs = X.mean(0), X.std(0) + 1e-9
        Zs, Xs, Rc = (Z - zm) / zs, (X - xm) / xs, h83.centre_within(G, R)
        w_treat = h83.fit_bilinear(Zs, Xs, Rc)
        perm = sorted(range(len(fit)), key=lambda j: h83.hkey("h83-shuffle", k, fit[j]))
        zmap = {idx[fit[j]]: lm[idx[fit[perm[j]]]] for j in range(len(fit))}
        w_shuf = h83.fit_bilinear((np.stack([zmap[int(g)] for g in G]) - zm) / zs, Xs, Rc)
        receipts.append(dict(fold=k, fit_rows=int(len(R)), w_treat_fro=float(np.linalg.norm(w_treat)), w_shuf_fro=float(np.linalg.norm(w_shuf))))
        for c in held:
            p = pools[c]; g = np.arange(p["lo"], p["hi"]); u, ids = p["u"], p["sha"]
            xs_c, zs_c = (proj[g] - xm) / xs, (lm[idx[c]] - zm) / zs
            prior = prior_by_fold[k][g]
            sc = {"PRIOR": prior, "CTX": ctx_by_fold[k][g], "TREAT": prior + xs_c @ (w_treat.T @ zs_c),
                  "SHUF": prior + xs_c @ (w_shuf.T @ zs_c)}
            res = {a: h83.u4(u, sc[a], ids)[1] for a in ("PRIOR", "TREAT", "CTX", "SHUF")}
            for name, s in (("TREAT", sc["TREAT"]), ("CTX", sc["CTX"]), ("SHUF", sc["SHUF"])):
                res[f"HEDGE_{name}"] = retained_best(u, hedge_picks(prior, s, ids))
                if name != "SHUF":
                    res[f"SHORT_{name}"] = retained_best(u, short_picks(prior, s, ids))
            per_ctx[c] = res
    ref = json.loads(h83_per_context.read_text())["per_context"]
    mism = [(c, a) for c in per_ctx for a in ("PRIOR", "TREAT", "CTX", "SHUF") if per_ctx[c][a] != ref[c][a]]
    integrity = dict(contexts=len(per_ctx), arms_checked=["PRIOR", "TREAT", "CTX", "SHUF"], mismatches=len(mism))
    if mism:
        raise SystemExit(f"H83 reproduction failed on {len(mism)} context/arm values (e.g. {mism[:3]}); refusing to report")
    (out / "per_context.json").write_text(json.dumps(dict(per_context=per_ctx, family={c: fam[c] for c in per_ctx},
                                                          fit_receipts=receipts, integrity=integrity)))
    return per_ctx, {c: fam[c] for c in per_ctx}, dict(fit_receipts=receipts, integrity=integrity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--h83-per-context", type=Path, required=True, help="H83 run's per_context.json (S3 per_context.json.gz, decompressed)")
    a = ap.parse_args()
    h83 = load_h83()
    h83.verify_pins()
    a.out.mkdir(parents=True, exist_ok=True)
    per_ctx, fam, extra = run(h83, a.out, a.h83_per_context)
    res = summarize(h83, per_ctx, fam, h83.load_elo())
    res.update(run_id=a.run_id, **extra, shortlist=SHORTLIST, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               h83_source_sha256=hashlib.sha256(H83_PATH.read_bytes()).hexdigest(), native_fits=0, gpu_seconds=0,
               adaptive_development="design informed by H83 on the same 205 families; not fresh confirmation")
    (a.out / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("contexts", "families", "family_mean_U4", "primaries_passing", "verdict", "integrity")}, indent=1))


if __name__ == "__main__":
    main()
