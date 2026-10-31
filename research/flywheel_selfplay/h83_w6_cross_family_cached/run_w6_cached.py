"""H83 — W6 cross-family transfer, cached held-family screen (CPU, zero native fits).

Question (PROPOSALS.md W6): do conditional preferences across recurring programs and structural neighbours improve
new-family selection beyond the generic program-only prior and beyond current raw-lm12 retrieval?

Leakage-free stacking from W1's stored out-of-fold priors (w1_train_only_priors.npz, sha256 5213faa8...):
`cal_cost_prog412_by_fold[k]` holds, for outer fold k, inner-pair out-of-fold calibrated costs on every fit-family
slot and outer-model costs on fold k's held slots. Each fold-k residual model is fit only on fit-family residuals
u - prior and scored on fold-k held contexts; families never cross folds (W1 fold manifest, 0 cross-family table
aliases). All 1,219 DL-Train contexts / 205 families are scored once, out of fold.

Arms (fixed before any held-family read; same four hash-ordered anchors, same unrevealed slots, SHA tie-break):
  PRIOR    W1 generic program-only prior, mu = -cal_cost_prog412 (control, Elo anchor)
  TREAT    PRIOR + rank-8 ridge bilinear residual z_c^T W x_i (z = standardized lm48_z, x = standardized proj16;
           ridge lambda 10 on the 768 outer-product features, residuals centred within context; SVD rank 8)
  SHUF     TREAT fit with z permuted across fit contexts (placebo: identical capacity, no real context signal)
  KNN      PRIOR + mean centred residual of the 32 nearest fit-family slots in standardized proj16 (structural
           neighbours, context-free)
  RETR     current raw-lm12 retrieval: champions of the 4 nearest fit-family contexts by lm48_z[:12]; candidates scored
           by -min proj16 distance to those champions
  CTX      W1 context-aware prior, mu = -cal_cost_prog412_ctx (reference)
  RANDOM4  four unrevealed slots by sha256-keyed order (floor)
Endpoint: verified U4 = best utility among the four anchors and the arm's four chosen unrevealed slots (U0 = anchors
only). Primary (Bonferroni m=2, one-sided .05): TREAT-PRIOR and TREAT-RETR, each with the common gate. Secondary:
TREAT-SHUF, KNN-PRIOR, CTX-PRIOR, PRIOR-RANDOM4; exact-program coverage census. Family Elo via the canonical
elo_tracker (PRIOR = 1500). No DL-Val/outer/sealed/holdout rows; no native fits; no GPU; DOPE not involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUT = Path.home() / "jope-data/h50/input"
W1 = Path.home() / "jope-data/w1"
H58_CHAMPS = Path.home() / "jope-data/h58/cache/champions_featurized.json"
ELO_SOURCE = REPO / "python/src/jope/north_star/elo_tracker.py"
PINS = {
    "selected": "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1",
    "w1_fold_manifest": "dd211134bba127960b1612687478064969dca422e35dfa889790760f316961c1",
    "w1_priors": "5213faa8d0bcc906e1d3c3226d6104e3846cf11c9da9c222298e9f0dc2fb46d9",
    "w1_context_features": "d9e2275f425cdac1107b1f010c75b69d64dd3f7d29b41f3e1d5fd91bf128fca6",
    "champions_featurized": "ea020d489ec9a813e4dfb2b6fe45c126ff9f8d8a1b59d0ff9f8242cfe5e6efc0",
    "elo_tracker": "9ece98326c32453179b0c765eb45944b68ca9de944d2828a1d42dee1401d1835",
}
ARMS = ("PRIOR", "TREAT", "SHUF", "KNN", "RETR", "CTX", "RANDOM4")
ANCHOR = "PRIOR"
PRIMARY = (("TREAT", "PRIOR"), ("TREAT", "RETR"))
SECONDARY = (("TREAT", "SHUF"), ("KNN", "PRIOR"), ("CTX", "PRIOR"), ("PRIOR", "RANDOM4"))
RANK, RIDGE, KNN_K, FIT_SLOTS, RETR_CONTEXTS = 8, 10.0, 32, 192, 4
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10_000, 20261031
TIE, HARM, GATE_LCB = 1e-4, 0.01, 0.001


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hkey(*parts) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()


# ----------------------------------------------------------------------------- pure pieces

def u4(u: np.ndarray, score: np.ndarray, ids: list[str]) -> tuple[float, float]:
    """(U0, U4): best of the four anchors, and best of anchors plus the scorer's top-4 unrevealed (SHA tie-break)."""
    un = sorted(range(4, len(u)), key=lambda i: (-float(score[i]), ids[i]))[:4]
    return float(np.max(u[:4])), float(max(np.max(u[:4]), np.max(u[un])))


def fit_bilinear(z: np.ndarray, x: np.ndarray, r: np.ndarray, ridge: float = RIDGE, rank: int = RANK) -> np.ndarray:
    """Ridge regression of r on vec(z x^T); returns the rank-truncated weight matrix W (dim_z x dim_x)."""
    feats = (z[:, :, None] * x[:, None, :]).reshape(len(r), -1)
    a = feats.T @ feats + ridge * np.eye(feats.shape[1])
    beta = np.linalg.solve(a, feats.T @ r)
    w = beta.reshape(z.shape[1], x.shape[1])
    uu, s, vt = np.linalg.svd(w, full_matrices=False)
    return (uu[:, :rank] * s[:rank]) @ vt[:rank]


def centre_within(groups: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean()
    return out


def contrast(values: np.ndarray, draws: np.ndarray, pct: float) -> dict:
    boot = values[draws].mean(axis=1)
    lcb = float(np.percentile(boot, pct))
    med, harm = float(np.median(values)), float(np.mean(values < -HARM))
    return dict(families=len(values), family_mean=float(values.mean()), family_median=med,
                families_losing_more_than_0_01_fraction=harm,
                family_win_tie_loss=[int(np.sum(values > TIE)), int(np.sum(np.abs(values) <= TIE)), int(np.sum(values < -TIE))],
                adjusted_one_sided_lcb=lcb, lcb_percentile=pct,
                bootstrap_p2_5=float(np.percentile(boot, 2.5)), bootstrap_p97_5=float(np.percentile(boot, 97.5)),
                common_gate_passed=bool(lcb > GATE_LCB and med >= 0.0 and harm <= 0.10))


def summarize(ctx: dict, fam: dict, elo=None) -> dict:
    families = sorted({fam[c] for c in ctx})
    fv = {f: {a: mean(ctx[c][a] for c in ctx if fam[c] == f) for a in ARMS + ("U0",)} for f in families}
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(families), size=(BOOTSTRAP_DRAWS, len(families)))
    d = lambda a, b: np.array([fv[f][a] - fv[f][b] for f in families])
    out = dict(contexts=len(ctx), families=len(families),
               family_mean_U4={a: mean(fv[f][a] for f in families) for a in ARMS},
               family_mean_U0=mean(fv[f]["U0"] for f in families),
               context_mean_U4={a: mean(v[a] for v in ctx.values()) for a in ARMS},
               primary={f"{a} - {b}": contrast(d(a, b), draws, 100 * 0.05 / len(PRIMARY)) for a, b in PRIMARY},
               secondary={f"{a} - {b}": contrast(d(a, b), draws, 5.0) for a, b in SECONDARY})
    if elo is not None:
        recs = [elo.DatasetEvalRecord(c, fam[c], "regression", {a: v[a] for a in ARMS}) for c, v in sorted(ctx.items())]
        out["family_elo"] = elo.compute_family_grouped_bradley_terry_elo(recs, anchor_contender=ANCHOR, anchor_elo=1500.0,
                                                                         tie_margin=TIE, prior=0.01, n_iters=400)
    passed = all(v["common_gate_passed"] for v in out["primary"].values())
    out["verdict"] = "ADVANCE_CACHED_SCREEN (both primaries pass)" if passed else "NO_ADVANCE (primary gate not met)"
    return out


# ----------------------------------------------------------------------------- data

def verify_pins():
    paths = {"selected": INPUT / "selected.json", "w1_fold_manifest": W1 / "fold_manifest.json",
             "w1_priors": W1 / "cache/w1_train_only_priors.npz", "w1_context_features": W1 / "cache/train_only_context_features.npz",
             "champions_featurized": H58_CHAMPS, "elo_tracker": ELO_SOURCE}
    for k, p in paths.items():
        if sha256_file(p) != PINS[k]:
            raise SystemExit(f"pin mismatch: {k}")


def load():
    sel = json.loads((INPUT / "selected.json").read_text())["train"]
    if len(sel) != 1219 or any(r["dl_split"] != "DL-Train" for r in sel):
        raise SystemExit("DL-Train roster mismatch")
    cids = [r["context_sha256"] for r in sel]
    fam = {r["context_sha256"]: r["family_id"] for r in sel}
    fm = json.loads((W1 / "fold_manifest.json").read_text())
    if fm["cross_family_table_aliases"] != 0:
        raise SystemExit("table aliases cross families")
    fold = {c: k for k, f in enumerate(fm["folds"]) for c in f["held_contexts"]}
    z = np.load(W1 / "cache/w1_train_only_priors.npz", allow_pickle=False)
    if [c.decode() if isinstance(c, bytes) else str(c) for c in z["cids"]] != cids:
        raise SystemExit("W1 priors are not in selected.json train order")
    cf = np.load(W1 / "cache/train_only_context_features.npz", allow_pickle=False)
    if [c.decode() if isinstance(c, bytes) else str(c) for c in cf["cids"]] != cids:
        raise SystemExit("train-only context features are not in selected.json train order")
    pools = {}
    for i, c in enumerate(cids):
        p = np.load(INPUT / "pools" / f"{c}.npz", allow_pickle=False)
        lo, hi = int(z["offsets"][i]), int(z["offsets"][i + 1])
        if hi - lo != len(p["fitness"]):
            raise SystemExit(f"slot count mismatch {c}")
        gate = p["gate"].astype(bool)
        pools[c] = dict(u=np.where(gate, -p["fitness"].astype(np.float64), -2.0), sha=[s.decode() for s in p["program_sha256"]], lo=lo, hi=hi)
    return cids, fam, fold, z, cf["lm48_z"].astype(np.float64), pools


def run(out: Path):
    cids, fam, fold, z, lm, pools = load()
    idx = {c: i for i, c in enumerate(cids)}
    proj = z["proj16"].astype(np.float64)
    prior_by_fold = -z["cal_cost_prog412_by_fold"].astype(np.float64)
    ctx_by_fold = -z["cal_cost_prog412_ctx_by_fold"].astype(np.float64)
    champs = [r for r in json.loads(H58_CHAMPS.read_text()) if r["dl_split"] == "DL-Train" and r["context_sha256"] in idx]
    sha_ctx = defaultdict(set)  # exact-program coverage census
    for c in cids:
        for s in pools[c]["sha"][4:]:
            sha_ctx[s].add(c)
    from sklearn.neighbors import NearestNeighbors
    per_ctx, census, fit_receipts = {}, dict(held_slots=0, slots_seen_in_fit_family=0), []
    for k in range(5):
        fit = [c for c in cids if fold[c] != k]
        held = [c for c in cids if fold[c] == k]
        rows_z, rows_x, rows_r, rows_g, rows_c = [], [], [], [], []
        for c in fit:
            p = pools[c]
            sl = sorted(range(4, p["hi"] - p["lo"]), key=lambda i: hkey("h83-slot", c, p["sha"][i]))[:FIT_SLOTS]
            g = np.array([p["lo"] + i for i in sl])
            rows_x.append(proj[g]); rows_r.append(p["u"][sl] - prior_by_fold[k][g])
            rows_z.append(np.repeat(lm[idx[c]][None], len(sl), 0)); rows_g.append(np.full(len(sl), idx[c])); rows_c.append(c)
        X, R, G = np.concatenate(rows_x), np.concatenate(rows_r), np.concatenate(rows_g)
        Z = np.concatenate(rows_z)
        zm, zs = Z.mean(0), Z.std(0) + 1e-9
        xm, xs = X.mean(0), X.std(0) + 1e-9
        Zs, Xs, Rc = (Z - zm) / zs, (X - xm) / xs, centre_within(G, R)
        w_treat = fit_bilinear(Zs, Xs, Rc)
        perm = sorted(range(len(fit)), key=lambda j: hkey("h83-shuffle", k, fit[j]))
        zmap = {idx[fit[j]]: lm[idx[fit[perm[j]]]] for j in range(len(fit))}  # context j gets another context's z
        Zsh = (np.stack([zmap[int(g)] for g in G]) - zm) / zs
        w_shuf = fit_bilinear(Zsh, Xs, Rc)
        nn = NearestNeighbors(n_neighbors=KNN_K).fit(Xs)
        fit_receipts.append(dict(fold=k, fit_contexts=len(fit), held_contexts=len(held), fit_rows=int(len(R)),
                                 w_treat_fro=float(np.linalg.norm(w_treat)), w_shuf_fro=float(np.linalg.norm(w_shuf))))
        fit_lm12 = np.stack([lm[idx[c]][:12] for c in fit])
        champ_by_ctx = defaultdict(list)
        for r in champs:
            if fold[r["context_sha256"]] != k:
                champ_by_ctx[r["context_sha256"]].append(np.asarray(r["proj16"], dtype=np.float64))
        for c in held:
            p = pools[c]; g = np.arange(p["lo"], p["hi"]); u, ids = p["u"], p["sha"]
            xs_c = (proj[g] - xm) / xs
            zs_c = (lm[idx[c]] - zm) / zs
            prior = prior_by_fold[k][g]
            scores = {"PRIOR": prior, "CTX": ctx_by_fold[k][g],
                      "TREAT": prior + xs_c @ (w_treat.T @ zs_c), "SHUF": prior + xs_c @ (w_shuf.T @ zs_c)}
            _, nb = nn.kneighbors(xs_c)
            scores["KNN"] = prior + Rc[nb].mean(axis=1)
            near = np.argsort(np.sum((fit_lm12 - lm[idx[c]][:12]) ** 2, axis=1), kind="stable")[:RETR_CONTEXTS]
            cp = [v for j in near for v in champ_by_ctx[fit[j]]]
            scores["RETR"] = -np.min(np.linalg.norm(proj[g][:, None, :] - np.stack(cp)[None], axis=2), axis=1) if cp else np.zeros(len(g))
            scores["RANDOM4"] = np.array([-int(hkey("h83-random", c, s)[:12], 16) for s in ids], dtype=np.float64)
            res = {}
            for a, sc in scores.items():
                u0, v = u4(u, sc, ids)
                res[a] = v
            res["U0"] = u0
            per_ctx[c] = res
            census["held_slots"] += len(ids) - 4
            census["slots_seen_in_fit_family"] += sum(1 for s in ids[4:] if any(fold[o] != k for o in sha_ctx[s] if o != c))
    with open(out / "per_context.json", "w") as f:
        json.dump(dict(per_context=per_ctx, family={c: fam[c] for c in per_ctx}, fit_receipts=fit_receipts, census=census), f)
    return per_ctx, fam, fit_receipts, census


def load_elo():
    import importlib.util
    spec = importlib.util.spec_from_file_location("h83_elo", ELO_SOURCE)
    m = importlib.util.module_from_spec(spec); sys.modules["h83_elo"] = m; spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    verify_pins()
    a.out.mkdir(parents=True, exist_ok=True)
    per_ctx, fam, receipts, census = run(a.out)
    res = summarize(per_ctx, fam, load_elo())
    res.update(run_id=a.run_id, fit_receipts=receipts, exact_program_coverage=census,
               config=dict(rank=RANK, ridge=RIDGE, knn_k=KNN_K, fit_slots_per_context=FIT_SLOTS, retrieval_contexts=RETR_CONTEXTS),
               source_sha256=sha256_file(Path(__file__)), pins=PINS, native_fits=0, gpu_seconds=0)
    (a.out / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("contexts", "families", "family_mean_U4", "family_mean_U0", "primary", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
