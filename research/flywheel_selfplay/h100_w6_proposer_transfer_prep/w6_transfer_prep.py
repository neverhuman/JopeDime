#!/usr/bin/env python3
"""H100 — W6 Cross-Family Outcome Transfer: Complete Registered Design & 4-Node Real-S3 CPU Smoke.

Addresses the remaining W6 contract requirements in PROPOSALS.md (lines 143-151) and
diagnoses the H94 cached-to-prospective reversal on real S3 data (DL-Train only; zero
DL-Val, outer-test, sealed, or holdout reads; zero GPU):

1. Complete Registered W6 Arm Panel (1,219 DL-Train contexts / 205 families):
   - R882_CONTINUITY: frozen r882 continuity control (from S3 W1 remediation vectors)
   - RETR: current raw-lm12 4-nearest fit-family champion retrieval
   - PRIOR: W1 generic program412 prior (-cal_cost_prog412)
   - FIT_PROG_MEAN: registered fit-family program mean (empirical-Bayes shrinkage on exact
     program_sha256 matches across fit families with physical-table-alias exclusion;
     falls back cleanly to PRIOR features on unseen IDs rather than fabricated zeros)
   - FIT_TEMPLATE_MEAN: structural AST-template empirical-Bayes residual transfer across
     fit families (75.02% unrevealed slot coverage; falls back to PRIOR on unseen templates)
   - TREAT: equal-context rank-8 ridge bilinear residual (lm48_z x proj16, H83 continuity)
   - TREAT_FAMBAL: registered family-balanced rank-8 ridge bilinear residual (PROPOSALS.md
     line 147: observed outcomes with family-balanced weights, rank 8 fixed before scoring)
   - SHUF: context-shuffled rank-8 bilinear residual placebo
   - NEAR_CHAMP_TREAT: proposer-regime rank-8 family-balanced bilinear residual fitted on
     passing top-32 near-champion slots per fit context (matching the near-champion edit
     distribution of live GP proposals rather than deep-failure pool tails)
   - HEDGE_TREAT_FAMBAL: PRIOR top-2 + TREAT_FAMBAL top-2 among remaining unrevealed slots
   - HEDGE_NEAR_CHAMP: PRIOR top-2 + NEAR_CHAMP_TREAT top-2 among remaining unrevealed slots
   - CTX: W1 context-aware GBM prior (-cal_cost_prog412_ctx)
   - HEDGE_CTX: PRIOR top-2 + CTX top-2 among remaining unrevealed slots

2. Both Registered Decision Tracks & Template/Exact-ID Strata (PROPOSALS.md line 148):
   - Track 2 Verified Search U4 + family-grouped Bradley-Terry Elo (10,000 draws, seed 20261031)
   - Track 1 Legacy Selection val_reg_inner + family-grouped Bradley-Terry Elo
   - Within-context paired Seen-AST-Template vs Unseen-AST-Template U4 and val_reg_inner
     across all 1,219 DL-Train contexts (205 families), testing the W6 template kill rule
   - Separate reporting of Exact-Program transfer (7.04% covered slots) vs Feature-Neighbor
     transfer (92.96% uncovered slots), reconciling H84's H72-fold census (98,820 / 1,404,287)
     and H83's W1-fold unrevealed census (98,091 / 1,399,411)

3. Real-S3 Prospective New-Program Audit (H94 fits.jsonl.gz + H95 confirmed_hwm.json + H85 W1 models):
   - Scores H94's 150 context-seeds (50 contexts / 44 families) with the true persisted W1
     fold-grouped calibrated outer models (w1_outer_fold_models.pkl) vs H94's interleaved refits,
     plus W6 cross-family transfer models, and measures recall on H95's 30 resplit-confirmed
     and 27 material-confirmed high-water marks.

4. Fail-Closed Verdict Completeness Gate:
   - Rejects duplicate (context, seed, arm) rows, missing seeds, missing contexts/families,
     incomplete evaluations, or proposal shortfalls with verdict = "INCOMPLETE".
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
from statistics import mean
import sys
import time
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE
ELO_SOURCE = REPO / "python/src/jope/north_star/elo_tracker.py"

PINS = {
    "selected": "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1",
    "w1_fold_manifest": "dd211134bba127960b1612687478064969dca422e35dfa889790760f316961c1",
    "w1_priors": "5213faa8d0bcc906e1d3c3226d6104e3846cf11c9da9c222298e9f0dc2fb46d9",
    "w1_context_features": "d9e2275f425cdac1107b1f010c75b69d64dd3f7d29b41f3e1d5fd91bf128fca6",
    "champions_featurized": "ea020d489ec9a813e4dfb2b6fe45c126ff9f8d8a1b59d0ff9f8242cfe5e6efc0",
    "elo_tracker": "9ece98326c32453179b0c765eb45944b68ca9de944d2828a1d42dee1401d1835",
    "h72_train_roster": "9d3656d9817239d07ac27d1cb4f90d688bf3421b492e15b62ac078a9d183190d",
    "h72_fold_manifest": "129002bcd6cfc4b2f3a0b9e0562967cd4510dcbf88364c2d11d0bbe8801c2c73",
    "h85_w1_vectors": "120d34990001feab338a5ea856007c71fe012b9e272f674de74b9dc2bfa2dc3d",
    "h85_w1_outer_models": "2ebc4859d10cef0bfff9849cf01bc831ca9a4cb1222b1ad6dd956a5ef8f5b09f",
    "h94_fits_gz": "097c33f53e2ba04c6bf89982e2b43c4456c8d86e12b519f633db32c486bade71",
    "h94_rows_gz": "871aec0115119aaf4749c1374f2bf23a4137eb94aec79197ea6effd63722e0f7",
    "h95_confirmed_hwm": "8d5f1d2d5b95ea097640bc24a8a4331486d4478228e0ffeed6ea13f86afdce9a",
}

ARMS = (
    "R882_CONTINUITY",
    "RETR",
    "PRIOR",
    "FIT_PROG_MEAN",
    "FIT_TEMPLATE_MEAN",
    "TREAT",
    "TREAT_FAMBAL",
    "SHUF",
    "NEAR_CHAMP_TREAT",
    "HEDGE_TREAT_FAMBAL",
    "HEDGE_NEAR_CHAMP",
    "CTX",
    "HEDGE_CTX",
)

PRIMARY_CONTRASTS = (
    ("TREAT_FAMBAL", "PRIOR"),
    ("TREAT_FAMBAL", "RETR"),
    ("FIT_PROG_MEAN", "PRIOR"),
    ("HEDGE_NEAR_CHAMP", "PRIOR"),
)

SECONDARY_CONTRASTS = (
    ("TREAT_FAMBAL", "SHUF"),
    ("NEAR_CHAMP_TREAT", "PRIOR"),
    ("NEAR_CHAMP_TREAT", "SHUF"),
    ("HEDGE_TREAT_FAMBAL", "PRIOR"),
    ("HEDGE_TREAT_FAMBAL", "HEDGE_NEAR_CHAMP"),
    ("FIT_TEMPLATE_MEAN", "PRIOR"),
    ("TREAT", "PRIOR"),
    ("CTX", "PRIOR"),
    ("HEDGE_CTX", "PRIOR"),
    ("PRIOR", "R882_CONTINUITY"),
    ("HEDGE_CTX", "R882_CONTINUITY"),
)

RANK = 8
RIDGE = 10.0
FIT_SLOTS = 192
NEAR_CHAMP_SLOTS = 32
RETR_CONTEXTS = 4
EB_TAU_EXACT = 2.0
EB_TAU_TEMPLATE = 5.0
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261031
TIE = 1e-4
HARM_THRESHOLD = 0.01
GATE_LCB = 0.001

# Structural AST template feature columns in program412 (excluding numeric threshold quantiles 338..363)
STRUCTURAL_COLS = np.array(
    [i for i in range(412) if not (338 <= i < 364)], dtype=np.int64
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hkey(*parts: Any) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def compute_template_hashes(prog_matrix: np.ndarray) -> list[str]:
    """Compute structural AST template hashes from program412 structural coordinates."""
    struct = np.ascontiguousarray(
        np.rint(prog_matrix[:, STRUCTURAL_COLS] * 1000.0).astype(np.int32)
    )
    return [hashlib.sha256(row.tobytes()).hexdigest()[:16] for row in struct]


def fit_weighted_bilinear(
    z: np.ndarray,
    x: np.ndarray,
    r: np.ndarray,
    weights: np.ndarray | None = None,
    ridge: float = RIDGE,
    rank: int = RANK,
) -> np.ndarray:
    """Fit rank-truncated ridge bilinear residual z_c^T W x_i with optional sample weights."""
    n = len(r)
    if n == 0:
        raise ValueError("fit_weighted_bilinear requires at least one sample")
    feats = (z[:, :, None] * x[:, None, :]).reshape(n, -1).astype(np.float64)
    target = r.astype(np.float64)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != n or np.any(w < 0.0) or float(w.sum()) <= 0.0:
            raise ValueError("weights must be non-negative with positive sum")
        w_norm = w * (n / float(w.sum()))
        sqrt_w = np.sqrt(w_norm)[:, None]
        fw = feats * sqrt_w
        rw = target * sqrt_w[:, 0]
        a = fw.T @ fw + ridge * np.eye(feats.shape[1], dtype=np.float64)
        beta = np.linalg.solve(a, fw.T @ rw)
    else:
        a = feats.T @ feats + ridge * np.eye(feats.shape[1], dtype=np.float64)
        beta = np.linalg.solve(a, feats.T @ target)
    w_mat = beta.reshape(z.shape[1], x.shape[1])
    uu, s, vt = np.linalg.svd(w_mat, full_matrices=False)
    k = min(rank, len(s))
    return (uu[:, :k] * s[:k]) @ vt[:k]


def fit_empirical_bayes_group_mean(
    group_ids: list[Any],
    residuals: np.ndarray,
    weights: np.ndarray | None = None,
    tau: float = EB_TAU_EXACT,
) -> dict[Any, float]:
    """Compute empirical-Bayes shrinkage residual mean per group ID: (n / (n + tau)) * mean_r.

    Unseen group IDs are absent from the returned dict (callers add .get(gid, 0.0) to PRIOR,
    preserving feature-based PRIOR predictions rather than fabricating zero utility).
    """
    if len(group_ids) != len(residuals):
        raise ValueError("group_ids and residuals length mismatch")
    if tau < 0.0:
        raise ValueError("tau must be non-negative")
    w = (
        np.ones(len(residuals), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    sum_w: dict[Any, float] = defaultdict(float)
    sum_wr: dict[Any, float] = defaultdict(float)
    counts: dict[Any, int] = defaultdict(int)
    for gid, r_val, w_val in zip(group_ids, residuals, w):
        if w_val <= 0.0:
            continue
        sum_w[gid] += float(w_val)
        sum_wr[gid] += float(w_val) * float(r_val)
        counts[gid] += 1
    eb_map: dict[Any, float] = {}
    for gid, cnt in counts.items():
        sw = sum_w[gid]
        if sw > 0.0:
            mean_r = sum_wr[gid] / sw
            shrink = float(cnt) / (float(cnt) + float(tau))
            eb_map[gid] = shrink * mean_r
    return eb_map


def centre_within(groups: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Subtract within-context mean from values."""
    out = values.astype(np.float64).copy()
    if len(out) == 0:
        return out
    _, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv)
    sums = np.bincount(inv, weights=out)
    means = sums / np.maximum(counts, 1)
    return out - means[inv]


def ranked_unrevealed(
    score: np.ndarray, ids: list[Any], candidate_indices: list[int] | None = None
) -> list[int]:
    """Rank unrevealed slot indices (>= 4) by descending score with program-SHA tie-break."""
    cand = list(range(4, len(ids))) if candidate_indices is None else list(candidate_indices)
    if any(i < 4 or i >= len(ids) for i in cand):
        raise ValueError("candidate_indices must be valid unrevealed indices (>= 4)")
    return sorted(cand, key=lambda i: (-float(score[i]), ids[i]))


def hedge_picks(
    prior: np.ndarray,
    secondary: np.ndarray,
    ids: list[Any],
    candidate_indices: list[int] | None = None,
) -> list[int]:
    """Prior top-2 unrevealed picks + secondary top-2 among the remaining unrevealed slots."""
    r_prior = ranked_unrevealed(prior, ids, candidate_indices)
    if len(r_prior) < 4:
        raise ValueError("need at least 4 unrevealed slots for hedge_picks")
    keep = r_prior[:2]
    r_sec = ranked_unrevealed(secondary, ids, candidate_indices)
    rest = [i for i in r_sec if i not in keep][:2]
    return keep + rest


def eval_picks(
    u: np.ndarray, picks: list[int]
) -> tuple[float, float, float]:
    """Return (U0, U4, val_reg_inner) for a 4-element ordered pick list in unrevealed slots."""
    if len(picks) != 4 or len(set(picks)) != 4 or any(i < 4 or i >= len(u) for i in picks):
        raise ValueError("picks must be 4 distinct unrevealed indices (>= 4)")
    u0 = float(np.max(u[:4]))
    u4_val = float(max(u0, float(np.max(u[picks]))))
    val_reg_inner = float(u[picks[0]])
    return u0, u4_val, val_reg_inner


def eval_scorer(
    u: np.ndarray,
    score: np.ndarray,
    ids: list[Any],
    candidate_indices: list[int] | None = None,
) -> tuple[float, float, float, list[int]]:
    """Evaluate a scorer vector on unrevealed slots: returns (U0, U4, val_reg_inner, top4_picks)."""
    top4 = ranked_unrevealed(score, ids, candidate_indices)[:4]
    u0, u4_val, val_reg = eval_picks(u, top4)
    return u0, u4_val, val_reg, top4


def compute_contrast(
    values: np.ndarray, draws: np.ndarray, lcb_pct: float
) -> dict[str, Any]:
    """Compute family-level paired bootstrap contrast statistics and Common Advance Gate check."""
    boot_means = values[draws].mean(axis=1)
    boot_medians = np.median(values[draws], axis=1)
    lcb = float(np.percentile(boot_means, lcb_pct))
    med = float(np.median(values))
    harm = float(np.mean(values < -HARM_THRESHOLD))
    w = int(np.sum(values > TIE))
    t = int(np.sum(np.abs(values) <= TIE))
    loss = int(np.sum(values < -TIE))
    gate_pass = bool(lcb > GATE_LCB and med >= 0.0 and harm <= 0.10)
    return {
        "families": int(len(values)),
        "family_mean": float(values.mean()),
        "family_median": med,
        "median_ci95": [
            float(np.percentile(boot_medians, 2.5)),
            float(np.percentile(boot_medians, 97.5)),
        ],
        "families_losing_more_than_0_01_fraction": harm,
        "family_win_tie_loss": [w, t, loss],
        "exact_tie_fraction": float(t / len(values)) if len(values) > 0 else 0.0,
        "adjusted_one_sided_lcb": lcb,
        "lcb_percentile": float(lcb_pct),
        "bootstrap_p2_5": float(np.percentile(boot_means, 2.5)),
        "bootstrap_p5_one_sided_lcb": float(np.percentile(boot_means, 5.0)),
        "bootstrap_p97_5": float(np.percentile(boot_means, 97.5)),
        "common_gate_passed": gate_pass,
    }


def analyze_w6_panel(
    rows: list[dict[str, Any]],
    expected_contexts: set[str],
    expected_families: set[str],
    expected_arms: tuple[str, ...],
    expected_seeds: set[int],
    primary_contrasts: tuple[tuple[str, str], ...],
    secondary_contrasts: tuple[tuple[str, str], ...],
    anchor_arm: str = "PRIOR",
    elo_module: Any = None,
    shortfalls: int = 0,
    metric_key: str = "u4",
) -> dict[str, Any]:
    """Fail-closed W6 panel analyzer enforcing seed-set, roster, and incomplete-row guards."""
    seen_keys: set[tuple[str, int, str]] = set()
    duplicate_keys: list[tuple[str, int, str]] = []
    incomplete_rows = 0
    seeds_by_ctx_arm: dict[tuple[str, str], list[int]] = defaultdict(list)
    vals_by_ctx_arm: dict[tuple[str, str], list[float]] = defaultdict(list)
    fam_by_ctx: dict[str, str] = {}

    for r in rows:
        c = str(r["context"])
        f = str(r["family"])
        s = int(r["seed"])
        a = str(r["arm"])
        if a not in expected_arms:
            continue
        fam_by_ctx[c] = f
        key = (c, s, a)
        if key in seen_keys:
            duplicate_keys.append(key)
        seen_keys.add(key)
        val = r.get(metric_key)
        if r.get("status") != "completed" or val is None or not np.isfinite(float(val)):
            incomplete_rows += 1
        else:
            seeds_by_ctx_arm[(c, a)].append(s)
            vals_by_ctx_arm[(c, a)].append(float(val))

    missing_or_mismatched_seed_pairs = 0
    for c in expected_contexts:
        for a in expected_arms:
            observed_list = seeds_by_ctx_arm.get((c, a), [])
            if len(observed_list) != len(expected_seeds) or set(observed_list) != expected_seeds:
                missing_or_mismatched_seed_pairs += 1

    observed_contexts = {
        c
        for c in expected_contexts
        if all(
            len(seeds_by_ctx_arm.get((c, a), [])) == len(expected_seeds)
            and set(seeds_by_ctx_arm.get((c, a), [])) == expected_seeds
            for a in expected_arms
        )
    }
    observed_families = {fam_by_ctx[c] for c in observed_contexts if c in fam_by_ctx}

    complete = bool(
        incomplete_rows == 0
        and shortfalls == 0
        and len(duplicate_keys) == 0
        and missing_or_mismatched_seed_pairs == 0
        and observed_contexts == expected_contexts
        and observed_families == expected_families
    )

    completeness_report = {
        "complete": complete,
        "expected_contexts": len(expected_contexts),
        "observed_complete_contexts": len(observed_contexts),
        "expected_families": len(expected_families),
        "observed_complete_families": len(observed_families),
        "expected_seeds": sorted(expected_seeds),
        "incomplete_rows": incomplete_rows,
        "shortfalls": int(shortfalls),
        "duplicate_key_count": len(duplicate_keys),
        "missing_or_mismatched_seed_pairs": missing_or_mismatched_seed_pairs,
    }

    if not observed_families:
        return {
            "metric": metric_key,
            "completeness": completeness_report,
            "verdict": "INCOMPLETE",
            "primaries_passing": [],
        }

    ctx_means = {
        c: {a: float(mean(vals_by_ctx_arm[(c, a)])) for a in expected_arms}
        for c in sorted(observed_contexts)
    }
    families = sorted(observed_families)
    fv = {
        f: {
            a: float(mean(ctx_means[c][a] for c in ctx_means if fam_by_ctx[c] == f))
            for a in expected_arms
        }
        for f in families
    }
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(families), size=(BOOTSTRAP_DRAWS, len(families))
    )

    def eval_pair(a: str, b: str, pct: float) -> dict[str, Any]:
        vec = np.array([fv[f][a] - fv[f][b] for f in families], dtype=np.float64)
        res = compute_contrast(vec, draws, pct)
        if not complete:
            res["common_gate_passed"] = False
        return res

    prim_pct = 100.0 * 0.05 / max(1, len(primary_contrasts))
    primary_out = {
        f"{a} - {b}": eval_pair(a, b, prim_pct) for a, b in primary_contrasts
    }
    secondary_out = {
        f"{a} - {b}": eval_pair(a, b, 5.0) for a, b in secondary_contrasts
    }

    out: dict[str, Any] = {
        "metric": metric_key,
        "contexts": len(ctx_means),
        "families": len(families),
        "completeness": completeness_report,
        "family_mean": {
            a: float(mean(fv[f][a] for f in families)) for a in expected_arms
        },
        "context_mean": {
            a: float(mean(ctx_means[c][a] for c in ctx_means)) for a in expected_arms
        },
        "primary": primary_out,
        "secondary": secondary_out,
    }

    if elo_module is not None:
        recs = [
            elo_module.DatasetEvalRecord(c, fam_by_ctx[c], "regression", ctx_means[c])
            for c in sorted(ctx_means)
        ]
        point_elo = elo_module.compute_family_grouped_bradley_terry_elo(
            recs,
            anchor_contender=anchor_arm,
            anchor_elo=1500.0,
            tie_margin=TIE,
            prior=0.01,
            n_iters=400,
        )
        ratings_map = point_elo.get("ratings", point_elo) if isinstance(point_elo, dict) else point_elo
        out["family_elo"] = {k: float(v) for k, v in ratings_map.items()}

    if not complete:
        out["primaries_passing"] = []
        out["verdict"] = "INCOMPLETE"
    else:
        passed = [k for k, v in primary_out.items() if v["common_gate_passed"]]
        out["primaries_passing"] = passed
        out["verdict"] = (
            f"ADVANCE_CACHED_SCREEN ({', '.join(passed)})"
            if passed
            else "NO_ADVANCE (primary gate not met)"
        )
    return out


def load_elo() -> Any:
    if sha256_file(ELO_SOURCE) != PINS["elo_tracker"]:
        raise SystemExit("elo_tracker.py SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("h98_elo", ELO_SOURCE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h98_elo"] = mod
    spec.loader.exec_module(mod)
    return mod


def pack_compact_bundle(
    out_npz_path: Path,
    input_dir: Path,
    w1_dir: Path,
    h72_roster_path: Path,
    h72_fold_path: Path,
    h85_vectors_path: Path,
) -> dict[str, Any]:
    """Verify all input pins and pack a self-contained numeric .npz bundle for 4-node sharding."""
    pin_checks = {
        "selected": (input_dir / "selected.json", PINS["selected"]),
        "w1_fold_manifest": (w1_dir / "fold_manifest.json", PINS["w1_fold_manifest"]),
        "w1_priors": (w1_dir / "cache/w1_train_only_priors.npz", PINS["w1_priors"]),
        "w1_context_features": (
            w1_dir / "cache/train_only_context_features.npz",
            PINS["w1_context_features"],
        ),
        "champions_featurized": (
            Path.home() / "jope-data/h58/cache/champions_featurized.json",
            PINS["champions_featurized"],
        ),
        "h72_train_roster": (h72_roster_path, PINS["h72_train_roster"]),
        "h72_fold_manifest": (h72_fold_path, PINS["h72_fold_manifest"]),
        "h85_w1_vectors": (h85_vectors_path, PINS["h85_w1_vectors"]),
    }
    for name, (p, expected_sha) in pin_checks.items():
        actual = sha256_file(p)
        if actual != expected_sha:
            raise SystemExit(f"SHA-256 mismatch for {name}: {actual} != {expected_sha}")

    sel = json.loads((input_dir / "selected.json").read_text())["train"]
    if len(sel) != 1219 or any(r["dl_split"] != "DL-Train" for r in sel):
        raise SystemExit("DL-Train roster mismatch")
    cids = [r["context_sha256"] for r in sel]
    idx = {c: i for i, c in enumerate(cids)}
    fam = [r["family_id"] for r in sel]

    h72_roster = json.loads(h72_roster_path.read_text())
    phys_table_by_cid = {
        r["context_sha256"]: r["physical_table_sha256"] for r in h72_roster["rows"]
    }
    phys_tables = [phys_table_by_cid[c] for c in cids]

    h72_folds = json.loads(h72_fold_path.read_text())
    h72_fold_by_cid = {
        r["context_sha256"]: int(r["fold"]) for r in h72_folds["assignments"]
    }
    h72_fold_arr = np.array([h72_fold_by_cid[c] for c in cids], dtype=np.int32)

    w1_fm = json.loads((w1_dir / "fold_manifest.json").read_text())
    w1_fold_by_cid = {
        c: k for k, f in enumerate(w1_fm["folds"]) for c in f["held_contexts"]
    }
    w1_fold_arr = np.array([w1_fold_by_cid[c] for c in cids], dtype=np.int32)

    h85_vecs = json.loads(h85_vectors_path.read_text())
    r882_u4_map = h85_vecs["track2_verified_u4_by_cid"]["R882_CONTINUITY"]
    r882_leg_map = h85_vecs["track1_unrev_argmin_by_cid"]["R882_CONTINUITY"]
    r882_u4_arr = np.array([float(r882_u4_map[c]) for c in cids], dtype=np.float64)
    r882_leg_arr = np.array([float(r882_leg_map[c]) for c in cids], dtype=np.float64)

    z = np.load(w1_dir / "cache/w1_train_only_priors.npz", allow_pickle=False)
    offsets = z["offsets"].astype(np.int64)
    proj16 = z["proj16"].astype(np.float32)
    prior_by_fold = (-z["cal_cost_prog412_by_fold"]).astype(np.float64)
    ctx_by_fold = (-z["cal_cost_prog412_ctx_by_fold"]).astype(np.float64)

    cf = np.load(w1_dir / "cache/train_only_context_features.npz", allow_pickle=False)
    lm48_z = cf["lm48_z"].astype(np.float64)

    champs_raw = [
        r
        for r in json.loads(
            (Path.home() / "jope-data/h58/cache/champions_featurized.json").read_text()
        )
        if r["dl_split"] == "DL-Train" and r["context_sha256"] in idx
    ]
    champ_ctx_idx = np.array(
        [idx[r["context_sha256"]] for r in champs_raw], dtype=np.int32
    )
    champ_proj16 = np.array([r["proj16"] for r in champs_raw], dtype=np.float64)

    n_total = int(offsets[-1])
    u_all = np.empty(n_total, dtype=np.float64)
    gate_all = np.empty(n_total, dtype=bool)
    all_shas: list[str] = []
    all_tpls: list[str] = []
    fit_slot_indices = np.full((len(cids), FIT_SLOTS), -1, dtype=np.int32)
    fit_slot_counts = np.zeros(len(cids), dtype=np.int32)
    shuf_perm = np.zeros((5, len(cids)), dtype=np.int32)

    for k in range(5):
        fit_cids = [c for c in cids if w1_fold_by_cid[c] != k]
        perm = sorted(
            range(len(fit_cids)), key=lambda j: hkey("h83-shuffle", k, fit_cids[j])
        )
        for j, pj in enumerate(perm):
            shuf_perm[k, idx[fit_cids[j]]] = idx[fit_cids[pj]]

    for i, c in enumerate(cids):
        p = np.load(input_dir / "pools" / f"{c}.npz", allow_pickle=False)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        gate = p["gate"].astype(bool)
        u_vec = np.where(gate, -p["fitness"].astype(np.float64), -2.0)
        sha_list = [s.decode() for s in p["program_sha256"]]
        tpl_list = compute_template_hashes(p["program"])
        u_all[lo:hi] = u_vec
        gate_all[lo:hi] = gate
        all_shas.extend(sha_list)
        all_tpls.extend(tpl_list)
        sl = sorted(range(4, hi - lo), key=lambda s_i: hkey("h83-slot", c, sha_list[s_i]))[
            :FIT_SLOTS
        ]
        fit_slot_counts[i] = len(sl)
        fit_slot_indices[i, : len(sl)] = np.asarray(sl, dtype=np.int32)

    # Map unique SHAs and unique templates to lexicographically sorted int32 IDs
    # so integer ordering is 100% identical to string lexicographic tie-breaking
    uniq_shas = sorted(set(all_shas))
    sha_to_id = {s: idx_s for idx_s, s in enumerate(uniq_shas)}
    sha_id_all = np.array([sha_to_id[s] for s in all_shas], dtype=np.int32)

    uniq_tpls = sorted(set(all_tpls))
    tpl_to_id = {t: idx_t for idx_t, t in enumerate(uniq_tpls)}
    tpl_id_all = np.array([tpl_to_id[t] for t in all_tpls], dtype=np.int32)

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz_path,
        cids=np.array(cids, dtype="<U64"),
        fam=np.array(fam, dtype="<U80"),
        phys_tables=np.array(phys_tables, dtype="<U64"),
        w1_fold=w1_fold_arr,
        h72_fold=h72_fold_arr,
        r882_u4=r882_u4_arr,
        r882_leg=r882_leg_arr,
        offsets=offsets,
        proj16=proj16,
        prior_by_fold=prior_by_fold,
        ctx_by_fold=ctx_by_fold,
        lm48_z=lm48_z,
        champ_ctx_idx=champ_ctx_idx,
        champ_proj16=champ_proj16,
        u_all=u_all,
        gate_all=gate_all,
        sha_id_all=sha_id_all,
        tpl_id_all=tpl_id_all,
        fit_slot_indices=fit_slot_indices,
        fit_slot_counts=fit_slot_counts,
        shuf_perm=shuf_perm,
    )
    return {
        "bundle_path": str(out_npz_path),
        "bundle_sha256": sha256_file(out_npz_path),
        "contexts": len(cids),
        "total_slots": n_total,
        "unique_shas": len(uniq_shas),
        "unique_templates": len(uniq_tpls),
    }


def run_fold_shard_from_bundle(
    fold_k: int,
    bundle_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Execute outer fold k in {0..4} from the verified compact numeric bundle."""
    t0_wall = time.time()
    t0_cpu = time.process_time()

    b = np.load(bundle_path, allow_pickle=False)
    cids = [str(c) for c in b["cids"]]
    fam = [str(f) for f in b["fam"]]
    phys_tables = [str(t) for t in b["phys_tables"]]
    w1_fold = b["w1_fold"].astype(np.int32)
    h72_fold = b["h72_fold"].astype(np.int32)
    r882_u4 = b["r882_u4"].astype(np.float64)
    r882_leg = b["r882_leg"].astype(np.float64)
    offsets = b["offsets"].astype(np.int64)
    proj = b["proj16"].astype(np.float64)
    prior_by_fold = b["prior_by_fold"].astype(np.float64)
    ctx_by_fold = b["ctx_by_fold"].astype(np.float64)
    lm = b["lm48_z"].astype(np.float64)
    champ_ctx_idx = b["champ_ctx_idx"].astype(np.int32)
    champ_proj16 = b["champ_proj16"].astype(np.float64)
    u_all = b["u_all"].astype(np.float64)
    gate_all = b["gate_all"].astype(bool)
    sha_id_all = b["sha_id_all"].astype(np.int32)
    tpl_id_all = b["tpl_id_all"].astype(np.int32)
    fit_slot_indices = b["fit_slot_indices"].astype(np.int32)
    fit_slot_counts = b["fit_slot_counts"].astype(np.int32)
    shuf_perm = b["shuf_perm"].astype(np.int32)

    fit_i = [i for i in range(len(cids)) if int(w1_fold[i]) != fold_k]
    held_i = [i for i in range(len(cids)) if int(w1_fold[i]) == fold_k]
    held_families = {fam[i] for i in held_i}
    held_tables = {phys_tables[i] for i in held_i}
    if any(fam[i] in held_families or phys_tables[i] in held_tables for i in fit_i):
        raise SystemExit(f"Fold {fold_k} donor leakage detected")

    fit_fam_counts: dict[str, int] = defaultdict(int)
    for i in fit_i:
        fit_fam_counts[fam[i]] += 1

    rows_z, rows_x, rows_r, rows_g, rows_w = [], [], [], [], []
    nc_z, nc_x, nc_r, nc_g, nc_w = [], [], [], [], []
    eb_prog_ids: list[int] = []
    eb_tpl_ids: list[int] = []
    eb_resids: list[float] = []
    eb_weights: list[float] = []
    fit_seen_templates: set[int] = set()

    for i in fit_i:
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        n_slots = hi - lo
        fam_w = 1.0 / float(fit_fam_counts[fam[i]])
        u_c = u_all[lo:hi]
        gate_c = gate_all[lo:hi]
        sha_c = sha_id_all[lo:hi]
        tpl_c = tpl_id_all[lo:hi]
        prior_c = prior_by_fold[fold_k, lo:hi]

        fit_seen_templates.update(int(t) for t in tpl_c)

        r_all = (u_c[4:] - float(u_c[4:].mean())) - (
            prior_c[4:] - float(prior_c[4:].mean())
        )
        eb_prog_ids.extend(int(s) for s in sha_c[4:])
        eb_tpl_ids.extend(int(t) for t in tpl_c[4:])
        eb_resids.extend(r_all.tolist())
        eb_weights.extend([fam_w] * len(r_all))

        cnt = int(fit_slot_counts[i])
        sl = fit_slot_indices[i, :cnt]
        g = lo + sl
        rows_x.append(proj[g])
        rows_r.append(u_c[sl] - prior_c[sl])
        rows_z.append(np.repeat(lm[i][None], len(sl), 0))
        rows_g.append(np.full(len(sl), i, dtype=np.int64))
        rows_w.append(np.full(len(sl), fam_w / float(len(sl)), dtype=np.float64))

        passing_unrev = [idx_s for idx_s in range(4, n_slots) if gate_c[idx_s]]
        if len(passing_unrev) >= 4:
            top_pass = sorted(
                passing_unrev, key=lambda idx_s: (-float(u_c[idx_s]), int(sha_c[idx_s]))
            )[:NEAR_CHAMP_SLOTS]
            top_arr = np.asarray(top_pass, dtype=np.int64)
            g_nc = lo + top_arr
            nc_x.append(proj[g_nc])
            nc_r.append(u_c[top_arr] - prior_c[top_arr])
            nc_z.append(np.repeat(lm[i][None], len(top_arr), 0))
            nc_g.append(np.full(len(top_arr), i, dtype=np.int64))
            nc_w.append(
                np.full(len(top_arr), fam_w / float(len(top_arr)), dtype=np.float64)
            )

    X = np.concatenate(rows_x)
    R = np.concatenate(rows_r)
    G = np.concatenate(rows_g)
    Z = np.concatenate(rows_z)
    W = np.concatenate(rows_w)
    zm, zs = Z.mean(0), Z.std(0) + 1e-9
    xm, xs = X.mean(0), X.std(0) + 1e-9
    Zs = (Z - zm) / zs
    Xs = (X - xm) / xs
    Rc = centre_within(G, R)

    w_treat = fit_weighted_bilinear(Zs, Xs, Rc, weights=None, ridge=RIDGE, rank=RANK)
    w_treat_fambal = fit_weighted_bilinear(
        Zs, Xs, Rc, weights=W, ridge=RIDGE, rank=RANK
    )
    Zsh = (np.stack([lm[int(shuf_perm[fold_k, int(g)])] for g in G]) - zm) / zs
    w_shuf = fit_weighted_bilinear(Zsh, Xs, Rc, weights=None, ridge=RIDGE, rank=RANK)

    X_nc = np.concatenate(nc_x)
    R_nc = np.concatenate(nc_r)
    G_nc = np.concatenate(nc_g)
    Z_nc = np.concatenate(nc_z)
    W_nc = np.concatenate(nc_w)
    zm_nc, zs_nc = Z_nc.mean(0), Z_nc.std(0) + 1e-9
    xm_nc, xs_nc = X_nc.mean(0), X_nc.std(0) + 1e-9
    Zs_nc = (Z_nc - zm_nc) / zs_nc
    Xs_nc = (X_nc - xm_nc) / xs_nc
    Rc_nc = centre_within(G_nc, R_nc)
    w_near_champ = fit_weighted_bilinear(
        Zs_nc, Xs_nc, Rc_nc, weights=W_nc, ridge=RIDGE, rank=RANK
    )

    eb_exact_map = fit_empirical_bayes_group_mean(
        eb_prog_ids,
        np.asarray(eb_resids, dtype=np.float64),
        weights=np.asarray(eb_weights, dtype=np.float64),
        tau=EB_TAU_EXACT,
    )
    eb_tpl_map = fit_empirical_bayes_group_mean(
        eb_tpl_ids,
        np.asarray(eb_resids, dtype=np.float64),
        weights=np.asarray(eb_weights, dtype=np.float64),
        tau=EB_TAU_TEMPLATE,
    )

    # H72-fold exact-ID coverage for H72 fold == fold_k
    h72_fit_exact_all: set[int] = set()
    for i in range(len(cids)):
        if int(h72_fold[i]) != fold_k:
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            h72_fit_exact_all.update(int(s) for s in sha_id_all[lo:hi])
    h72_held_i = [i for i in range(len(cids)) if int(h72_fold[i]) == fold_k]
    h72_total_slots = sum(int(offsets[i + 1] - offsets[i]) for i in h72_held_i)
    h72_covered_slots = sum(
        sum(
            1
            for s in sha_id_all[int(offsets[i]) : int(offsets[i + 1])]
            if int(s) in h72_fit_exact_all
        )
        for i in h72_held_i
    )

    fit_lm12 = np.stack([lm[i][:12] for i in fit_i])
    champ_by_ctx: dict[int, list[np.ndarray]] = defaultdict(list)
    for c_i, c_p in zip(champ_ctx_idx, champ_proj16):
        if int(w1_fold[int(c_i)]) != fold_k:
            champ_by_ctx[int(c_i)].append(c_p)

    per_ctx_u4: dict[str, dict[str, float]] = {}
    per_ctx_leg: dict[str, dict[str, float]] = {}
    per_ctx_seen_tpl_u4: dict[str, dict[str, float]] = {}
    per_ctx_unseen_tpl_u4: dict[str, dict[str, float]] = {}
    per_ctx_seen_tpl_leg: dict[str, dict[str, float]] = {}
    per_ctx_unseen_tpl_leg: dict[str, dict[str, float]] = {}

    w1_unrev_slots = 0
    w1_unrev_seen_exact = 0
    w1_unrev_seen_tpl = 0
    exact_id_top4_changed_contexts = 0

    for i in held_i:
        c = cids[i]
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        u = u_all[lo:hi]
        ids = [int(s) for s in sha_id_all[lo:hi]]
        tpls = [int(t) for t in tpl_id_all[lo:hi]]

        xs_c = (proj[lo:hi] - xm) / xs
        zs_c = (lm[i] - zm) / zs
        xs_nc_c = (proj[lo:hi] - xm_nc) / xs_nc
        zs_nc_c = (lm[i] - zm_nc) / zs_nc

        prior = prior_by_fold[fold_k, lo:hi]
        ctx_sc = ctx_by_fold[fold_k, lo:hi]
        treat_sc = prior + xs_c @ (w_treat.T @ zs_c)
        treat_fb_sc = prior + xs_c @ (w_treat_fambal.T @ zs_c)
        shuf_sc = prior + xs_c @ (w_shuf.T @ zs_c)
        nc_sc = prior + xs_nc_c @ (w_near_champ.T @ zs_nc_c)

        exact_adj = np.array(
            [eb_exact_map.get(s, 0.0) for s in ids], dtype=np.float64
        )
        tpl_adj = np.array([eb_tpl_map.get(t, 0.0) for t in tpls], dtype=np.float64)
        fit_prog_sc = prior + exact_adj
        fit_tpl_sc = prior + tpl_adj

        near = np.argsort(
            np.sum((fit_lm12 - lm[i][:12]) ** 2, axis=1), kind="stable"
        )[:RETR_CONTEXTS]
        cp = [v for j in near for v in champ_by_ctx[fit_i[int(j)]]]
        retr_sc = (
            -np.min(
                np.linalg.norm(
                    proj[lo:hi][:, None, :] - np.stack(cp)[None], axis=2
                ),
                axis=1,
            )
            if cp
            else np.zeros(hi - lo, dtype=np.float64)
        )

        scorer_map = {
            "RETR": retr_sc,
            "PRIOR": prior,
            "FIT_PROG_MEAN": fit_prog_sc,
            "FIT_TEMPLATE_MEAN": fit_tpl_sc,
            "TREAT": treat_sc,
            "TREAT_FAMBAL": treat_fb_sc,
            "SHUF": shuf_sc,
            "NEAR_CHAMP_TREAT": nc_sc,
            "CTX": ctx_sc,
        }
        hedge_map = {
            "HEDGE_TREAT_FAMBAL": treat_fb_sc,
            "HEDGE_NEAR_CHAMP": nc_sc,
            "HEDGE_CTX": ctx_sc,
        }

        u4_row: dict[str, float] = {"R882_CONTINUITY": float(r882_u4[i])}
        leg_row: dict[str, float] = {"R882_CONTINUITY": float(r882_leg[i])}

        prior_top4: list[int] = []
        fit_prog_top4: list[int] = []
        for a, sc in scorer_map.items():
            u0, u4_v, leg_v, top4 = eval_scorer(u, sc, ids)
            u4_row[a] = u4_v
            leg_row[a] = leg_v
            u4_row["U0"] = u0
            if a == "PRIOR":
                prior_top4 = top4
            elif a == "FIT_PROG_MEAN":
                fit_prog_top4 = top4

        if set(prior_top4) != set(fit_prog_top4):
            exact_id_top4_changed_contexts += 1

        for a, sec_sc in hedge_map.items():
            hpicks = hedge_picks(prior, sec_sc, ids)
            _, u4_v, leg_v = eval_picks(u, hpicks)
            u4_row[a] = u4_v
            leg_row[a] = leg_v

        per_ctx_u4[c] = u4_row
        per_ctx_leg[c] = leg_row

        seen_idx = [idx_s for idx_s in range(4, len(ids)) if tpls[idx_s] in fit_seen_templates]
        unseen_idx = [
            idx_s for idx_s in range(4, len(ids)) if tpls[idx_s] not in fit_seen_templates
        ]
        if len(seen_idx) < 4 or len(unseen_idx) < 4:
            raise SystemExit(f"Context {c} has < 4 seen or unseen template slots")

        s_u4: dict[str, float] = {}
        u_u4: dict[str, float] = {}
        s_leg: dict[str, float] = {}
        u_leg: dict[str, float] = {}
        for a, sc in scorer_map.items():
            _, s_u4[a], s_leg[a], _ = eval_scorer(u, sc, ids, seen_idx)
            _, u_u4[a], u_leg[a], _ = eval_scorer(u, sc, ids, unseen_idx)
        for a, sec_sc in hedge_map.items():
            _, s_u4[a], s_leg[a] = eval_picks(
                u, hedge_picks(prior, sec_sc, ids, seen_idx)
            )
            _, u_u4[a], u_leg[a] = eval_picks(
                u, hedge_picks(prior, sec_sc, ids, unseen_idx)
            )
        per_ctx_seen_tpl_u4[c] = s_u4
        per_ctx_unseen_tpl_u4[c] = u_u4
        per_ctx_seen_tpl_leg[c] = s_leg
        per_ctx_unseen_tpl_leg[c] = u_leg

        w1_unrev_slots += len(ids) - 4
        w1_unrev_seen_exact += sum(1 for s in ids[4:] if s in eb_exact_map)
        w1_unrev_seen_tpl += len(seen_idx)

    shard_out = {
        "fold": fold_k,
        "fit_contexts": len(fit_i),
        "held_contexts": len(held_i),
        "held_families": len(held_families),
        "w_treat_fro": float(np.linalg.norm(w_treat)),
        "w_treat_fambal_fro": float(np.linalg.norm(w_treat_fambal)),
        "w_shuf_fro": float(np.linalg.norm(w_shuf)),
        "w_near_champ_fro": float(np.linalg.norm(w_near_champ)),
        "eb_exact_unique_programs": len(eb_exact_map),
        "eb_template_unique_templates": len(eb_tpl_map),
        "w1_fold_unrevealed_slots": w1_unrev_slots,
        "w1_fold_unrevealed_seen_exact_slots": w1_unrev_seen_exact,
        "w1_fold_unrevealed_seen_template_slots": w1_unrev_seen_tpl,
        "exact_id_top4_changed_contexts": exact_id_top4_changed_contexts,
        "h72_fold_held_contexts": len(h72_held_i),
        "h72_fold_total_slots": h72_total_slots,
        "h72_fold_seen_exact_slots": h72_covered_slots,
        "wall_s": time.time() - t0_wall,
        "cpu_s": time.process_time() - t0_cpu,
        "per_ctx_u4": per_ctx_u4,
        "per_ctx_leg": per_ctx_leg,
        "per_ctx_seen_tpl_u4": per_ctx_seen_tpl_u4,
        "per_ctx_unseen_tpl_u4": per_ctx_unseen_tpl_u4,
        "per_ctx_seen_tpl_leg": per_ctx_seen_tpl_leg,
        "per_ctx_unseen_tpl_leg": per_ctx_unseen_tpl_leg,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(shard_out, indent=2, sort_keys=True))
    return shard_out


def evaluate_prospective_h94_with_true_w1(
    input_dir: Path,
    w1_dir: Path,
    h85_models_path: Path,
    h94_fits_gz_path: Path,
    h94_rows_gz_path: Path,
    h95_confirmed_hwm_path: Path,
) -> dict[str, Any]:
    """Re-score H94's real-S3 prospective proposals with persisted W1 outer models & audit H95 HWMs."""
    if sha256_file(h85_models_path) != PINS["h85_w1_outer_models"]:
        raise SystemExit("h85_models_path SHA-256 mismatch")
    if sha256_file(h94_fits_gz_path) != PINS["h94_fits_gz"]:
        raise SystemExit("h94_fits_gz_path SHA-256 mismatch")
    if sha256_file(h94_rows_gz_path) != PINS["h94_rows_gz"]:
        raise SystemExit("h94_rows_gz_path SHA-256 mismatch")
    if sha256_file(h95_confirmed_hwm_path) != PINS["h95_confirmed_hwm"]:
        raise SystemExit("h95_confirmed_hwm_path SHA-256 mismatch")

    with open(h85_models_path, "rb") as f:
        w1_bundle = pickle.load(f)
    outer_models = w1_bundle["outer_models"]

    fits_map: dict[tuple[str, str], dict[str, Any]] = {}
    with gzip.open(h94_fits_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                ctx_id = rec.get("context") or rec["context_sha256"]
                fits_map[(ctx_id, rec["program_sha256"])] = rec

    h94_rows: list[dict[str, Any]] = []
    with gzip.open(h94_rows_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                h94_rows.append(json.loads(line))

    h95_hwms = json.loads(h95_confirmed_hwm_path.read_text())
    confirmed_set = {
        (r["context_sha256"], r["program_sha256"]) for r in h95_hwms
    }
    material_set = {
        (r["context_sha256"], r["program_sha256"])
        for r in h95_hwms
        if r.get("material")
    }

    by_cs: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in h94_rows:
        by_cs[(r["context"], int(r["seed"]))][r["arm"]] = r

    micro_py = Path.home() / "JopeDimeMicro/python"
    if str(micro_py) not in sys.path:
        sys.path.insert(0, str(micro_py))
    from rust_evaluator import featurize_dsl_412, propose_flywheel_candidates

    sel = json.loads((input_dir / "selected.json").read_text())["train"]
    idx = {r["context_sha256"]: i for i, r in enumerate(sel)}
    fam = {r["context_sha256"]: r["family_id"] for r in sel}
    w1_fm = json.loads((w1_dir / "fold_manifest.json").read_text())
    fold_by_cid = {
        c: k for k, fl in enumerate(w1_fm["folds"]) for c in fl["held_contexts"]
    }
    cf = np.load(w1_dir / "cache/train_only_context_features.npz", allow_pickle=False)
    lm = cf["lm48_z"].astype(np.float64)
    dope = cf["dope629"].astype(np.float32)
    ctxf = np.concatenate([lm.astype(np.float32), dope], axis=1)

    champ_by_cid: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in json.loads((input_dir / "champions.json").read_text()):
        champ_by_cid[r["context_sha256"]].append(
            {"dsl": r["dsl"], "program_sha256": r["program_sha256"]}
        )
    bank = [
        r
        for r in json.loads(
            (Path.home() / "jope-data/h58/cache/champions_featurized.json").read_text()
        )
        if r["dl_split"] == "DL-Train" and r["context_sha256"] in idx
    ]

    def hseed(*parts: Any) -> int:
        return int(
            hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:15],
            16,
        )

    def ranked_idx(sc: np.ndarray, ids: list[str], cand: list[int] | None = None) -> list[int]:
        pool_idx = list(range(len(ids))) if cand is None else cand
        return sorted(pool_idx, key=lambda i: (-float(sc[i]), ids[i]))

    top4_overlap_prior: list[float] = []
    top4_overlap_ctx: list[float] = []
    top4_overlap_hedge: list[float] = []
    true_w1_fully_evaluated_cs = 0

    hwm_recall_confirmed: dict[str, set[tuple[str, str]]] = defaultdict(set)
    hwm_recall_material: dict[str, set[tuple[str, str]]] = defaultdict(set)
    evaluated_subset_rows: list[dict[str, Any]] = []

    for (c, seed), arm_rows in sorted(by_cs.items()):
        p = np.load(input_dir / "pools" / f"{c}.npz", allow_pickle=False)
        known = {s.decode() for s in p["program_sha256"]}
        gate_p = p["gate"].astype(bool)
        u_pool = np.where(gate_p, -p["fitness"].astype(np.float64), -2.0)
        anchors_best = float(np.max(u_pool[:4]))

        props, seen, batch = [], set(known), 0
        while len(props) < 16 and batch < 50:
            for q in propose_flywheel_candidates(
                champ_by_cid[c],
                lm[idx[c]],
                bank,
                fam[c],
                n_proposals=16,
                seed=hseed("h94-stream", c, seed, batch) % (2**31),
            ):
                if q["program_sha256"] not in seen and len(props) < 16:
                    seen.add(q["program_sha256"])
                    props.append(q)
            batch += 1

        ids = [q["program_sha256"] for q in props]
        for a_name, r_row in arm_rows.items():
            for rev_id in r_row["revealed"]:
                if rev_id not in ids:
                    raise SystemExit(
                        f"Proposal stream mismatch on ({c}, {seed}, {a_name}): {rev_id} not in regenerated ids"
                    )

        feats = np.stack(
            [np.asarray(featurize_dsl_412(q["dsl"])[0], dtype=np.float32) for q in props]
        )
        k = fold_by_cid[c]
        m_entry = outer_models[f"fold_{k}"]
        raw_prog = m_entry["m_prog"].predict(feats).astype(np.float64)
        mu_prog_true = (
            float(m_entry["a_prog"]) + float(m_entry["b_prog"]) * raw_prog
        )

        x_ctx = np.concatenate(
            [feats, np.repeat(ctxf[idx[c]][None], len(props), 0)], axis=1
        )
        raw_ctx = m_entry["m_ctx"].predict(x_ctx).astype(np.float64)
        mu_ctx_true = (
            float(m_entry["a_ctx"]) + float(m_entry["b_ctx"]) * raw_ctx
        )

        true_prior_top4 = [ids[i] for i in ranked_idx(mu_prog_true, ids)[:4]]
        true_ctx_top4 = [ids[i] for i in ranked_idx(mu_ctx_true, ids)[:4]]
        keep2 = ranked_idx(mu_prog_true, ids)[:2]
        true_hedge_top4 = [
            ids[i]
            for i in (keep2 + [j for j in ranked_idx(mu_ctx_true, ids) if j not in keep2][:2])
        ]

        h94_prior_top4 = arm_rows["PRIOR_top4"]["revealed"]
        h94_ctx_top4 = arm_rows["CTX_top4"]["revealed"]
        h94_hedge_top4 = arm_rows["HEDGE_CTX"]["revealed"]

        top4_overlap_prior.append(
            len(set(true_prior_top4) & set(h94_prior_top4)) / 4.0
        )
        top4_overlap_ctx.append(len(set(true_ctx_top4) & set(h94_ctx_top4)) / 4.0)
        top4_overlap_hedge.append(
            len(set(true_hedge_top4) & set(h94_hedge_top4)) / 4.0
        )

        if all((c, pid) in fits_map for pid in true_prior_top4 + true_ctx_top4 + true_hedge_top4):
            true_w1_fully_evaluated_cs += 1

        for arm_label, picked_ids in [
            ("H94_PRIOR_top4", h94_prior_top4),
            ("H94_CTX_top4", h94_ctx_top4),
            ("H94_HEDGE_CTX", h94_hedge_top4),
            ("TRUE_W1_PRIOR_top4", true_prior_top4),
            ("TRUE_W1_CTX_top4", true_ctx_top4),
            ("TRUE_W1_HEDGE_CTX", true_hedge_top4),
        ]:
            for pid in picked_ids:
                if (c, pid) in confirmed_set:
                    hwm_recall_confirmed[arm_label].add((c, pid))
                if (c, pid) in material_set:
                    hwm_recall_material[arm_label].add((c, pid))

        eval_indices = [i for i, pid in enumerate(ids) if (c, pid) in fits_map]
        if len(eval_indices) >= 4:
            def best_of_4(pick_idx: list[int]) -> float:
                vals = []
                for idx_i in pick_idx:
                    f_rec = fits_map[(c, ids[idx_i])]
                    u_v = (
                        -float(f_rec["fitness"])
                        if (f_rec["status"] == "completed" and f_rec["gate"])
                        else -2.0
                    )
                    vals.append(u_v)
                return float(max([anchors_best] + vals))

            p_true_idx = ranked_idx(mu_prog_true, ids, eval_indices)[:4]
            c_true_idx = ranked_idx(mu_ctx_true, ids, eval_indices)[:4]
            k2_eval = ranked_idx(mu_prog_true, ids, eval_indices)[:2]
            h_true_idx = k2_eval + [
                j for j in ranked_idx(mu_ctx_true, ids, eval_indices) if j not in k2_eval
            ][:2]

            for a_lbl, val_u4 in [
                ("H94_PRIOR_top4", float(arm_rows["PRIOR_top4"]["retained_best"])),
                ("H94_CTX_top4", float(arm_rows["CTX_top4"]["retained_best"])),
                ("H94_HEDGE_CTX", float(arm_rows["HEDGE_CTX"]["retained_best"])),
                ("TRUE_W1_PRIOR_EVAL_SUBSET", best_of_4(p_true_idx)),
                ("TRUE_W1_CTX_EVAL_SUBSET", best_of_4(c_true_idx)),
                ("TRUE_W1_HEDGE_CTX_EVAL_SUBSET", best_of_4(h_true_idx)),
            ]:
                evaluated_subset_rows.append(
                    {
                        "context": c,
                        "family": fam[c],
                        "seed": seed,
                        "arm": a_lbl,
                        "status": "completed",
                        "u4": val_u4,
                    }
                )

    prospective_arms = (
        "H94_PRIOR_top4",
        "H94_CTX_top4",
        "H94_HEDGE_CTX",
        "TRUE_W1_PRIOR_EVAL_SUBSET",
        "TRUE_W1_CTX_EVAL_SUBSET",
        "TRUE_W1_HEDGE_CTX_EVAL_SUBSET",
    )
    exp_ctxs = {r["context"] for r in h94_rows}
    exp_fams = {r["family"] for r in h94_rows}
    prospective_panel = analyze_w6_panel(
        rows=evaluated_subset_rows,
        expected_contexts=exp_ctxs,
        expected_families=exp_fams,
        expected_arms=prospective_arms,
        expected_seeds={701, 702, 703},
        primary_contrasts=(
            ("H94_HEDGE_CTX", "H94_PRIOR_top4"),
            ("TRUE_W1_HEDGE_CTX_EVAL_SUBSET", "TRUE_W1_PRIOR_EVAL_SUBSET"),
        ),
        secondary_contrasts=(
            ("H94_CTX_top4", "H94_PRIOR_top4"),
            ("TRUE_W1_CTX_EVAL_SUBSET", "TRUE_W1_PRIOR_EVAL_SUBSET"),
            ("TRUE_W1_PRIOR_EVAL_SUBSET", "H94_PRIOR_top4"),
            ("TRUE_W1_HEDGE_CTX_EVAL_SUBSET", "H94_HEDGE_CTX"),
        ),
        anchor_arm="H94_PRIOR_top4",
        elo_module=load_elo(),
        shortfalls=0,
        metric_key="u4",
    )

    return {
        "context_seeds_checked": len(by_cs),
        "natively_evaluated_unique_programs": len(fits_map),
        "h95_confirmed_hwms_total": len(confirmed_set),
        "h95_material_hwms_total": len(material_set),
        "top4_set_overlap_true_w1_vs_h94_refit": {
            "PRIOR_top4_mean_jaccard4": float(mean(top4_overlap_prior)),
            "CTX_top4_mean_jaccard4": float(mean(top4_overlap_ctx)),
            "HEDGE_CTX_mean_jaccard4": float(mean(top4_overlap_hedge)),
            "context_seeds_fully_covered_by_h94_fits": true_w1_fully_evaluated_cs,
        },
        "h95_hwm_recall_by_arm": {
            arm_k: {
                "confirmed_30_recalled": len(hwm_recall_confirmed[arm_k]),
                "confirmed_30_recall_fraction": float(
                    len(hwm_recall_confirmed[arm_k]) / max(1, len(confirmed_set))
                ),
                "material_27_recalled": len(hwm_recall_material[arm_k]),
                "material_27_recall_fraction": float(
                    len(hwm_recall_material[arm_k]) / max(1, len(material_set))
                ),
            }
            for arm_k in (
                "H94_PRIOR_top4",
                "H94_CTX_top4",
                "H94_HEDGE_CTX",
                "TRUE_W1_PRIOR_top4",
                "TRUE_W1_CTX_top4",
                "TRUE_W1_HEDGE_CTX",
            )
        },
        "evaluated_subset_panel": prospective_panel,
    }


def merge_and_analyze_shards(
    shard_paths: list[Path],
    input_dir: Path,
    out_dir: Path,
    run_id: str,
    prospective_audit: dict[str, Any] | None = None,
    host_receipts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Merge all 5 fold shards, run fail-closed analysis across all tracks/strata, and write metrics."""
    sel = json.loads((input_dir / "selected.json").read_text())["train"]
    expected_contexts = {r["context_sha256"] for r in sel}
    expected_families = {r["family_id"] for r in sel}
    fam = {r["context_sha256"]: r["family_id"] for r in sel}

    shards = [json.loads(p.read_text()) for p in shard_paths]
    if sorted(s["fold"] for s in shards) != [0, 1, 2, 3, 4]:
        raise SystemExit("Shard folds must be exactly [0, 1, 2, 3, 4]")

    rows_u4: list[dict[str, Any]] = []
    rows_leg: list[dict[str, Any]] = []
    rows_seen_tpl_u4: list[dict[str, Any]] = []
    rows_unseen_tpl_u4: list[dict[str, Any]] = []
    rows_seen_tpl_leg: list[dict[str, Any]] = []
    rows_unseen_tpl_leg: list[dict[str, Any]] = []

    per_context_u4: dict[str, dict[str, float]] = {}
    per_context_leg: dict[str, dict[str, float]] = {}
    u0_by_ctx: dict[str, float] = {}

    tpl_arms = tuple(a for a in ARMS if a != "R882_CONTINUITY")

    for s in shards:
        for c, u4_map in s["per_ctx_u4"].items():
            per_context_u4[c] = u4_map
            u0_by_ctx[c] = float(u4_map["U0"])
            for a in ARMS:
                rows_u4.append(
                    {
                        "context": c,
                        "family": fam[c],
                        "seed": 0,
                        "arm": a,
                        "status": "completed",
                        "u4": float(u4_map[a]),
                    }
                )
        for c, leg_map in s["per_ctx_leg"].items():
            per_context_leg[c] = leg_map
            for a in ARMS:
                rows_leg.append(
                    {
                        "context": c,
                        "family": fam[c],
                        "seed": 0,
                        "arm": a,
                        "status": "completed",
                        "val_reg_inner": float(leg_map[a]),
                    }
                )
        for c, su4 in s["per_ctx_seen_tpl_u4"].items():
            for a in tpl_arms:
                rows_seen_tpl_u4.append(
                    {
                        "context": c,
                        "family": fam[c],
                        "seed": 0,
                        "arm": a,
                        "status": "completed",
                        "u4": float(su4[a]),
                    }
                )
        for c, uu4 in s["per_ctx_unseen_tpl_u4"].items():
            for a in tpl_arms:
                rows_unseen_tpl_u4.append(
                    {
                        "context": c,
                        "family": fam[c],
                        "seed": 0,
                        "arm": a,
                        "status": "completed",
                        "u4": float(uu4[a]),
                    }
                )
        for c, sleg in s["per_ctx_seen_tpl_leg"].items():
            for a in tpl_arms:
                rows_seen_tpl_leg.append(
                    {
                        "context": c,
                        "family": fam[c],
                        "seed": 0,
                        "arm": a,
                        "status": "completed",
                        "val_reg_inner": float(sleg[a]),
                    }
                )
        for c, uleg in s["per_ctx_unseen_tpl_leg"].items():
            for a in tpl_arms:
                rows_unseen_tpl_leg.append(
                    {
                        "context": c,
                        "family": fam[c],
                        "seed": 0,
                        "arm": a,
                        "status": "completed",
                        "val_reg_inner": float(uleg[a]),
                    }
                )

    elo_mod = load_elo()

    track2_u4_panel = analyze_w6_panel(
        rows=rows_u4,
        expected_contexts=expected_contexts,
        expected_families=expected_families,
        expected_arms=ARMS,
        expected_seeds={0},
        primary_contrasts=PRIMARY_CONTRASTS,
        secondary_contrasts=SECONDARY_CONTRASTS,
        anchor_arm="PRIOR",
        elo_module=elo_mod,
        shortfalls=0,
        metric_key="u4",
    )
    recs_u4 = [
        elo_mod.DatasetEvalRecord(
            c, fam[c], "regression", {a: per_context_u4[c][a] for a in ARMS}
        )
        for c in sorted(per_context_u4)
    ]
    r882_u4_elo = elo_mod.compute_family_grouped_bradley_terry_elo(
        recs_u4,
        anchor_contender="R882_CONTINUITY",
        anchor_elo=1500.0,
        tie_margin=TIE,
        prior=0.01,
        n_iters=400,
    )
    track2_u4_panel["family_elo_r882_anchor1500"] = {
        k: float(v)
        for k, v in r882_u4_elo.get("ratings", r882_u4_elo).items()
    }

    track1_leg_panel = analyze_w6_panel(
        rows=rows_leg,
        expected_contexts=expected_contexts,
        expected_families=expected_families,
        expected_arms=ARMS,
        expected_seeds={0},
        primary_contrasts=PRIMARY_CONTRASTS,
        secondary_contrasts=SECONDARY_CONTRASTS,
        anchor_arm="PRIOR",
        elo_module=elo_mod,
        shortfalls=0,
        metric_key="val_reg_inner",
    )
    recs_leg = [
        elo_mod.DatasetEvalRecord(
            c, fam[c], "regression", {a: per_context_leg[c][a] for a in ARMS}
        )
        for c in sorted(per_context_leg)
    ]
    r882_leg_elo = elo_mod.compute_family_grouped_bradley_terry_elo(
        recs_leg,
        anchor_contender="R882_CONTINUITY",
        anchor_elo=1500.0,
        tie_margin=TIE,
        prior=0.01,
        n_iters=400,
    )
    track1_leg_panel["family_elo_r882_anchor1500"] = {
        k: float(v)
        for k, v in r882_leg_elo.get("ratings", r882_leg_elo).items()
    }

    tpl_sec = tuple(
        (a, b)
        for a, b in SECONDARY_CONTRASTS
        if a != "R882_CONTINUITY" and b != "R882_CONTINUITY"
    )
    seen_tpl_u4_panel = analyze_w6_panel(
        rows=rows_seen_tpl_u4,
        expected_contexts=expected_contexts,
        expected_families=expected_families,
        expected_arms=tpl_arms,
        expected_seeds={0},
        primary_contrasts=PRIMARY_CONTRASTS,
        secondary_contrasts=tpl_sec,
        anchor_arm="PRIOR",
        elo_module=elo_mod,
        shortfalls=0,
        metric_key="u4",
    )
    unseen_tpl_u4_panel = analyze_w6_panel(
        rows=rows_unseen_tpl_u4,
        expected_contexts=expected_contexts,
        expected_families=expected_families,
        expected_arms=tpl_arms,
        expected_seeds={0},
        primary_contrasts=PRIMARY_CONTRASTS,
        secondary_contrasts=tpl_sec,
        anchor_arm="PRIOR",
        elo_module=elo_mod,
        shortfalls=0,
        metric_key="u4",
    )
    seen_tpl_leg_panel = analyze_w6_panel(
        rows=rows_seen_tpl_leg,
        expected_contexts=expected_contexts,
        expected_families=expected_families,
        expected_arms=tpl_arms,
        expected_seeds={0},
        primary_contrasts=PRIMARY_CONTRASTS,
        secondary_contrasts=tpl_sec,
        anchor_arm="PRIOR",
        elo_module=elo_mod,
        shortfalls=0,
        metric_key="val_reg_inner",
    )
    unseen_tpl_leg_panel = analyze_w6_panel(
        rows=rows_unseen_tpl_leg,
        expected_contexts=expected_contexts,
        expected_families=expected_families,
        expected_arms=tpl_arms,
        expected_seeds={0},
        primary_contrasts=PRIMARY_CONTRASTS,
        secondary_contrasts=tpl_sec,
        anchor_arm="PRIOR",
        elo_module=elo_mod,
        shortfalls=0,
        metric_key="val_reg_inner",
    )

    w1_unrev_total = sum(s["w1_fold_unrevealed_slots"] for s in shards)
    w1_unrev_exact = sum(s["w1_fold_unrevealed_seen_exact_slots"] for s in shards)
    w1_unrev_tpl = sum(s["w1_fold_unrevealed_seen_template_slots"] for s in shards)
    h72_all_total = sum(s["h72_fold_total_slots"] for s in shards)
    h72_all_exact = sum(s["h72_fold_seen_exact_slots"] for s in shards)
    exact_changed_ctxs = sum(s["exact_id_top4_changed_contexts"] for s in shards)

    coverage_summary = {
        "w1_folds_unrevealed_slots": w1_unrev_total,
        "w1_folds_unrevealed_seen_exact_slots": w1_unrev_exact,
        "w1_folds_unrevealed_seen_exact_fraction": float(
            w1_unrev_exact / max(1, w1_unrev_total)
        ),
        "w1_folds_unrevealed_seen_template_slots": w1_unrev_tpl,
        "w1_folds_unrevealed_seen_template_fraction": float(
            w1_unrev_tpl / max(1, w1_unrev_total)
        ),
        "h72_folds_total_slots": h72_all_total,
        "h72_folds_seen_exact_slots": h72_all_exact,
        "h72_folds_seen_exact_fraction": float(h72_all_exact / max(1, h72_all_total)),
        "exact_id_top4_changed_contexts_count": exact_changed_ctxs,
        "exact_id_top4_changed_contexts_fraction": float(
            exact_changed_ctxs / len(expected_contexts)
        ),
    }

    shard_summaries = [
        {k: v for k, v in s.items() if not k.startswith("per_ctx_")} for s in shards
    ]

    families_sorted = sorted(expected_families)
    u0_family_mean = float(
        mean(
            mean(u0_by_ctx[c] for c in expected_contexts if fam[c] == f)
            for f in families_sorted
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_context_vectors.json").write_text(
        json.dumps(
            {
                "per_context_u4": per_context_u4,
                "per_context_val_reg_inner": per_context_leg,
                "family": fam,
            },
            sort_keys=True,
        )
    )

    metrics = {
        "schema": "jope.w6.h100-proposer-transfer-prep/v1",
        "wave": "W6",
        "h": "H100",
        "run_id": run_id,
        "scope": "DL-Train only (1,219 contexts / 205 families + 50-context H94/H95 prospective audit); zero DL-Val, outer-test, sealed, or holdout reads; zero GPU",
        "pins": PINS,
        "source_sha256": sha256_file(Path(__file__)),
        "native_fits": 0,
        "gpu_seconds": 0,
        "family_mean_U0": u0_family_mean,
        "coverage_and_stratum_census": coverage_summary,
        "track2_verified_u4": track2_u4_panel,
        "track1_legacy_val_reg_inner": track1_leg_panel,
        "within_context_seen_templates_u4": seen_tpl_u4_panel,
        "within_context_unseen_templates_u4": unseen_tpl_u4_panel,
        "within_context_seen_templates_val_reg_inner": seen_tpl_leg_panel,
        "within_context_unseen_templates_val_reg_inner": unseen_tpl_leg_panel,
        "prospective_h94_h95_true_w1_audit": prospective_audit,
        "shard_receipts": shard_summaries,
        "host_receipts": host_receipts or [],
        "verdict": {
            "cached_screen_track2_u4": track2_u4_panel["verdict"],
            "cached_screen_track1_legacy": track1_leg_panel["verdict"],
            "unseen_templates_track2_u4": unseen_tpl_u4_panel["verdict"],
            "prospective_new_programs": (
                prospective_audit["evaluated_subset_panel"]["verdict"]
                if prospective_audit
                else "NOT_RUN"
            ),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_pack = sub.add_parser("pack", help="Verify pins and pack compact numeric bundle")
    sp_pack.add_argument("--out", type=Path, required=True)
    sp_pack.add_argument(
        "--input-dir", type=Path, default=Path.home() / "jope-data/h50/input"
    )
    sp_pack.add_argument("--w1-dir", type=Path, default=Path.home() / "jope-data/w1")
    sp_pack.add_argument("--h72-roster", type=Path, required=True)
    sp_pack.add_argument("--h72-folds", type=Path, required=True)
    sp_pack.add_argument("--h85-vectors", type=Path, required=True)

    sp_shard = sub.add_parser("shard", help="Run a single outer fold shard k in 0..4")
    sp_shard.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    sp_shard.add_argument("--bundle", type=Path, required=True)
    sp_shard.add_argument("--out", type=Path, required=True)

    sp_merge = sub.add_parser("merge", help="Merge 5 fold shards and run prospective audit")
    sp_merge.add_argument("--shards", type=Path, nargs=5, required=True)
    sp_merge.add_argument("--out-dir", type=Path, required=True)
    sp_merge.add_argument("--run-id", required=True)
    sp_merge.add_argument(
        "--input-dir", type=Path, default=Path.home() / "jope-data/h50/input"
    )
    sp_merge.add_argument("--w1-dir", type=Path, default=Path.home() / "jope-data/w1")
    sp_merge.add_argument("--h85-models", type=Path, required=True)
    sp_merge.add_argument("--h94-fits-gz", type=Path, required=True)
    sp_merge.add_argument("--h94-rows-gz", type=Path, required=True)
    sp_merge.add_argument("--h95-confirmed-hwm", type=Path, required=True)
    sp_merge.add_argument("--host-receipts", type=Path, default=None)

    args = ap.parse_args()
    if args.cmd == "pack":
        info = pack_compact_bundle(
            out_npz_path=args.out,
            input_dir=args.input_dir,
            w1_dir=args.w1_dir,
            h72_roster_path=args.h72_roster,
            h72_fold_path=args.h72_folds,
            h85_vectors_path=args.h85_vectors,
        )
        print(json.dumps(info, indent=2), flush=True)
        return 0
    if args.cmd == "shard":
        res = run_fold_shard_from_bundle(
            fold_k=args.fold,
            bundle_path=args.bundle,
            out_path=args.out,
        )
        print(
            json.dumps(
                {
                    k: res[k]
                    for k in (
                        "fold",
                        "held_contexts",
                        "held_families",
                        "w1_fold_unrevealed_seen_exact_slots",
                        "h72_fold_seen_exact_slots",
                        "wall_s",
                        "cpu_s",
                    )
                }
            ),
            flush=True,
        )
        return 0
    if args.cmd == "merge":
        prosp = evaluate_prospective_h94_with_true_w1(
            input_dir=args.input_dir,
            w1_dir=args.w1_dir,
            h85_models_path=args.h85_models,
            h94_fits_gz_path=args.h94_fits_gz,
            h94_rows_gz_path=args.h94_rows_gz,
            h95_confirmed_hwm_path=args.h95_confirmed_hwm,
        )
        h_rec = (
            json.loads(args.host_receipts.read_text())
            if args.host_receipts and args.host_receipts.exists()
            else []
        )
        m = merge_and_analyze_shards(
            shard_paths=list(args.shards),
            input_dir=args.input_dir,
            out_dir=args.out_dir,
            run_id=args.run_id,
            prospective_audit=prosp,
            host_receipts=h_rec,
        )
        print(json.dumps(m["verdict"], indent=2), flush=True)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
