#!/usr/bin/env python3
"""Synthetic unit verification for Wave C (`w5_online_ast_screener`): 5 invariants, zero real table read."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w5_screener_core as W5  # noqa: E402


def main() -> None:
    # 1. Grammar completeness on H420 extended stages, kernels, and symbolic expressions (zero ops stripped)
    d_base = "(pipe X (standardize) (gpr kernel=rbf m=500 steps=20))"
    d_ext = "(pipe X (standardize) (norm_row append=true) (derive append=true expr=c1/c4) (gpr kernel=rq+linear m=2000 steps=120))"
    v_base = W5.encode_program_ast(d_base)
    v_ext = W5.encode_program_ast(d_ext)
    c1 = bool(v_ext.shape == (48,) and np.all(np.isfinite(v_ext)) and v_ext[40] >= 2.0 and v_ext[42] == 1.0 and v_ext[45] == 1.0 and not np.allclose(v_base, v_ext))

    # 2. Dataset meta-profile finite and bounded on skewed + missing synthetic data
    rng = np.random.default_rng(435)
    X = rng.normal(size=(180, 12))
    X[rng.random(X.shape) < 0.08] = np.nan
    X[:, 0] = rng.integers(0, 6, size=180)
    y = np.exp(0.7 * np.sin(np.nan_to_num(X[:, 1]))) + 0.1 * rng.normal(size=180)
    phi = W5.compute_dataset_meta_profile(X, y)
    c2 = bool(phi.shape == (16,) and np.all(np.isfinite(phi)))

    # 3. Cross-family surrogate + Online AST Ridge Adapter beats random & head-only adapter on synthetic family pool
    dsl_pool = [
        "(pipe X (standardize) (ridge_gcv))",
        "(pipe X (blom) (spline append=true knots=8) (ridge_gcv))",
        "(pipe X (standardize) (gpr kernel=rbf m=500 steps=20))",
        "(pipe X (standardize) (derive append=true expr=c0/c1) (gpr kernel=rq+linear m=2000 steps=120))",
        "(pipe X (row_stats append=true) (map append=true expr=x-row_mean) (histgbm2 bags=5 leaves=63 lr=0.03 rounds=800))",
        "(pipe X (histgbm depth=4 lr=0.1 rounds=100))",
        "(pipe X (svd_project append=true k=8) (oblivious depth=7 l2=3.0 lr=0.03 rounds=800))",
        "(pipe X (standardize) (krr m=2000))",
    ] * 8  # 64 candidates with hyperparam jitter
    dsl_pool = [f"{d[:-1]} seed={i})" if "ridge_gcv" not in d else d for i, d in enumerate(dsl_pool)]

    records = []
    for f_i in range(12):
        phi_f = rng.normal(size=16)
        # True latent utility rewards H420 rq+linear & derive when phi_f[12] > 0, else spline ridge
        psis = np.array([W5.encode_program_ast(d) for d in dsl_pool])
        u_f = 0.8 * psis[:, 42] + 0.7 * psis[:, 45] + (0.6 if f_i % 2 == 0 else -0.4) * psis[:, 7] + 0.15 * rng.normal(size=len(dsl_pool))
        records.append((f"fam_{f_i // 2}", phi_f, dsl_pool, u_f))

    surr = W5.GrammarCompleteASTSurrogate(n_members=5).fit(records[:10], seed=99)
    _, phi_te, dsls_te, _ = records[10]
    psis_te = np.array([W5.encode_program_ast(d) for d in dsls_te])
    # Novel table-specific twist: derive ratio (slot 45) + histgbm2 (slot 21) strongly favored on this table!
    u_te = 1.5 * psis_te[:, 45] + 1.2 * psis_te[:, 21] - 0.8 * psis_te[:, 15] + 0.05 * rng.normal(size=len(dsls_te))
    obs_idx = list(range(24))
    rem_idx = list(range(24, 64))
    ucb_sc, mean_sc = surr.adapt_online_and_score(
        phi_te, [dsls_te[i] for i in obs_idx], u_te[obs_idx], [dsls_te[i] for i in rem_idx]
    )
    raw_blind = rng.normal(size=len(dsls_te)) * 0.01  # blind screener
    h01_sc = W5.h01a_head_residual_adapt(
        raw_blind[obs_idx], [dsls_te[i] for i in obs_idx], u_te[obs_idx], raw_blind[rem_idx], [dsls_te[i] for i in rem_idx]
    )
    picks_w5 = W5.select_diverse_topk(ucb_sc, [dsls_te[i] for i in rem_idx], k_select=6)
    picks_h01 = np.argsort(-h01_sc)[:6].tolist()
    c3 = bool(float(np.max(u_te[rem_idx][picks_w5])) >= float(np.max(u_te[rem_idx][picks_h01])))

    # 4. Deterministic UCB diversity selection returns exact unique indices
    picks_rep = W5.select_diverse_topk(ucb_sc, [dsls_te[i] for i in rem_idx], k_select=6)
    c4 = bool(picks_w5 == picks_rep and len(set(picks_w5)) == 6)

    # 5. Head family parser extracts top-level head accurately
    c5 = bool(
        W5.head_family("(pipe X (standardize) (gpr kernel=rq m=1200))") == "gpr"
        and W5.head_family("(portfolio (pipe X (ridge_gcv)) (pipe X (krr m=400)))") == "portfolio"
    )

    checks = {
        "h420_grammar_complete": c1,
        "meta_profile_finite": c2,
        "online_adapter_beats_blind": c3,
        "ucb_diverse_deterministic": c4,
        "head_family_parser": c5,
    }
    res = {"checks": checks, "all_passed": all(checks.values())}
    print(json.dumps(res, sort_keys=True))
    if not res["all_passed"]:
        raise SystemExit(1)
    print("W5_UNIT_TESTS_OK 5/5")


if __name__ == "__main__":
    main()
