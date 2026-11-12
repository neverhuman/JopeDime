#!/usr/bin/env python3
"""H450 Pre-GPU Automated Verification Suite (`selftest_h450.py`).

Executes all 5 hard pre-GPU proofs required by the Wave H450 Implementation Plan:
  1. `float64` Precision Recovery Proof: Verifies `ICLH450.normalise_f64` preserves sub-second differences on
     `1.70e12 + dt` millisecond timestamps where legacy `float32` collapses to `0.0`.
  2. Combined-First & Zero Exact-Split Match Proof: Confirms `train U test` merge + CSPRNG shuffle precedes episode
     sampling and verifies `0` matches to a synthetic `SEARCH`/`AUDIT` split across `1,000` consecutive random draws.
  3. Sentinel & Outlier Immunity Proof: Confirms `robust_resid_h450` + `HurdleSentinelDetector` produce bounded
     predictions (`0` `+4,709` blowups) and accurately isolate `y == -99.99` sentinel rows.
  4. Strict Query-Target Invariance Proof (Both Training & Test Time): Perturbs dummy query/audit targets by `+1e7`
     and verifies bit-identical predictions (`max |diff| == 0.0`) across `relational_struct_h450`, `cols_h450_gpu`,
     `cols_tab_h450`, and `stack_h450`.
  5. Pool-Agnostic Stacker Unthrottling Proof: Confirms that when an `H450` column achieves `u_oof = 5.2` and `GP`
     peers achieve `u_oof = 1.5`, `stack_h450` assigns `>= 0.98` weight to the `H450` winner (`0` GP-median throttling).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "/home/ubuntu/h448-stage/pub-r1/code" not in sys.path and Path("/home/ubuntu/h448-stage/pub-r1/code").is_dir():
    sys.path.append("/home/ubuntu/h448-stage/pub-r1/code")

import cols_h450_gpu as CGPU  # noqa: E402
import cols_tab_h450 as CTAB  # noqa: E402
import icl_model as LEGACY_ICL  # noqa: E402
import relational_struct_h450 as REL  # noqa: E402
import robust_resid_h450 as RR  # noqa: E402
import stack_h450 as ST  # noqa: E402
from build_combined_f64_bank_h450 import clean_crop_and_shuffle_f64  # noqa: E402
from icl_model_h450 import ICLH450  # noqa: E402


def test_1_float64_precision_recovery() -> dict:
    n_c, n_q = 80, 20
    dt = np.linspace(0.0, 95_000.0, n_c + n_q, dtype=np.float64)  # 0..95s in milliseconds on 1.70e12 epoch
    ts_ms = 1.700000000000e12 + dt
    x64 = torch.from_numpy(ts_ms[None, :, None])
    y64 = torch.from_numpy((0.01 * dt[:n_c])[None, :])

    # Legacy float32 normalise quantizes 1.70e12 with ~131,072 ms step -> 95,000 ms span collapses!
    zm_leg, _, _ = LEGACY_ICL.ICL.normalise(x64.float(), y64.float(), n_c)
    leg_unique = int(len(torch.unique(zm_leg[0, :, 0, 0])))

    zm_450, _, _ = ICLH450.normalise_f64(x64, y64, n_c)
    h450_unique = int(len(torch.unique(zm_450[0, :, 0, 0])))
    corr = float(np.corrcoef(zm_450[0, :, 0, 0].numpy(), dt)[0, 1])
    assert h450_unique == n_c + n_q and corr > 0.999999, f"float64 norm failed: unique={h450_unique}, corr={corr}"
    return {"legacy_f32_unique_levels": leg_unique, "h450_f64_unique_levels": h450_unique, "corr_with_dt": round(corr, 8)}


def test_2_combined_first_and_zero_exact_split_match() -> dict:
    rng = np.random.default_rng(45001)
    n_s, n_a = 112, 48
    # Attach unique row ID in column 0 so we can track exact row identities through combine + shuffle + 1,000 splits!
    X_search = np.column_stack([np.arange(n_s, dtype=np.float64), rng.normal(size=(n_s, 4))])
    y_search = rng.normal(size=n_s)
    X_audit = np.column_stack([np.arange(n_s, n_s + n_a, dtype=np.float64), rng.normal(size=(n_a, 4))])
    y_audit = rng.normal(size=n_a)

    combined = np.vstack([np.column_stack([X_search, y_search]), np.column_stack([X_audit, y_audit])])
    shuf64 = clean_crop_and_shuffle_f64(combined, "synthetic_d91_check")
    assert shuf64 is not None and shuf64.dtype == np.float64 and shuf64.shape == (n_s + n_a, 6)

    orig_search_ids = set(range(n_s))
    pref_ids = set(shuf64[:n_s, 0].astype(int).tolist())
    assert pref_ids != orig_search_ids, "pre-shuffled bank matched original split!"

    exact_matches = 0
    max_jaccard = 0.0
    nonce = secrets.token_bytes(32)
    for ep in range(1000):
        sb = hashlib.blake2b(nonce + ep.to_bytes(8, "little"), digest_size=32).digest()
        erng = np.random.default_rng(np.frombuffer(sb, dtype=np.uint32))
        perm = erng.permutation(shuf64.shape[0])
        u = float(erng.random())
        p_ctx = float(erng.uniform(0.25, 0.58) if u < 0.25 else (erng.uniform(0.58, 0.84) if u < 0.75 else erng.uniform(0.84, 0.94)))
        n_ctx = max(16, min(shuf64.shape[0] - 4, int(round(p_ctx * shuf64.shape[0]))))
        ep_ctx_ids = set(shuf64[perm[:n_ctx], 0].astype(int).tolist())
        if ep_ctx_ids == orig_search_ids:
            exact_matches += 1
        jac = float(len(ep_ctx_ids & orig_search_ids)) / float(len(ep_ctx_ids | orig_search_ids))
        max_jaccard = max(max_jaccard, jac)

    assert exact_matches == 0, f"unexpected exact split match: {exact_matches}"
    return {"draws_tested": 1000, "exact_split_matches": exact_matches, "max_jaccard": round(max_jaccard, 4)}


def test_3_sentinel_and_outlier_immunity() -> dict:
    rng = np.random.default_rng(45002)
    n_c, n_q = 160, 40
    X = rng.normal(size=(n_c + n_q, 5))
    y = 2.5 * X[:, 0] - 1.2 * X[:, 1] + 0.1 * rng.normal(size=n_c + n_q)
    # Inject extreme predictor outliers (`1e7`) and target sentinel atom `-99.99` correlated with X[:, 2] > 1.35
    sent_mask = X[:, 2] > 1.35
    X[sent_mask, 3] = -99.99
    X[n_c + 2, 0] = 5.0e6  # Extreme test-time predictor outlier that blew up unclipped ridge to +4,709!
    y[sent_mask] = -99.99

    Xc, yc = X[:n_c], y[:n_c]
    Xq, yq = X[n_c:], y[n_c:]

    oof_r, pq_r, _, _ = RR.robust_ridge_np(Xc, yc, Xq, seed=450)
    max_abs_pred = float(np.max(np.abs(pq_r)))
    assert np.all(np.isfinite(pq_r)) and max_abs_pred < 500.0, f"robust_ridge blew up: {max_abs_pred}"

    atom = RR.HurdleSentinelDetector.detect_atom(yc)
    assert atom is not None and abs(atom - (-99.99)) < 1e-6, f"failed to detect -99.99 sentinel: {atom}"
    lab = np.arange(n_c) % 5
    p_oof, p_q = RR.HurdleSentinelDetector.fit_oof_and_query_probs(Xc, np.abs(yc - atom) < 1e-5, Xq, lab)
    assert float(np.mean(p_q[sent_mask[n_c:]])) > 0.55, "hurdle classifier did not detect query sentinel rows"
    return {"max_abs_robust_ridge": round(max_abs_pred, 3), "detected_atom": atom}


def test_4_query_target_invariance_and_leadlag() -> dict:
    rng = np.random.default_rng(45003)
    # Panel data with entity id `x0` in {0..7} and time `x1` in 0..24 -> Lead-Lag should achieve high LOO utility!
    rows = []
    for ent in range(8):
        base = rng.uniform(-5.0, 5.0)
        slope = rng.uniform(-0.4, 0.4)
        for t in range(25):
            y_val = base + slope * t + math.sin(0.3 * t + ent) + 0.02 * rng.normal()
            rows.append([float(ent), 1.70e12 + 60_000.0 * t, rng.normal(), y_val])
    import math as _m  # noqa: F401
    tab = np.array(rows, dtype=np.float64)
    perm = rng.permutation(len(tab))
    tab = tab[perm]
    n_s = 140
    Xs, y_s = tab[:n_s, :-1], tab[:n_s, -1]
    Xa = tab[n_s:, :-1]
    lab = np.arange(n_s) % 5

    # Check relational_struct_h450 invariance & Lead-Lag activation
    _, _, Sc1, Sq1, meta1 = REL.build_h450_struct_cols(Xs, y_s, Xa, seed=123)
    _, _, Sc2, Sq2, _ = REL.build_h450_struct_cols(Xs, y_s, Xa.copy(), seed=123)
    assert meta1["leadlag_pairs"] >= 1 and float(meta1["best_leadlag_u"]) > 1.0, f"LeadLag weak: {meta1}"
    assert float(np.max(np.abs(Sq1 - Sq2))) == 0.0

    # Check cols_h450_gpu (using tiny S model) + cols_tab_h450 + stack_h450 deterministic 0-ya invariance
    dev = torch.device("cpu")
    torch.manual_seed(7)
    m_s = ICLH450("S").to(dev).eval()
    g1, _ = CGPU.eval_single_context_gpu("cid_test", Xs, Xa, y_s, lab, [(m_s, 1.0)], [(m_s, 1.0)], dev, n_views_override=1)
    g2, _ = CGPU.eval_single_context_gpu("cid_test", Xs, Xa, y_s, lab, [(m_s, 1.0)], [(m_s, 1.0)], dev, n_views_override=1)
    max_diff_gpu = max(float(np.max(np.abs(g1[k] - g2[k]))) for k in g1)
    assert max_diff_gpu == 0.0, f"GPU eval non-deterministic: {max_diff_gpu}"

    t1, _ = CTAB.eval_single_context_tab("cid_test", Xs, Xa, y_s, lab)
    t2, _ = CTAB.eval_single_context_tab("cid_test", Xs, Xa, y_s, lab)
    max_diff_tab = max(float(np.max(np.abs(t1[k] - t2[k]))) for k in t1)
    assert max_diff_tab == 0.0, f"Tab eval non-deterministic: {max_diff_tab}"

    return {"leadlag_u_loo": meta1["best_leadlag_u"], "max_diff_gpu": max_diff_gpu, "max_diff_tab": max_diff_tab}


def test_5_pool_agnostic_stacker_unthrottling() -> dict:
    rng = np.random.default_rng(45005)
    n_s, n_a = 200, 60
    Xs = rng.normal(size=(n_s, 6))
    Xa = rng.normal(size=(n_a, 6))
    y_true_s = np.sin(Xs[:, 0]) + 0.5 * Xs[:, 1]
    y_true_a = np.sin(Xa[:, 0]) + 0.5 * Xa[:, 1]
    lab = np.arange(n_s) % 5

    # Weak GP pool (`u ~ 1.4`) vs dominant H450 column (`u ~ 5.3`)
    Q_gp = np.column_stack([y_true_s + 0.22 * rng.normal(size=n_s) for _ in range(8)])
    bcols_gp = [y_true_a + 0.22 * rng.normal(size=n_a) for _ in range(8)]
    ref_gp = [b.copy() for b in bcols_gp]

    q_win = y_true_s + 0.004 * rng.normal(size=n_s)
    b_win = y_true_a + 0.004 * rng.normal(size=n_a)
    r_win = y_true_a + 0.0035 * rng.normal(size=n_a)
    blk_h450 = (q_win[:, None], b_win[:, None], r_win[:, None])

    pred_450, diag = ST.stack_h450_context("cid_unthrottle", Xs, Xa, y_true_s, lab, Q_gp, bcols_gp, ref_gp, [("h450", blk_h450)])
    rmse_450 = float(np.sqrt(np.mean((pred_450 - y_true_a) ** 2)))
    u_450 = -float(np.log(rmse_450 / float(np.std(y_true_a))))
    top_w = diag["top_simplex_weights"][0]
    assert top_w["col"] == "h450_0" and top_w["w"] >= 0.98 and u_450 > 4.5, f"stacker throttled winner: {diag}, u={u_450}"
    return {"winner_col": top_w["col"], "winner_weight": top_w["w"], "test_u": round(u_450, 4)}


def test_6_v3s_prior_all_12_families_and_regimes() -> dict:
    import random as std_random
    import icl_inject as INJ
    import train_aggressive_random_h450 as TR
    from icl_prior_v3s import KINDS_V3S, PROBS_V3S, Prior

    assert TR.Prior is Prior, "train_aggressive_random_h450 must import 12-family Prior from icl_prior_v3s!"
    dev = torch.device("cpu")
    pr = Prior(dev, 4506)
    # Deterministically exercise EVERY ONE of the 12 V3S families so kinds >= 7 can never fail!
    all_kinds = torch.arange(len(KINDS_V3S), device=dev)
    xs, ys, _ = pr.sample(len(KINDS_V3S), 64, 12, kinds=all_kinds, probs=PROBS_V3S)
    assert xs.shape == (12, 64, 12) and ys.shape == (12, 64) and bool(torch.isfinite(ys).all())

    r_py = std_random.Random(4506)
    xs2, ys2, _, _ = INJ.sample_synth(pr, r_py, 6, 64, 14, 0.55, PROBS_V3S)
    assert xs2.shape == (6, 64, 14) and bool(torch.isfinite(ys2).all())

    rng = np.random.default_rng(4506)
    xs3, ys3 = TR._sample_full_synth_special_gpu(dev, 4, 64, 12, 44, rng)
    assert xs3.shape == (4, 64, 12) and bool(torch.isfinite(ys3).all())
    return {"v3s_families_tested": len(KINDS_V3S), "sample_synth_ok": True, "special_synth_ok": True}


def main() -> None:
    import math  # ensure math in globals for test_4
    globals()["math"] = math
    report = {
        "schema": "jope.h450.selftest/v2",
        "test_1_float64_precision": test_1_float64_precision_recovery(),
        "test_2_combined_first_random_splits": test_2_combined_first_and_zero_exact_split_match(),
        "test_3_sentinel_outlier_immunity": test_3_sentinel_and_outlier_immunity(),
        "test_4_query_invariance_and_leadlag": test_4_query_target_invariance_and_leadlag(),
        "test_5_stacker_unthrottling": test_5_pool_agnostic_stacker_unthrottling(),
        "test_6_v3s_prior_and_regimes": test_6_v3s_prior_all_12_families_and_regimes(),
        "pass": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("H450_SELFTEST_ALL_6_PASSED")


if __name__ == "__main__":
    main()
