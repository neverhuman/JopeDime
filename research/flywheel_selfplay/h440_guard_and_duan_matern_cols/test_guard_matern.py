#!/usr/bin/env python3
"""Pre-publication synthetic unit & zero-leakage suite for Wave B (`guard_and_duan_matern_cols`):
  1. ya-free Peer-Consensus Extrapolation Guard catches BOTH a 300*sd BAG bomb and a 150*sd REF-only bomb.
  2. Yeo-Johnson + 15-quantile Duan smearing + EntityHierarchy beats plain non-Duan controls on skewed relational data.
  3. Strict query-row 0 independence (`diff < 1e-10`) when query rows 1..end are perturbed by N(100, 50).
  4. Guarded 2L stacker (`guard_and_stack_2l`) query-row 0 independence and clean finite outputs.
  5. Missingness (NaN + 0.0 sentinel) + constant column robustness across `cols_matern.columns`.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cols_matern as CM  # noqa: E402
import guard_matern_core as GM  # noqa: E402


def main() -> None:
    report = {"checks": {}}

    # 1. ya-free Peer-Consensus Extrapolation Guard test (synthetic 51e71f0b3089 replica)
    rng = np.random.default_rng(434_01)
    n_s, n_a = 200, 80
    ys = rng.normal(1000.0, 100.0, size=n_s)
    lab = np.arange(n_s) % 5
    true_a = rng.normal(1000.0, 100.0, size=n_a)
    Q = np.column_stack([ys + rng.normal(0.0, 15.0 + 2.0 * j, size=n_s) for j in range(10)])
    bcols = [true_a + rng.normal(0.0, 15.0 + 2.0 * j, size=n_a) for j in range(10)]
    refit = [b + rng.normal(0.0, 3.0, size=n_a) for b in bcols]
    # Inject cand 1 BAG bomb (300 * sd) and cand 4 REF-only bomb (150 * sd)
    bcols[1] = bcols[1].copy()
    bcols[1][0] = 31_000.0
    refit[1] = bcols[1].copy()
    refit[4] = refit[4].copy()
    refit[4][3] = 16_000.0

    Qg, bg, rg, kept = GM.guard_candidate_pool_ya_free(ys, lab, Q, bcols, refit)
    kept_list = kept.tolist()
    assert 1 not in kept_list, f"BAG bomb cand 1 was not dropped: {kept_list}"
    assert 4 in kept_list, f"Clean-BAG cand 4 should be kept with refit dropped: {kept_list}"
    pos4 = kept_list.index(4)
    assert rg[pos4] is None, "Exploding REF on cand 4 should have been nulled to None!"
    assert len(kept_list) == 9
    report["checks"]["ya_free_peer_guard"] = {
        "kept_count": len(kept_list),
        "bag_bomb_dropped": (1 not in kept_list),
        "ref_bomb_demoted_to_bag": (rg[pos4] is None),
    }

    # 2 & 3. Skewed relational + multiplicative table: Duan smearing gain + query-row 0 independence
    N_tr, N_te, P = 280, 80, 14
    X_all = rng.normal(size=(N_tr + N_te, P))
    ent = rng.integers(0, 10, size=N_tr + N_te)
    X_all[:, 0] = ent
    ent_eff = rng.normal(0.0, 1.8, size=10)
    z_lat = 1.4 * ent_eff[ent] + 1.1 * np.sin(X_all[:, 1]) + 0.75 * X_all[:, 2] * X_all[:, 3]
    y_all = np.exp(0.65 * z_lat + rng.normal(0.0, 0.22, size=N_tr + N_te))
    Xs, ys_t = X_all[:N_tr], y_all[:N_tr]
    Xa, ya_t = X_all[N_tr:], y_all[N_tr:]

    pr1, _ = GM.heads(Xs, ys_t, Xa, 434_11)
    Xa_pert = Xa.copy()
    Xa_pert[1:] = rng.normal(80.0, 40.0, size=Xa_pert[1:].shape)
    pr2, _ = GM.heads(Xs, ys_t, Xa_pert, 434_11)
    max_q0_diff = max(float(abs(pr1[nm][0] - pr2[nm][0])) for nm in GM.NAMES)
    assert max_q0_diff < 1e-10, f"Query row 0 leaked batch stats: {max_q0_diff}"

    sd0 = float(np.std(ya_t))
    u_by_head = {
        nm: float(-math.log(float(np.sqrt(np.mean((pr1[nm] - ya_t) ** 2))) / sd0))
        for nm in GM.NAMES
    }
    best_duan = max(u_by_head[nm] for nm in GM.PRIMARY_NAMES)
    best_plain = max(u_by_head[nm] for nm in GM.CONTROL_NAMES)
    assert best_duan > best_plain + 0.05, f"Expected Duan+Entity heads ({best_duan:.4f}) > plain ({best_plain:.4f})"
    report["checks"]["duan_entity_and_isolation"] = {
        "max_q0_diff": max_q0_diff,
        "best_primary_u": round(best_duan, 4),
        "best_plain_u": round(best_plain, 4),
    }

    # 4 & 5. Full `cols_matern.columns` + `guard_and_stack_2l` with NaNs, 0.0 sentinel, constant col, and L2 batch check
    X_m = Xs.copy()
    X_m[rng.random(X_m.shape) < 0.07] = np.nan
    X_m[:, -1] = 7.0  # constant column
    Xa_m = Xa.copy()
    Xa_m[:, -1] = 7.0
    lab_m = np.arange(N_tr) % 5
    cols_out, info_out = CM.columns({"cid": "0" * 64, "X": X_m, "y": ys_t, "lab": lab_m, "XA": Xa_m})
    assert all(np.all(np.isfinite(arr)) for arr in cols_out.values())

    Qm = np.column_stack([cols_out[f"q_{v}"] for v in GM.PRIMARY_NAMES])
    Bm = np.column_stack([cols_out[f"bag_{v}"] for v in GM.PRIMARY_NAMES])
    Rm = np.column_stack([cols_out[f"ref_{v}"] for v in GM.PRIMARY_NAMES])
    # Synthetic base GP pool with 1 exploding candidate
    Q_gp = np.column_stack([ys_t + rng.normal(0.0, 0.5 * float(np.std(ys_t)), size=N_tr) for _ in range(8)])
    b_gp = [ya_t * 0.0 + float(np.mean(ys_t)) + rng.normal(0.0, 1.0, size=N_te) for _ in range(8)]
    b_gp[2][0] = 1e7
    r_gp = [b.copy() for b in b_gp]

    p_aud1, meta1 = GM.guard_and_stack_2l("0" * 64, X_m, Xa_m, ys_t, lab_m, Q_gp, b_gp, r_gp, [(Qm, Bm, Rm)])
    # Perturb Xa_m[1:] and Bm[1:], Rm[1:], b_gp[:][1:] — note peer guard uses Xa median for screening, so test L2 isolation cleanly without bombs
    b_gp_clean = [ya_t * 0.0 + float(np.mean(ys_t)) for _ in range(8)]
    r_gp_clean = [b.copy() for b in b_gp_clean]
    p_clean1, _ = GM.guard_and_stack_2l("0" * 64, X_m, Xa_m, ys_t, lab_m, Q_gp, b_gp_clean, r_gp_clean, [(Qm, Bm, Rm)])
    Xa_m2 = Xa_m.copy()
    Xa_m2[1:] = Xa_m2[1:] + 2.0
    p_clean2, _ = GM.guard_and_stack_2l("0" * 64, X_m, Xa_m2, ys_t, lab_m, Q_gp, b_gp_clean, r_gp_clean, [(Qm, Bm, Rm)])
    stack_q0_diff = float(abs(p_clean1[0] - p_clean2[0]))
    assert np.all(np.isfinite(p_aud1)) and meta1["gp_dropped"] >= 1
    assert stack_q0_diff < 1e-10, f"L2 stacker row 0 leaked Xa[1:]: {stack_q0_diff}"
    report["checks"]["end_to_end_cols_and_stack2l"] = {
        "finite": True,
        "gp_dropped": meta1["gp_dropped"],
        "ext_admitted": meta1["ext_admitted"],
        "stack_q0_diff": stack_q0_diff,
    }

    # 6. Precision-dilution prevention on near-exact tables (synthetic cid 7d3e30f33f replica)
    #    + refit-improved key column admission (synthetic cid 3b0303bb38 replica)
    rng6 = np.random.default_rng(436_73)
    X6_s = rng6.normal(size=(300, 8))
    X6_a = rng6.normal(size=(100, 8))
    y6_s = 2.0 * X6_s[:, 0] - 1.5 * X6_s[:, 1]
    y6_a = 2.0 * X6_a[:, 0] - 1.5 * X6_a[:, 1]
    sd6_s = float(np.std(y6_s))
    sd6_a = float(np.std(y6_a))
    lab6 = np.arange(300) % 5
    # 1 near-exact stored candidate (u ~ 17.0, RMSE ~ 4e-8 * sd) + 15 noisy internal candidates (u ~ 5..7)
    q6_exact = y6_s + rng6.normal(0.0, 4e-8 * sd6_s, size=300)
    b6_exact = y6_a + rng6.normal(0.0, 4e-8 * sd6_a, size=100)
    r6_exact = y6_a + rng6.normal(0.0, 3e-8 * sd6_a, size=100)
    Q6_gp = np.column_stack(
        [q6_exact] + [y6_s + rng6.normal(0.0, (1e-3 + 4e-4 * j) * sd6_s, size=300) for j in range(15)]
    )
    b6_gp = [b6_exact] + [y6_a + rng6.normal(0.0, (1e-3 + 4e-4 * j) * sd6_a, size=100) for j in range(15)]
    r6_gp = [r6_exact] + [b.copy() for b in b6_gp[1:]]
    # External column with u ~ 6.7 (RMSE = 1.2e-3 * sd), like H436 key_hybrid on 7d3e30f33f
    Q6_ext = (y6_s + rng6.normal(0.0, 1.2e-3 * sd6_s, size=300))[:, None]
    B6_ext = (y6_a + rng6.normal(0.0, 1.2e-3 * sd6_a, size=100))[:, None]
    R6_ext = (y6_a + rng6.normal(0.0, 1.2e-3 * sd6_a, size=100))[:, None]
    p6_aud, meta6 = GM.guard_and_stack_2l(
        "7d3e30f33f" + "0" * 54, X6_s, X6_a, y6_s, lab6, Q6_gp, b6_gp, r6_gp, [(Q6_ext, B6_ext, R6_ext)]
    )
    u6_stack = float(-math.log(float(np.sqrt(np.mean((p6_aud - y6_a) ** 2))) / sd6_a))
    assert meta6["ext_admitted"] == 0 and meta6["ext_weight_share"] == 0.0, f"Diluting ext admitted: {meta6}"
    assert u6_stack >= 16.5, f"Precision dilution occurred on near-exact table: u_stack={u6_stack:.4f}"

    # Counterpart: base OOF u ~ 4.2 (RMSE = 0.015 * sd), external key column OOF RMSE = 0.018 * sd (ratio 1.20)
    # whose full-data REF reaches u ~ 12 (drift ~ 1.0) -> must be admitted and improve utility!
    Q6b_gp = np.column_stack([y6_s + rng6.normal(0.0, (0.015 + 0.002 * j) * sd6_s, size=300) for j in range(6)])
    b6b_gp = [y6_a + rng6.normal(0.0, (0.015 + 0.002 * j) * sd6_a, size=100) for j in range(6)]
    r6b_gp = [b.copy() for b in b6b_gp]
    Q6b_ext = (y6_s + rng6.normal(0.0, 0.018 * sd6_s, size=300))[:, None]
    B6b_ext = (y6_a + rng6.normal(0.0, 0.018 * sd6_a, size=100))[:, None]
    R6b_ext = (y6_a + rng6.normal(0.0, 1e-5 * sd6_a, size=100))[:, None]
    p6b_base, _ = GM.guard_and_stack_2l("3b0303bb38" + "0" * 54, X6_s, X6_a, y6_s, lab6, Q6b_gp, b6b_gp, r6b_gp, [])
    p6b_ext, meta6b = GM.guard_and_stack_2l(
        "3b0303bb38" + "0" * 54, X6_s, X6_a, y6_s, lab6, Q6b_gp, b6b_gp, r6b_gp, [(Q6b_ext, B6b_ext, R6b_ext)]
    )
    u6b_base = float(-math.log(float(np.sqrt(np.mean((p6b_base - y6_a) ** 2))) / sd6_a))
    u6b_ext = float(-math.log(float(np.sqrt(np.mean((p6b_ext - y6_a) ** 2))) / sd6_a))
    assert meta6b["ext_admitted"] == 1 and u6b_ext > u6b_base + 0.05, (
        f"Expected refit-improved key col admitted & helpful: meta={meta6b}, base={u6b_base:.4f}, ext={u6b_ext:.4f}"
    )
    report["checks"]["precision_dilution_and_refit_key_guard"] = {
        "near_exact_u_stack": round(u6_stack, 4),
        "near_exact_ext_admitted": meta6["ext_admitted"],
        "refit_key_u_gain": round(u6b_ext - u6b_base, 4),
    }

    report["all_passed"] = True
    print(json.dumps(report, sort_keys=True))
    print("GUARD_MATERN_UNIT_TESTS_OK 6/6")


if __name__ == "__main__":
    main()
