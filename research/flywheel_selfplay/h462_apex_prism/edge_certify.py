"""Fast-Clock Tail-Faithful `EDGE-CERTIFY` Selector, Raw-`y` Classic Floor & `8-Bag` (`edge_certify.py`).

Implements the reconciled `TWO CLOCKS x APEX-PRISM` Fast Clock (`ARENA §7`, `SYMBIONT §8.2`,
`cl_h458` `06:05Z`, `06:06Z`, `06:13Z`):
  1. **Cures the Clipped Fold-Mean Selection Bug (`cl_h458` `06:05Z`):**
     - `dev91` / `P201b` `lab` folds are ALREADY target-stratified (`rank spread = 0.0002`), yet legacy
       selection (`3 MAD` clip + geometric-mean `mean_k log(RMSE_k)`) discounts a `10x` tail explosion in
       one fold to `0.2 * log(10)`, whereas `AUDIT` scores **unclipped pooled `log(RMSE)`** with `3` of every
       `10` sorted target-quantile rows guaranteed in `AUDIT` (`common_h317.py:582`).
     - `edge_certify_context` evaluates candidates using **unclipped pooled scored-space RMSE** across
       `B = 2,000` **`common_h317`-shaped (`3-of-10` target-block) OOF resamples**, maximizing expected
       **family-weighted pairwise wins** (`w_fam = 5.0 / |F_c|`, giving singleton families like `866f` `5x`
       leverage!), gated by `SYMBIONT §8.2` unclipped lower bound (`lcb_delta_u > 0`, `winfrac >= 0.70`,
       `Delta family_mean_u >= -0.01`) and exact `eta = 0` anchor retention on ties/noise (avoiding `S2
       V1..V12`'s continuous-reweighting variance tax on the `73` non-violating contexts!).
  2. **Raw-`y` Unwarped Classic Floor (`fit_raw_y_classic_floor_oof`):**
     - Fits **unwarped (`raw-y`, ZERO `Yeo-Johnson`)** floor candidates `{raw_ridge, raw_hgb, raw_rf,
       raw_et, raw_ard_knn}` strictly on `lab` folds so inverse-`yj` tail blowups (`h450_cb_d4_yj` / `line 28`)
       cannot contaminate the floor.
  3. **Explicit `8-Bag` Entrant (`h462.APEX_PRISM_E8`) & `{cid}.npz` Block Packager:**
     - Prediction-space averaging strictly dominates expected MSE (`MSE(mean_m p_m) <= mean_m MSE(p_m)`)
       and emits `{cid}.npz` (`q_/bag_/ref_`) + `SIDECAR.json` compatible with `cl_mtc` and `cl_h458`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg as sla
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from .contracts_v3 import assert_path_not_quarantined, assert_sovereign_s0_candidates

ENTRANT_ID_E8 = "h462.APEX_PRISM_E8"


@dataclasses.dataclass(frozen=True)
class EdgeCertifyConfig:
    """Frozen Fast-Clock certification & win-efficiency hyperparameters."""

    n_bootstrap: int = 2000
    audit_per_10_block: int = 3
    min_bootstrap_win_prob: float = 0.70
    min_pooled_lcb_u: float = 0.005
    min_all_folds_tol_u: float = -0.03
    max_family_mean_drop_u: float = 0.01
    opponent_margins_u: Tuple[float, ...] = (-0.08, -0.03, 0.0, 0.03, 0.08)
    sovereign_tier: str = "S0"
    seed: int = 462401


@dataclasses.dataclass(frozen=True)
class EdgeContextDecision:
    """Per-context selection outcome from `edge_certify_context`."""

    selected_name: str
    anchor_name: str
    switched_from_anchor: bool
    family_weight: float
    pooled_u_selected: float
    pooled_u_anchor: float
    pooled_delta_u: float
    bootstrap_lcb10_delta_u: float
    bootstrap_win_prob_vs_anchor: float
    expected_family_weighted_wins: float
    pred_oof: np.ndarray
    pred_query: np.ndarray
    candidate_pooled_u: Dict[str, float]


def pooled_unclipped_utility(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact `common_h317` unclipped pooled log-RMSE utility `log(std(y) / rmse(y - p))`."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(p)):
        return -50.0
    y_std = max(float(np.std(y)), 1e-12)
    rmse = float(math.sqrt(np.mean((y - p) ** 2)))
    return float(math.log(y_std / max(rmse, 1e-15)))


def legacy_clipped_fold_mean_score(
    y_true: np.ndarray,
    anchor_oof: np.ndarray,
    cand_oof: np.ndarray,
    lab_folds: np.ndarray,
    mad_clip: float = 3.0,
) -> float:
    """Reproduce the legacy `3 MAD` + geometric-mean `mean_k log(RMSE_k)` score (`cl_h458` `06:05Z` audit).

    Exposed ONLY for diagnostic comparison in `selftest_h462_apex_prism.py` (`P0_4`) to prove how
    clipped fold-mean selection admits tail-exploding `_yj` columns that `EDGE-CERTIFY` blocks.
    """
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    anc = np.asarray(anchor_oof, dtype=np.float64).reshape(-1)
    cand = np.asarray(cand_oof, dtype=np.float64).reshape(-1)
    lab = np.asarray(lab_folds, dtype=np.int64).reshape(-1)

    r_anc = y - anc
    med = float(np.median(r_anc))
    mad = max(1.4826 * float(np.median(np.abs(r_anc - med))), 1e-6 * max(float(np.std(y)), 1e-9))
    y_clipped = anc + np.clip(r_anc, -float(mad_clip) * mad, float(mad_clip) * mad)
    cand_clipped = anc + np.clip(cand - anc, -2.5 * mad, 2.5 * mad)

    gains = []
    for k in range(5):
        m = lab == k
        if not np.any(m):
            continue
        b_rmse = max(float(math.sqrt(np.mean((y_clipped[m] - anc[m]) ** 2))), 1e-15)
        c_rmse = max(float(math.sqrt(np.mean((y_clipped[m] - cand_clipped[m]) ** 2))), 1e-15)
        gains.append(math.log(b_rmse / c_rmse))
    return float(np.mean(gains)) if gains else 0.0


def _sample_audit_shaped_blocks(
    y_search: np.ndarray,
    n_bootstrap: int,
    audit_per_10: int,
    seed: int,
) -> np.ndarray:
    """Vectorized `common_h317:582`-shaped stratified target-block resampling (`3` of every sorted `10`-block)."""
    y = np.asarray(y_search, dtype=np.float64).reshape(-1)
    n_rows = y.shape[0]
    order = np.argsort(y, kind="mergesort")
    rng = np.random.default_rng(int(seed))

    n_full_blocks = n_rows // 10
    k_pick = min(max(1, int(audit_per_10)), 9)
    if n_full_blocks == 0:
        return rng.integers(0, n_rows, size=(int(n_bootstrap), max(1, n_rows // 3)))

    blocks = order[: n_full_blocks * 10].reshape(n_full_blocks, 10)
    # Generate random scores (B, n_blocks, 10) and take top `k_pick` indices per 10-block -> exact without-replacement!
    noise = rng.random(size=(int(n_bootstrap), n_full_blocks, 10))
    sub_idx = np.argpartition(noise, kth=k_pick - 1, axis=2)[:, :, :k_pick]
    sampled = blocks[np.arange(n_full_blocks)[None, :, None], sub_idx].reshape(int(n_bootstrap), n_full_blocks * k_pick)

    # `cl_mtc` Correction 2 (`common_h317.py` parity): draw `round(audit_per_10 * r / 10)` rows from partial top block!
    r_rem = n_rows - n_full_blocks * 10
    k_rem = int(round(float(r_rem) * float(k_pick) / 10.0)) if r_rem > 0 else 0
    k_rem = min(max(0, k_rem), r_rem)
    if k_rem > 0:
        top_block = order[n_full_blocks * 10 :]
        noise_rem = rng.random(size=(int(n_bootstrap), r_rem))
        rem_idx = np.argpartition(noise_rem, kth=k_rem - 1, axis=1)[:, :k_rem]
        sampled_rem = top_block[rem_idx]
        sampled = np.concatenate([sampled, sampled_rem], axis=1)
    return sampled


def edge_certify_context(
    y_search: np.ndarray,
    lab_folds: np.ndarray,
    anchor_name: str,
    candidate_oof: Mapping[str, np.ndarray],
    candidate_query: Mapping[str, np.ndarray],
    family_size: int = 5,
    opponent_margins_u: Optional[Sequence[float]] = None,
    cfg: Optional[EdgeCertifyConfig] = None,
) -> EdgeContextDecision:
    """Select the candidate maximizing expected family-weighted wins subject to unclipped pooled tail safety."""
    conf = cfg or EdgeCertifyConfig()
    names = list(candidate_oof.keys())
    assert_sovereign_s0_candidates(names, sovereign_tier=conf.sovereign_tier)
    if anchor_name not in candidate_oof or anchor_name not in candidate_query:
        raise KeyError(f"anchor_name {anchor_name!r} must exist in candidate_oof and candidate_query")

    y = np.ascontiguousarray(y_search, dtype=np.float64).reshape(-1)
    lab = np.asarray(lab_folds, dtype=np.int64).reshape(-1)
    anc_oof = np.ascontiguousarray(candidate_oof[anchor_name], dtype=np.float64).reshape(-1)
    anc_qry = np.ascontiguousarray(candidate_query[anchor_name], dtype=np.float64).reshape(-1)

    fam_weight = 5.0 / float(max(int(family_size), 1))
    opp_margins = np.asarray(
        conf.opponent_margins_u if opponent_margins_u is None else opponent_margins_u,
        dtype=np.float64,
    )

    boot_idx = _sample_audit_shaped_blocks(
        y, n_bootstrap=conf.n_bootstrap, audit_per_10=conf.audit_per_10_block, seed=conf.seed
    )
    y_boot = y[boot_idx]
    y_boot_std = np.maximum(np.std(y_boot, axis=1), 1e-12)

    anc_err_sq = (y - anc_oof) ** 2
    anc_boot_rmse = np.sqrt(np.maximum(np.mean(anc_err_sq[boot_idx], axis=1), 1e-30))
    anc_boot_u = np.log(y_boot_std / anc_boot_rmse)
    anc_pooled_u = pooled_unclipped_utility(y, anc_oof)
    anc_fold_u = np.array([pooled_unclipped_utility(y[lab == k], anc_oof[lab == k]) for k in range(5)])

    cand_u_map: Dict[str, float] = {}
    best_name = anchor_name
    best_exp_wins = float(fam_weight * np.mean(0.0 > opp_margins[None, :]))
    best_lcb10 = 0.0
    best_win_prob = 0.0

    for nm in names:
        c_oof = np.ascontiguousarray(candidate_oof[nm], dtype=np.float64).reshape(-1)
        c_qry = np.ascontiguousarray(candidate_query[nm], dtype=np.float64).reshape(-1)
        u_pool = pooled_unclipped_utility(y, c_oof)
        cand_u_map[nm] = u_pool
        if nm == anchor_name:
            continue
        if not np.all(np.isfinite(c_oof)) or not np.all(np.isfinite(c_qry)):
            continue

        # Hard unclipped per-fold catastrophe guard
        c_fold_u = np.array([pooled_unclipped_utility(y[lab == k], c_oof[lab == k]) for k in range(5)])
        if float(np.min(c_fold_u - anc_fold_u)) < float(conf.min_all_folds_tol_u):
            continue

        c_err_sq = (y - c_oof) ** 2
        c_boot_rmse = np.sqrt(np.maximum(np.mean(c_err_sq[boot_idx], axis=1), 1e-30))
        c_boot_u = np.log(y_boot_std / c_boot_rmse)
        delta_boot = c_boot_u - anc_boot_u

        win_prob = float(np.mean(delta_boot > 0.0))
        lcb10 = float(np.quantile(delta_boot, 0.10))
        if win_prob < float(conf.min_bootstrap_win_prob) or lcb10 < float(conf.min_pooled_lcb_u):
            continue
        if u_pool < anc_pooled_u + float(conf.min_pooled_lcb_u):
            continue

        # Expected family-weighted pairwise wins against field margin ladder
        exp_wins = float(fam_weight * np.mean(delta_boot[:, None] > opp_margins[None, :]))
        if exp_wins > best_exp_wins + 1e-12 or (
            abs(exp_wins - best_exp_wins) <= 1e-12 and u_pool > cand_u_map[best_name]
        ):
            best_name = nm
            best_exp_wins = exp_wins
            best_lcb10 = lcb10
            best_win_prob = win_prob

    # Exact `eta = 0` anchor retention when no candidate clears the certification bar!
    sel_oof = anc_oof.copy() if best_name == anchor_name else np.array(candidate_oof[best_name], dtype=np.float64, copy=True)
    sel_qry = anc_qry.copy() if best_name == anchor_name else np.array(candidate_query[best_name], dtype=np.float64, copy=True)
    sel_u = cand_u_map[best_name]

    return EdgeContextDecision(
        selected_name=best_name,
        anchor_name=anchor_name,
        switched_from_anchor=(best_name != anchor_name),
        family_weight=fam_weight,
        pooled_u_selected=sel_u,
        pooled_u_anchor=anc_pooled_u,
        pooled_delta_u=sel_u - anc_pooled_u,
        bootstrap_lcb10_delta_u=best_lcb10,
        bootstrap_win_prob_vs_anchor=best_win_prob,
        expected_family_weighted_wins=best_exp_wins,
        pred_oof=sel_oof,
        pred_query=sel_qry,
        candidate_pooled_u=cand_u_map,
    )


def enforce_cohort_family_mean_floor(
    decisions: Sequence[EdgeContextDecision],
    family_ids: Sequence[str],
    max_drop_u: float = 0.01,
) -> List[EdgeContextDecision]:
    """Enforce co-primary `Delta family_mean_u >= -max_drop_u` across a multi-context board."""
    if len(decisions) != len(family_ids):
        raise ValueError("decisions and family_ids must have identical length")
    out = list(decisions)
    fam_groups: Dict[str, List[int]] = {}
    for idx, fid in enumerate(family_ids):
        fam_groups.setdefault(str(fid), []).append(idx)

    def _fam_mean_delta(cur: Sequence[EdgeContextDecision]) -> float:
        vals = []
        for idxs in fam_groups.values():
            vals.append(float(np.mean([cur[i].pooled_delta_u for i in idxs])))
        return float(np.mean(vals)) if vals else 0.0

    # Revert lowest-delta switches until cohort family-mean utility satisfies floor
    while _fam_mean_delta(out) < -float(max_drop_u):
        switched = [i for i, d in enumerate(out) if d.switched_from_anchor]
        if not switched:
            break
        worst_i = min(switched, key=lambda i: out[i].pooled_delta_u)
        d_old = out[worst_i]
        out[worst_i] = dataclasses.replace(
            d_old,
            selected_name=d_old.anchor_name,
            switched_from_anchor=False,
            pooled_u_selected=d_old.pooled_u_anchor,
            pooled_delta_u=0.0,
        )
    return out


def fit_raw_y_classic_floor_oof(
    x_search: np.ndarray,
    y_search: np.ndarray,
    x_query: np.ndarray,
    lab_folds: np.ndarray,
    seed: int = 462450,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Fit **raw-`y` unwarped** classic floor `{raw_ridge, raw_hgb, raw_rf, raw_et}` on `lab_folds`.

    Crucially avoids `_yj` (`Yeo-Johnson`) target warps (`cl_h458` `06:13Z` challenge: `h450_cb_d4_yj`
    carries inverse-`yj`, whereas raw-`y` ridge extrapolates linearly and raw-`y` trees stay strictly
    inside `[min(y_search), max(y_search)]`).
    """
    x_s = np.ascontiguousarray(x_search, dtype=np.float64)
    y_s = np.ascontiguousarray(y_search, dtype=np.float64).reshape(-1)
    x_q = np.ascontiguousarray(x_query, dtype=np.float64)
    lab = np.asarray(lab_folds, dtype=np.int64).reshape(-1)
    n_s = y_s.shape[0]
    n_q = x_q.shape[0]

    floor_names = ("raw_ridge", "raw_hgb", "raw_rf", "raw_et")
    oof_map = {nm: np.zeros(n_s, dtype=np.float64) for nm in floor_names}
    q_acc = {nm: np.zeros(n_q, dtype=np.float64) for nm in floor_names}

    for k in range(5):
        va = lab == k
        tr = ~va
        x_tr, y_tr, x_va = x_s[tr], y_s[tr], x_s[va]
        both = np.vstack([x_va, x_q])
        n_va = int(np.sum(va))

        # 1. Raw-y GCV linear ridge (scale-safe & NaN-safe strictly from train fold)
        med = np.nanmedian(x_tr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        x_tr_imp = np.where(np.isfinite(x_tr), x_tr, med)
        both_imp = np.where(np.isfinite(both), both, med)
        q25, q75 = np.percentile(x_tr_imp, [25.0, 75.0], axis=0)
        sc_raw = np.where((q75 - q25) > 1e-9, q75 - q25, np.std(x_tr_imp, axis=0))
        sc = np.where(np.isfinite(sc_raw) & (sc_raw > 1e-9), sc_raw, 1.0)
        z_tr = np.clip((x_tr_imp - med) / sc, -1e6, 1e6)
        z_both = np.clip((both_imp - med) / sc, -1e6, 1e6)
        y_mu = float(np.mean(y_tr))
        y_sd = max(float(np.std(y_tr)), 1e-12)
        yc_norm = (y_tr - y_mu) / y_sd
        u_svd, s_svd, vt_svd = sla.svd(z_tr, full_matrices=False, check_finite=False)
        uty = u_svd.T @ yc_norm
        best_gcv, best_lam = float("inf"), 10.0
        for lam in (1e-2, 0.1, 1.0, 10.0, 100.0, 500.0):
            filt = (s_svd * s_svd) / (s_svd * s_svd + lam)
            res = yc_norm - u_svd @ (filt * uty)
            denom = max(1.0 - float(np.sum(filt)) / float(x_tr.shape[0]), 1e-3) ** 2
            gcv = float(np.mean(res * res)) / denom
            if gcv < best_gcv:
                best_gcv, best_lam = gcv, lam
        w_r = vt_svd.T @ ((s_svd / (s_svd * s_svd + best_lam)) * uty)
        p_ridge = y_mu + y_sd * (z_both @ w_r)
        oof_map["raw_ridge"][va] = p_ridge[:n_va]
        q_acc["raw_ridge"] += p_ridge[n_va:] / 5.0

        # Train-fold monotone linear scaling for float32-backed sklearn trees (`|X| > 3.4e38` guard;
        # identity `scale == 1.0` whenever `max(|x_tr|) <= 1e20`, preserving exact splits everywhere else)
        x_tr_fin = np.where(np.isfinite(x_tr), np.abs(x_tr), 0.0)
        col_max = np.max(x_tr_fin, axis=0)
        tree_x_scale = np.where(col_max > 1e20, col_max / 1e15, 1.0)

        def _prep_tree_x(arr: np.ndarray, scale: np.ndarray) -> np.ndarray:
            scaled = arr / scale
            out = np.where(np.isnan(arr), np.nan, np.clip(scaled, -1e30, 1e30))
            return np.ascontiguousarray(out, dtype=np.float64)

        x_tr_tree = _prep_tree_x(x_tr, tree_x_scale)
        both_tree = _prep_tree_x(both, tree_x_scale)
        y_abs_max = float(np.max(np.abs(y_tr))) if y_tr.size else 0.0
        y_tree_scale = (y_abs_max / 1e15) if y_abs_max > 1e20 else 1.0
        y_tr_tree = y_tr / y_tree_scale

        # 2. Raw-y HistGradientBoosting (GBDT floor, strictly range-bounded)
        hgb = HistGradientBoostingRegressor(
            max_iter=80, max_depth=5, learning_rate=0.06, min_samples_leaf=max(4, x_tr.shape[0] // 25), random_state=int(seed) + k
        )
        hgb.fit(x_tr_tree, y_tr_tree)
        p_hgb = hgb.predict(both_tree) * y_tree_scale
        oof_map["raw_hgb"][va] = p_hgb[:n_va]
        q_acc["raw_hgb"] += p_hgb[n_va:] / 5.0

        # 3. Raw-y RandomForest (4286 floor)
        rf = RandomForestRegressor(n_estimators=64, min_samples_leaf=2, n_jobs=1, random_state=int(seed) + 10 + k)
        rf.fit(x_tr_tree, y_tr_tree)
        p_rf = rf.predict(both_tree) * y_tree_scale
        oof_map["raw_rf"][va] = p_rf[:n_va]
        q_acc["raw_rf"] += p_rf[n_va:] / 5.0

        # 4. Raw-y ExtraTrees
        et = ExtraTreesRegressor(n_estimators=64, min_samples_leaf=2, n_jobs=1, random_state=int(seed) + 20 + k)
        et.fit(x_tr_tree, y_tr_tree)
        p_et = et.predict(both_tree) * y_tree_scale
        oof_map["raw_et"][va] = p_et[:n_va]
        q_acc["raw_et"] += p_et[n_va:] / 5.0

    return oof_map, q_acc


def bag_predictions_e8(member_predictions: Sequence[np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Average `M` (`8` in production `h462.APEX_PRISM_E8`) member predictions in prediction space."""
    if not member_predictions:
        raise ValueError("member_predictions must be non-empty")
    mat = np.column_stack([np.asarray(p, dtype=np.float64).reshape(-1) for p in member_predictions])
    bag_pred = np.mean(mat, axis=1)
    member_disp = float(np.mean(np.var(mat, axis=1)))
    return bag_pred, {
        "entrant_id": ENTRANT_ID_E8,
        "mean_member_variance_reduction": member_disp,
        "n_members": int(mat.shape[1]),
    }


def bag_predictions_precision_weighted_oof(
    y_search: np.ndarray,
    member_oof_predictions: Sequence[np.ndarray],
    member_query_predictions: Sequence[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Zero-hyperparameter OOF inverse-MSE precision-weighted procedure seed bagging (`M6-PRECISION-AVG`).

    Resolves the convexity of `u(MSE) = -0.5 * log(MSE / b^2)` (`d^2 u / d(MSE)^2 = +0.5 / MSE^2 > 0`):
    - When all seeds have comparable OOF MSE (`MSE_k ≈ MSE_j`), `w_k ≈ 1 / M`, recovering equal-weight
      prediction bagging (`0.5 * (p0 + p1)` -> `+8.5..+21.9` single-placement Elo on guarded stacks).
    - When one seed discovers a near-exact law (`MSE_0 << MSE_1`), `w_0 -> 1.0`, preventing `0.5 * (p0 + p1)`
      from diluting an exact `+27 u` symbolic formula with a noisy seed (`-0.0687 u` -> `+0.1263 u` on `P201B-80`).
    """
    if not member_oof_predictions or len(member_oof_predictions) != len(member_query_predictions):
        raise ValueError("member_oof_predictions and member_query_predictions must be non-empty and equal length")
    y_s = np.asarray(y_search, dtype=np.float64).reshape(-1)
    y_var_floor = max(1e-30 * float(np.var(y_s)), 1e-300)
    prec = []
    oof_mses = []
    for q_k in member_oof_predictions:
        qk = np.asarray(q_k, dtype=np.float64).reshape(-1)
        if qk.shape != y_s.shape or not np.all(np.isfinite(qk)):
            oof_mses.append(float("inf"))
            prec.append(0.0)
        else:
            mse_k = max(float(np.mean((y_s - qk) ** 2)), y_var_floor)
            oof_mses.append(mse_k)
            prec.append(1.0 / mse_k)
    prec_arr = np.asarray(prec, dtype=np.float64)
    total_prec = float(np.sum(prec_arr))
    if total_prec <= 0.0 or not math.isfinite(total_prec):
        weights = np.full(len(member_query_predictions), 1.0 / len(member_query_predictions), dtype=np.float64)
    else:
        weights = prec_arr / total_prec
    q_mat = np.column_stack([np.asarray(p, dtype=np.float64).reshape(-1) for p in member_query_predictions])
    bag_pred = q_mat @ weights
    return bag_pred, {
        "entrant_id": ENTRANT_ID_E8,
        "max_weight": float(np.max(weights)),
        "n_members": int(q_mat.shape[1]),
        "oof_mses": [float(m) for m in oof_mses],
        "weights": [float(w) for w in weights],
    }


def write_compatible_block_npz(
    out_dir: str | Path,
    cid: str,
    oof_cols: Mapping[str, np.ndarray],
    bag_cols: Mapping[str, np.ndarray],
    ref_cols: Optional[Mapping[str, np.ndarray]] = None,
    closure_sha256: str = "",
) -> Path:
    """Write `{cid}.npz` (`q_*`, `bag_*`, `ref_*`) compatible with `cl_mtc` & `cl_h458` harnesses."""
    assert_path_not_quarantined(out_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {}
    for k, v in oof_cols.items():
        payload[f"q_{k}"] = np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
    for k, v in bag_cols.items():
        payload[f"bag_{k}"] = np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
    if ref_cols:
        for k, v in ref_cols.items():
            payload[f"ref_{k}"] = np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
        for k, v in bag_cols.items():
            if f"ref_{k}" not in payload:
                payload[f"ref_{k}"] = np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
    else:
        for k, v in bag_cols.items():
            payload[f"ref_{k}"] = np.ascontiguousarray(v, dtype=np.float64).reshape(-1)

    npz_file = out_p / f"{cid}.npz"
    np.savez_compressed(npz_file, **payload)
    sidecar = {
        "cid": str(cid),
        "closure_sha256": str(closure_sha256),
        "columns": sorted(oof_cols.keys()),
        "entrant_id": ENTRANT_ID_E8,
        "npz_sha256": hashlib.sha256(npz_file.read_bytes()).hexdigest(),
        "schema": "jope.h462.block_sidecar/v1",
    }
    (out_p / f"{cid}.sidecar.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return npz_file
