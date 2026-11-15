"""Gate G1 Fixed-Bank & Multi-Regime Equilibrium Replay Benchmark (`replay_g1_s9_pool.py`).

Implements the CPU-only `Gate G1` ladder agreed across `cl_mtc`, `cl_fwriter`, and `cl_h458`:
  - `L0`: Equal-weight / simplex average over incumbent anchor columns (`H416` baseline).
  - `L1`: Equal-compute cheap static control (`C_static`).
  - `L2`: Sequential greedy forward residual boosting (freeze prior weights, fit next residual).
  - `L3`: Simultaneous dense Ridge over all candidate bank columns.
  - `L4`: `APEX-EQ` Joint Covariance Equilibrium (QR span-novelty guard + robust IRLS noise +
          Foster-George multiplicity threshold + Empirical Bayes weight re-estimation +
          5-fold LFO `eta`-damping + SEARCH target-range safety clamp).

Board & Quarantine Guards:
  - Refuses any path containing `confirm`, `holdout`, `sealed`, `dl-val`, or `probe_ya`.
  - Enforces `cl_mtc`'s 04:07Z ruling: design runs on `p201e` (or `synthetic`); reading
    `p201b_descriptive` requires `--s9-selection-receipt` to exist on disk first.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg as sla

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from research.flywheel_selfplay.h460_apex_eq.eq_core import (
        EquilibriumConfig,
        EquilibriumState,
        calibrate_placebo_c0,
    )
    from research.flywheel_selfplay.h460_apex_eq.fit_closure import FitScopeV2, learn_whole_procedure
else:
    from .eq_core import EquilibriumConfig, EquilibriumState, calibrate_placebo_c0
    from .fit_closure import FitScopeV2, learn_whole_procedure

FORBIDDEN_PATH_TOKENS = (
    "confirm",
    "holdout",
    "sealed",
    "dl-val",
    "dl_val",
    "fmgap_b",
    "fmgap_c",
    "probe_ya",
)


def assert_clean_train_path(path: Path) -> None:
    """Refuse any sealed, confirm, or holdout evaluation artifact path."""
    low = str(path).lower()
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok in low:
            raise ValueError(f"Quarantine refusal: forbidden token '{tok}' in path {path}")


def make_deterministic_folds(n_rows: int, n_folds: int = 5, seed: int = 460101) -> List[np.ndarray]:
    """Deterministic row partition for inner K-fold LFO damping selection."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(int(n_rows))
    return [np.sort(chunk) for chunk in np.array_split(perm, max(2, int(n_folds)))]


def _fit_sequential_greedy_l2(
    y_train: np.ndarray,
    inc_train: np.ndarray,
    inc_val: np.ndarray,
    bank_train: np.ndarray,
    bank_val: np.ndarray,
    max_steps: int = 8,
    shrinkage: float = 0.7,
    ridge_pen: float = 5.0,
) -> np.ndarray:
    """`L2`: Serial greedy residual boosting with frozen previous stage coefficients."""
    res = y_train - inc_train
    pred_val = inc_val.copy()
    used: set[int] = set()
    n_cands = bank_train.shape[1]

    for _ in range(min(int(max_steps), n_cands)):
        best_idx = -1
        best_corr = 0.0
        for j in range(n_cands):
            if j in used:
                continue
            col = bank_train[:, j]
            denom = float(np.dot(col, col)) + float(ridge_pen)
            score = (float(np.dot(col, res)) ** 2) / max(denom, 1e-12)
            if score > best_corr:
                best_corr = score
                best_idx = j
        if best_idx < 0 or best_corr <= 1e-12:
            break
        used.add(best_idx)
        col_b = bank_train[:, best_idx]
        coef = float(shrinkage) * float(np.dot(col_b, res)) / (float(np.dot(col_b, col_b)) + float(ridge_pen))
        res = res - coef * col_b
        pred_val = pred_val + coef * bank_val[:, best_idx]
    return pred_val


def _fit_simultaneous_ridge_l3(
    y_train: np.ndarray,
    inc_train: np.ndarray,
    inc_val: np.ndarray,
    bank_train: np.ndarray,
    bank_val: np.ndarray,
    folds: Sequence[np.ndarray],
    lam_grid: Sequence[float] = (0.5, 2.0, 10.0, 50.0, 200.0),
) -> np.ndarray:
    """`L3`: Simultaneous dense L2 Ridge over all candidate columns with inner CV `lambda`."""
    res = y_train - inc_train
    n_rows, n_cands = bank_train.shape
    if n_cands == 0:
        return inc_val.copy()

    best_lam = float(lam_grid[0])
    best_mse = float("inf")
    all_idx = np.arange(n_rows)
    for lam in lam_grid:
        oof = np.zeros(n_rows, dtype=np.float64)
        for f_idx in folds:
            tr_mask = np.ones(n_rows, dtype=bool)
            tr_mask[f_idx] = False
            tr_i = all_idx[tr_mask]
            b_tr = bank_train[tr_i]
            gram = b_tr.T @ b_tr + float(lam) * np.eye(n_cands, dtype=np.float64)
            w_hat = sla.solve(gram, b_tr.T @ res[tr_i], assume_a="pos", check_finite=False)
            oof[f_idx] = bank_train[f_idx] @ w_hat
        mse = float(np.mean((res - oof) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_lam = float(lam)

    gram_full = bank_train.T @ bank_train + best_lam * np.eye(n_cands, dtype=np.float64)
    w_full = sla.solve(gram_full, bank_train.T @ res, assume_a="pos", check_finite=False)
    return inc_val + (bank_val @ w_full)


def _fit_apex_eq_l4(
    y_train: np.ndarray,
    inc_train: np.ndarray,
    inc_val: np.ndarray,
    bank_train: np.ndarray,
    bank_val: np.ndarray,
    cand_names: Sequence[str],
    folds: Sequence[np.ndarray],
    cfg: Optional[EquilibriumConfig] = None,
    calibrate_placebo: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """`L4`: `APEX-EQ` Joint Covariance Equilibrium over a fixed candidate bank."""
    eq_cfg = cfg or EquilibriumConfig()
    n_tr = int(y_train.shape[0])
    n_va = int(inc_val.shape[0])
    res_var = max(float(np.var(y_train - inc_train)), 1e-6)
    state = EquilibriumState(
        y_train=y_train,
        incumbent_train=inc_train,
        incumbent_val=inc_val,
        anchor_h_train=np.ones((n_tr, 1), dtype=np.float64),
        anchor_h_val=np.ones((n_va, 1), dtype=np.float64),
        tau_h2=max(1.0, 4.0 * res_var),
        cfg=eq_cfg,
    )
    state.update_robust_noise_irls()

    # Standardize candidate bank strictly from training rows
    mu = np.mean(bank_train, axis=0)
    sd = np.maximum(np.std(bank_train, axis=0), 1e-9)
    z_tr = (bank_train - mu) / sd
    z_va = (bank_val - mu) / sd

    c0_cal = (
        calibrate_placebo_c0(state, z_tr, n_trials=30, seed=460201)
        if calibrate_placebo and z_tr.shape[1] > 0
        else eq_cfg.foster_george_c0
    )

    admitted: List[str] = []
    reweighted_in_span: int = 0
    available = np.ones(z_tr.shape[1], dtype=bool)

    for _ in range(eq_cfg.max_active_groups):
        avail_idx = np.flatnonzero(available)
        if avail_idx.size == 0:
            break
        sub_names = [cand_names[int(i)] for i in avail_idx]
        dec = state.screen_dictionary(
            z_tr[:, avail_idx],
            sub_names,
            cand_val_matrix=z_va[:, avail_idx],
            c0_override=c0_cal,
        )
        orig_idx = int(avail_idx[dec.candidate_index]) if dec.candidate_index >= 0 else -1
        if dec.action == "IN_SPAN_REWEIGHT" and orig_idx >= 0:
            available[orig_idx] = False
            reweighted_in_span += 1
            state.reestimate_group_weights_empirical_bayes(n_passes=1)
            continue
        if dec.action != "ADD" or orig_idx < 0:
            break
        available[orig_idx] = False
        state.add_group(
            name=str(cand_names[orig_idx]),
            phi_train=z_tr[:, orig_idx],
            phi_val=z_va[:, orig_idx],
            weight=dec.optimal_weight,
            family_tag="bank_col",
        )
        admitted.append(str(cand_names[orig_idx]))
        state.reestimate_group_weights_empirical_bayes(n_passes=2)
        state.update_robust_noise_irls()

    eta_star, eta_grid_mse = state.select_damping_eta(folds)
    pred_val = state.predict_val(eta=eta_star, apply_range_clamp=True)
    meta = {
        "admitted_columns": admitted,
        "c0_calibrated": float(c0_cal),
        "eta_grid_lfo_mse": eta_grid_mse,
        "eta_selected": float(eta_star),
        "marginal_log_evidence": state.marginal_log_evidence(),
        "n_active_groups": len(state.groups),
        "reweighted_in_span": reweighted_in_span,
        "state_digest": state.state_digest(),
    }
    return pred_val, meta


def solve_fixed_bank_l0_to_l4(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    incumbent_train: np.ndarray,
    incumbent_val: np.ndarray,
    bank_train: np.ndarray,
    bank_val: np.ndarray,
    cand_names: Sequence[str],
    static_ctrl_val: Optional[np.ndarray] = None,
    cfg: Optional[EquilibriumConfig] = None,
) -> Dict[str, object]:
    """Run `L0, L1, L2, L3, L4` strictly inside `learn_whole_procedure` (`0` `y_val` access)."""

    def _learner(
        _x_tr: np.ndarray,
        y_tr: np.ndarray,
        _x_va: np.ndarray,
        scope: FitScopeV2,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        folds = make_deterministic_folds(y_tr.shape[0], n_folds=5)
        l0_val = incumbent_val.copy()
        l1_val = incumbent_val.copy() if static_ctrl_val is None else np.asarray(static_ctrl_val, dtype=np.float64)
        l2_val = _fit_sequential_greedy_l2(y_tr, incumbent_train, incumbent_val, bank_train, bank_val)
        l3_val = _fit_simultaneous_ridge_l3(y_tr, incumbent_train, incumbent_val, bank_train, bank_val, folds)
        l4_val, l4_meta = _fit_apex_eq_l4(
            y_tr, incumbent_train, incumbent_val, bank_train, bank_val, cand_names, folds, cfg=cfg
        )
        return l4_val, {
            "fit_scope_sha256": scope.scope_sha256,
            "l4_meta": l4_meta,
            "predictions": {
                "L0_incumbent": l0_val,
                "L1_c_static": l1_val,
                "L2_serial_boost": l2_val,
                "L3_dense_ridge": l3_val,
                "L4_apex_eq": l4_val,
            },
            "state_digest": l4_meta["state_digest"],
        }

    pred_l4, scope, bundle = learn_whole_procedure(x_train, y_train, x_val, _learner, fold_id="g1_fixed_bank")
    bundle["primary_prediction"] = pred_l4
    bundle["scope"] = dataclasses.asdict(scope) if hasattr(scope, "__dataclass_fields__") else {}
    return bundle


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", choices=("synthetic", "p201e", "p201b_descriptive"), default="synthetic")
    ap.add_argument("--bank-dir", type=Path, default=None, help="Directory of `.npz` fixed-bank contexts")
    ap.add_argument("--s9-selection-receipt", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.board == "p201b_descriptive":
        if args.s9_selection_receipt is None or not args.s9_selection_receipt.is_file():
            raise SystemExit(
                "REFUSED: --board p201b_descriptive requires an existing --s9-selection-receipt "
                "(per cl_mtc 04:07Z ruling: G1 never touches P201B-80 before S9 selects)."
            )
        assert_clean_train_path(args.s9_selection_receipt)

    if args.bank_dir is not None:
        assert_clean_train_path(args.bank_dir)

    rng = np.random.default_rng(460301)
    x_tr = rng.normal(size=(180, 6))
    x_va = rng.normal(size=(80, 6))
    true_fn_tr = 1.4 * (x_tr[:, 0] > 0.2).astype(float) + 0.8 * np.sin(x_tr[:, 1]) - 0.6 * x_tr[:, 2]
    true_fn_va = 1.4 * (x_va[:, 0] > 0.2).astype(float) + 0.8 * np.sin(x_va[:, 1]) - 0.6 * x_va[:, 2]
    y_tr = true_fn_tr + 0.15 * rng.normal(size=180)
    inc_tr = 0.5 * np.sin(x_tr[:, 1])
    inc_va = 0.5 * np.sin(x_va[:, 1])
    noise_tr = rng.normal(size=(180, 20))
    noise_va = rng.normal(size=(80, 20))
    bank_tr = np.column_stack([(x_tr[:, 0] > 0.2).astype(float), np.sin(x_tr[:, 1]), x_tr[:, 2], noise_tr])
    bank_va = np.column_stack([(x_va[:, 0] > 0.2).astype(float), np.sin(x_va[:, 1]), x_va[:, 2], noise_va])
    names = ["step_c0", "sin_c1", "lin_c2"] + [f"noise_{i}" for i in range(20)]

    res = solve_fixed_bank_l0_to_l4(x_tr, y_tr, x_va, inc_tr, inc_va, bank_tr, bank_va, names)
    preds = res["predictions"]
    mse_by_arm = {k: round(float(np.mean((true_fn_va - v) ** 2)), 6) for k, v in preds.items()}
    summary = {"board": args.board, "mse_vs_noiseless_truth": mse_by_arm, "l4_meta": res["l4_meta"]}
    if args.out_json is not None:
        assert_clean_train_path(args.out_json)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
