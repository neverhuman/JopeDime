"""H295 (h295_label_free_validity_guard) core implementation:
- Step 1: Bit-exact reproduction of H287 ENS_VOTE2 DL-Train inner-CV median (+0.059931533238705134 / 0.059932)
  from published artifacts before any new fit.
- Step 2 (`fit_and_lock`): 4-fold family-grouped outer CV on DL-Train (205 families, 1,219 contexts;
  seeds 295101-295103, shuffled-label null seeds 295201-295203). Fits the label-free hybrid
  LogisticRegression + LGBMClassifier validity guard on candidate features only inside each outer training
  fold `tr` (with `va` utilities and validity labels masked out so zero outer-fold scoring can occur before lock),
  chooses `tau` per outer fold via 3-fold family-grouped nested inner CV on `tr`, saves all outer-fold and
  full-DL-Train guard models, and writes `fit_lock.json`.
- Step 3 (`outer_score`): Disables `.fit()`, verifies `fit_lock.json` and all model file SHA256 digests,
  scores the held-out outer folds `va` across seeds 295101-295103 (and null seeds 295201-295203), and
  evaluates the strict H295 decision rule.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import analysis_h295 as stat
import h265_core as core
import veto

H295_SEEDS: Tuple[int, int, int] = (295101, 295102, 295103)
H295_NULL_SEEDS: Tuple[int, int, int] = (295201, 295202, 295203)
TAU_VAL_GRID: Tuple[float, ...] = (0.030, 0.035, 0.040, 0.045, 0.050, 0.060)
TAU_HARM_GRID: Tuple[float, ...] = (0.15, 0.18, 0.20, 0.22, 0.25, 0.30)
SLOT0_ESCAPE_FLOOR: float = 0.055
SLOT0_ALT_MAX_PINV: float = 0.015
HARM_LOSS_THRESH: float = 0.01

ARM_NAMES: Tuple[str, ...] = (
    "SLOT0",
    "ENS_VOTE2",
    "ENS_VOTE2+VALGUARD",
    "ENS_VOTE2+VALGUARD+HARMVETO",
    "SHUFFLED_NULL",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verify_h287_ens_vote2_regression(
    oof_cache_path: Path,
    h287_panel_path: Path,
    h287_metrics_path: Path,
) -> Dict[str, Any]:
    """Reproduce H287 ENS_VOTE2 DL-Train inner-CV median and per-seed cells bit-exact from published artifacts."""
    z = np.load(oof_cache_path, allow_pickle=False)
    fams = [str(f) for f in z["fams"]]
    u_k = np.asarray(z["utility_k"], dtype=np.float64)
    gate_k = np.asarray(z["gate_k"], dtype=bool)
    u_pbag = np.asarray(z["u_pbag"], dtype=np.int32)

    n_ctx = len(fams)
    fam_ids = sorted(set(fams))
    fams_arr = np.asarray(fams)
    fam_masks = [fams_arr == f for f in fam_ids]

    panel = np.load(h287_panel_path, allow_pickle=False)["panel"]
    with h287_metrics_path.open("r") as f:
        m287 = json.load(f)
    pub_ens2 = m287["inner_cv_dl_train"]["contrasts"]["ENS_VOTE2_minus_slot0"]
    pub_inv = m287["inner_cv_dl_train"]["invalid_by_seed"]["ENS_VOTE2"]
    pub_chg = m287["inner_cv_dl_train"]["changed_by_seed"]["ENS_VOTE2"]

    per_seed_gains_exact: List[float] = []
    per_seed_gains_round6: List[float] = []
    per_seed_harmed: List[int] = []
    per_seed_invalid: List[int] = []
    per_seed_changed: List[int] = []
    max_abs_diffs: List[float] = []

    for r in range(3):
        picks = np.zeros(n_ctx, dtype=np.int32)
        for i in range(n_ctx):
            nz = [int(u_pbag[r, m, i]) for m in range(3) if int(u_pbag[r, m, i]) != 0]
            for cand in sorted(set(nz)):
                if nz.count(cand) >= 2:
                    picks[i] = int(cand)
                    break
        u_vec = u_k[np.arange(n_ctx), picks].astype(np.float64)
        u0_vec = u_k[:, 0].astype(np.float64)
        ref_vec = np.asarray(panel[r, :, 5], dtype=np.float64)  # index 5 is ENS_VOTE2 in H287 panel
        max_diff = float(np.max(np.abs(u_vec - ref_vec)))
        max_abs_diffs.append(max_diff)

        fam_diff = np.array([float(u_vec[m].mean() - u0_vec[m].mean()) for m in fam_masks], dtype=np.float64)
        gain = float(np.mean(fam_diff))
        per_seed_gains_exact.append(gain)
        per_seed_gains_round6.append(round(gain, 6))
        per_seed_harmed.append(int(np.sum(fam_diff < -0.01 - 1e-12)))
        per_seed_invalid.append(int(np.sum(~gate_k[np.arange(n_ctx), picks])))
        per_seed_changed.append(int(np.sum(picks != 0)))

    med_gain_exact = float(np.median(per_seed_gains_exact))
    med_gain_round6 = round(med_gain_exact, 6)
    expected_med_round6 = float(pub_ens2["median_gain"])

    bit_exact = bool(
        max(max_abs_diffs) == 0.0
        and med_gain_round6 == expected_med_round6
        and per_seed_gains_round6 == list(pub_ens2["per_seed_gain"])
        and per_seed_harmed == list(pub_ens2["harmed_001_by_seed"])
        and per_seed_invalid == list(pub_inv)
        and per_seed_changed == list(pub_chg)
    )
    if not bit_exact:
        raise RuntimeError(
            f"H287 ENS_VOTE2 regression failed: max_abs_diffs={max_abs_diffs}, "
            f"med_gain={med_gain_round6} vs {expected_med_round6}, "
            f"gains={per_seed_gains_round6}, harmed={per_seed_harmed}, inv={per_seed_invalid}"
        )

    return {
        "cell": "H287_ENS_VOTE2_dl_train_inner_cv",
        "n_families": len(fam_ids),
        "n_contexts": n_ctx,
        "median_gain_exact": med_gain_exact,
        "median_gain_round6": med_gain_round6,
        "published_median_gain_round6": expected_med_round6,
        "per_seed_gain_exact": per_seed_gains_exact,
        "per_seed_gain_round6": per_seed_gains_round6,
        "utility_vector_max_abs_diff_by_seed": max_abs_diffs,
        "harmed_001_by_seed": per_seed_harmed,
        "invalid_by_seed": per_seed_invalid,
        "changed_by_seed": per_seed_changed,
        "bit_exact": bit_exact,
    }


def make_outer_family_folds(fams: Sequence[str], seed: int, n_folds: int = 4) -> np.ndarray:
    """Assign each context to outer fold 1..n_folds by deterministic family permutation."""
    fams_str = [str(f) for f in fams]
    unique_fams = sorted(set(fams_str))
    rng = np.random.default_rng(core.stream_seed(int(seed), "h295-outer-folds"))
    perm = list(rng.permutation(unique_fams))
    fam_to_fold = {f: 1 + (idx % n_folds) for idx, f in enumerate(perm)}
    return np.array([fam_to_fold[f] for f in fams_str], dtype=np.int32)


def build_candidate_guard_features(X: np.ndarray) -> np.ndarray:
    """Build 652-dim label-free candidate feature matrix `(n_ctx, 8, 652)` from `X` `(n_ctx, 8, 568)` only:
    - 568 cols: raw H278 top-8 candidate features `X[i, k, :]`
    - 40 cols: `X[i, k, :40] - X[i, 0, :40]` (cost/rank/cosine differences relative to slot 0)
    - 40 cols: `X[i, k, :40] - mean_k(X[i, :, :40])` (within-context slice centering)
    - 4 cols: L2 norm and L-inf max of `sem_z` (`cols 452:476`) and `proj_z` (`cols 476:492`).
    Zero fitness and zero validity labels are used.
    """
    b40 = X[:, :, :40]
    diff0 = b40 - b40[:, 0:1, :]
    diffm = b40 - b40.mean(axis=1, keepdims=True)
    sem_z = X[:, :, 452:476]
    proj_z = X[:, :, 476:492]
    norms = np.stack(
        [
            np.linalg.norm(sem_z, axis=2),
            np.max(np.abs(sem_z), axis=2),
            np.linalg.norm(proj_z, axis=2),
            np.max(np.abs(proj_z), axis=2),
        ],
        axis=2,
    ).astype(np.float32)
    return np.concatenate([X, diff0, diffm, norms], axis=2).astype(np.float32)


def fit_hybrid_valguard(
    X_raw_tr: np.ndarray,
    X_aug_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    """Fit small hybrid L2-LogisticRegression + shallow LGBMClassifier validity guard on candidate features only."""
    gbm = veto.fit_gate(X_raw_tr, y_tr, core.stream_seed(int(seed), "gbm"))
    scaler = StandardScaler()
    X_s = np.nan_to_num(scaler.fit_transform(X_aug_tr), 0.0)
    lr = LogisticRegression(
        C=0.05,
        max_iter=500,
        random_state=core.stream_seed(int(seed), "lr"),
    )
    lr.fit(X_s, y_tr.astype(int))
    return {
        "gbm_booster": gbm.booster_,
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "lr_coef": np.asarray(lr.coef_, dtype=np.float64),
        "lr_intercept": np.asarray(lr.intercept_, dtype=np.float64),
    }


def save_hybrid_valguard(mod: Dict[str, Any], gbm_path: Path, lr_path: Path) -> Tuple[str, str]:
    gbm_path.parent.mkdir(parents=True, exist_ok=True)
    mod["gbm_booster"].save_model(str(gbm_path))
    np.savez(
        lr_path,
        scaler_mean=mod["scaler_mean"],
        scaler_scale=mod["scaler_scale"],
        lr_coef=mod["lr_coef"],
        lr_intercept=mod["lr_intercept"],
    )
    return sha256_file(gbm_path), sha256_file(lr_path)


def load_hybrid_valguard(gbm_path: Path, lr_path: Path) -> Dict[str, Any]:
    booster = lgb.Booster(model_file=str(gbm_path))
    z = np.load(lr_path, allow_pickle=False)
    return {
        "gbm_booster": booster,
        "scaler_mean": np.asarray(z["scaler_mean"], dtype=np.float64),
        "scaler_scale": np.asarray(z["scaler_scale"], dtype=np.float64),
        "lr_coef": np.asarray(z["lr_coef"], dtype=np.float64),
        "lr_intercept": np.asarray(z["lr_intercept"], dtype=np.float64),
    }


def predict_hybrid_valguard(
    mod: Dict[str, Any],
    X_raw_ev: np.ndarray,
    X_aug_ev: np.ndarray,
    n_ev_ctx: int,
    k: int = 8,
) -> np.ndarray:
    """Predict per-candidate invalid probability `0.5 * p_gbm + 0.5 * p_lr` without using sklearn `.fit()`."""
    p_gbm = np.asarray(mod["gbm_booster"].predict(X_raw_ev), dtype=np.float64).reshape(n_ev_ctx, k)
    scale = np.where(mod["scaler_scale"] == 0.0, 1.0, mod["scaler_scale"])
    X_s = np.nan_to_num((X_aug_ev.astype(np.float64) - mod["scaler_mean"]) / scale, 0.0)
    logits = (X_s @ mod["lr_coef"].T + mod["lr_intercept"]).reshape(-1)
    logits_clip = np.clip(logits, -40.0, 40.0)
    p_lr = (1.0 / (1.0 + np.exp(-logits_clip))).reshape(n_ev_ctx, k)
    return 0.5 * p_gbm + 0.5 * p_lr


def get_voted_candidates_for_context(
    u_pbag_ri: np.ndarray,
    u_votes_ri: np.ndarray,
    u_raw_ri: np.ndarray,
) -> Tuple[int, List[int]]:
    """Return `(c0, next_cands)` for a context `i`:
    - `c0`: primary ENS_VOTE2 pick (`cand > 0` with `>= 2` member `u_pbag` votes, else `0`).
    - `next_cands`: remaining member-voted non-zero candidates `cand > 0` (`cand != c0` with `>= 1` member
      `u_pbag` vote), ordered by `(member_vote_count desc, bag_vote_sum desc, mean_raw_score desc, -cand)`.
    """
    nz_pbag = [int(x) for x in u_pbag_ri if int(x) > 0]
    c0 = 0
    for cand in sorted(set(nz_pbag)):
        if nz_pbag.count(cand) >= 2:
            c0 = int(cand)
            break
    cand_set = set(nz_pbag)
    cand_set.discard(c0)
    mean_raw = u_raw_ri.mean(axis=0)
    scored: List[Tuple[Tuple[int, int, float, int], int]] = []
    for c in cand_set:
        pbag_cnt = sum(1 for m in range(3) if int(u_pbag_ri[m]) == c)
        v_sum = sum(int(u_votes_ri[m]) for m in range(3) if int(u_pbag_ri[m]) == c)
        scored.append(((pbag_cnt, v_sum, float(mean_raw[c]), -c), int(c)))
    scored.sort(reverse=True)
    next_cands = [c for _, c in scored]
    return c0, next_cands


def apply_valguard_on_indices(
    indices: np.ndarray,
    r: int,
    u_pbag: np.ndarray,
    u_votes: np.ndarray,
    u_raw: np.ndarray,
    p_inv_mat: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Apply label-free validity guard on top of ENS_VOTE2 for context indices `indices`:
    - If ENS_VOTE2 voted for a non-zero candidate `c0 > 0`:
      if `p_inv[i, c0] <= tau`, keep `c0`; otherwise fall back to the first `alt` in `next_cands`
      with `p_inv[i, alt] <= tau`, else `slot 0` (unless `p_inv[i, 0] > p_inv[i, c0]`, in which case
      do not switch to a strictly riskier `slot 0`).
    - If ENS_VOTE2 stayed at `slot 0` (`c0 == 0`):
      if `p_inv[i, 0] > max(tau, SLOT0_ESCAPE_FLOOR)` (slot 0 itself is flagged as high invalid risk),
      fall back to the first `alt` in `next_cands` with `p_inv[i, alt] <= min(tau, SLOT0_ALT_MAX_PINV)`,
      else `slot 0`.
    """
    picks = np.zeros(len(indices), dtype=np.int32)
    for idx_pos, i in enumerate(indices):
        c0, next_cands = get_voted_candidates_for_context(
            u_pbag[r, :, i], u_votes[r, :, i], u_raw[r, :, i]
        )
        if c0 > 0:
            if p_inv_mat[i, c0] <= tau:
                picks[idx_pos] = int(c0)
            else:
                chosen = None
                for alt in next_cands:
                    if p_inv_mat[i, alt] <= tau:
                        chosen = int(alt)
                        break
                if chosen is not None:
                    picks[idx_pos] = chosen
                elif p_inv_mat[i, 0] > p_inv_mat[i, c0]:
                    picks[idx_pos] = int(c0)
                else:
                    picks[idx_pos] = 0
        else:
            if p_inv_mat[i, 0] > max(float(tau), SLOT0_ESCAPE_FLOOR):
                chosen = 0
                for alt in next_cands:
                    if p_inv_mat[i, alt] <= min(float(tau), SLOT0_ALT_MAX_PINV):
                        chosen = int(alt)
                        break
                picks[idx_pos] = chosen
            else:
                picks[idx_pos] = 0
    return picks


def build_harm_features_for_picks(
    X_all: np.ndarray,
    u_raw_r: np.ndarray,
    u_bag_r: np.ndarray,
    u_pinv_r: np.ndarray,
    u_pbag_r: np.ndarray,
    u_jref_r: np.ndarray,
    cand_picks: np.ndarray,
) -> np.ndarray:
    """Build 636-dim label-free context feature matrix for secondary `ENS_VOTE2+VALGUARD+HARMVETO` arm."""
    n_ctx = int(X_all.shape[0])
    margins = np.array([0.02, 0.10, 0.10], dtype=np.float64)
    m_marg = u_raw_r - u_raw_r[:, :, 0:1]
    b_marg = u_bag_r - u_bag_r[:, :, :, 0:1]
    b_marg_24 = b_marg.reshape(24, n_ctx, 8)
    b_votes = np.stack(
        [(b_marg[m] > margins[m]).sum(axis=0) for m in range(3)],
        axis=0,
    ).astype(np.float32)
    best_alt_raw = u_raw_r[:, :, 1:].max(axis=2)
    slot0_marg = u_raw_r[:, :, 0] - best_alt_raw

    rows: List[np.ndarray] = []
    for i in range(n_ctx):
        j = int(cand_picks[i]) if int(cand_picks[i]) > 0 else 1
        mm = m_marg[:, i, j]
        bm = b_marg_24[:, i, j]
        bv = b_votes[:, i, j]
        pi = u_pinv_r[:, i, j]
        pi0 = u_pinv_r[:, i, 0]
        s0m = slot0_marg[:, i]
        pbag_agree = sum(1 for m in range(3) if int(u_pbag_r[m, i]) == j)
        jref_agree = sum(1 for m in range(3) if int(u_jref_r[m, i]) == j)
        n_nz = sum(1 for m in range(3) if int(u_pbag_r[m, i]) != 0)
        extra = np.array(
            [
                *mm,
                float(mm.mean()),
                float(mm.std()),
                float(mm.min()),
                float(mm.max()),
                *bv,
                float(bv.sum()),
                float(bv.min()),
                float(bv.max()),
                float(bm.mean()),
                float(bm.std()),
                float(bm.min()),
                float(np.median(bm)),
                float(bm.max()),
                float((bm > 0.0).mean()),
                float((bm > 0.10).mean()),
                *pi,
                float(pi.mean()),
                float(pi.max()),
                float(pi0.mean()),
                float(pi0.max()),
                *s0m,
                float(s0m.mean()),
                float(s0m.min()),
                float(pbag_agree),
                float(jref_agree),
                float(n_nz),
                *(X_all[i, j, :32] - X_all[i, 0, :32]),
            ],
            dtype=np.float32,
        )
        rows.append(np.concatenate([X_all[i, j, :], extra], axis=0))
    return np.stack(rows, axis=0)


def fit_harm_clf(X_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=60,
        learning_rate=0.05,
        num_leaves=7,
        max_depth=3,
        min_child_samples=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        n_jobs=1,
        random_state=int(seed),
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    clf.fit(X_tr, y_tr.astype(int))
    return clf


def _eval_tr_subset(
    tr_indices: np.ndarray,
    picks_sub: np.ndarray,
    u_k_tr_only: np.ndarray,
    gate_k_tr_only: np.ndarray,
    fams_arr: np.ndarray,
) -> Tuple[float, int, int]:
    """Evaluate family-equal mean gain, harmed families, and invalid count on training indices `tr_indices` ONLY.
    Asserts that no NaN (masked outer-validation `va`) entry is ever accessed.
    """
    u = u_k_tr_only[tr_indices, picks_sub]
    u0 = u_k_tr_only[tr_indices, 0]
    if np.any(np.isnan(u)) or np.any(np.isnan(u0)):
        raise RuntimeError("Isolation violation: outer validation utility accessed before lock!")
    g_raw = gate_k_tr_only[tr_indices, picks_sub]
    if np.any(g_raw < 0):
        raise RuntimeError("Isolation violation: outer validation gate label accessed before lock!")
    diff = u - u0
    f_sub = fams_arr[tr_indices]
    sub_fams = sorted(set(str(x) for x in f_sub))
    fg = np.array([float(np.mean(diff[f_sub == f])) for f in sub_fams], dtype=np.float64)
    return float(np.mean(fg)), int(np.sum(fg < -HARM_LOSS_THRESH - 1e-12)), int(np.sum(g_raw == 0))


def run_fit_and_lock(
    oof_cache_path: Path,
    h287_panel_path: Path,
    h287_metrics_path: Path,
    export_dir: Path,
    capsule_fit_lock_path: Path,
) -> Dict[str, Any]:
    """Stage 1 (`--mode fit_and_lock`):
    1. Verify H287 ENS_VOTE2 DL-Train inner-CV median bit-exact.
    2. For seeds 295101-295103 (and null seeds 295201-295203), assign 4 family-grouped outer folds.
    3. Mask out outer validation fold `va` utilities and validity labels (`NaN` / `-1`) before any inner CV or fit,
       proving `outer_fold_labels_read_before_lock == 0`.
    4. Select `tau_val`, `tau_shuf`, and `tau_harm` strictly by 3-fold nested inner CV inside `tr`.
    5. Fit and save all outer-fold models and full-DL-Train models to `export_dir`, and write `fit_lock.json`.
    """
    t0 = time.time()
    export_dir.mkdir(parents=True, exist_ok=True)

    reg_receipt = verify_h287_ens_vote2_regression(oof_cache_path, h287_panel_path, h287_metrics_path)

    z = np.load(oof_cache_path, allow_pickle=False)
    cids = [str(c) for c in z["cids"]]
    fams = [str(f) for f in z["fams"]]
    fams_arr = np.asarray(fams)
    X_all = np.asarray(z["X"], dtype=np.float32)
    u_k_full = np.asarray(z["utility_k"], dtype=np.float64)
    gate_k_full = np.asarray(z["gate_k"], dtype=bool)
    u_pbag = np.asarray(z["u_pbag"], dtype=np.int32)
    u_jref = np.asarray(z["u_jref"], dtype=np.int32)
    u_votes = np.asarray(z["u_votes"], dtype=np.int32)
    u_raw = np.asarray(z["u_raw"], dtype=np.float64)
    u_bag = np.asarray(z["u_bag"], dtype=np.float64)
    u_pinv = np.asarray(z["u_p_inv"], dtype=np.float64)

    n_ctx = len(fams)
    X_aug = build_candidate_guard_features(X_all)

    ens2_base = np.zeros((3, n_ctx), dtype=np.int32)
    for r in range(3):
        for i in range(n_ctx):
            c0, _ = get_voted_candidates_for_context(u_pbag[r, :, i], u_votes[r, :, i], u_raw[r, :, i])
            ens2_base[r, i] = c0

    outer_fold_models: List[Dict[str, Any]] = []
    full_export_models: List[Dict[str, Any]] = []
    outer_fold_ids_all = np.zeros((3, n_ctx), dtype=np.int32)
    taus_val = np.zeros((3, 4), dtype=np.float64)
    taus_shuf = np.zeros((3, 4), dtype=np.float64)
    taus_harm = np.zeros((3, 4), dtype=np.float64)

    for r in range(3):
        seed = H295_SEEDS[r]
        null_seed = H295_NULL_SEEDS[r]
        fids = make_outer_family_folds(fams, seed, n_folds=4)
        outer_fold_ids_all[r] = fids

        for hold in (1, 2, 3, 4):
            tr = np.where(fids != hold)[0]
            va = np.where(fids == hold)[0]

            # Mask outer validation fold `va` so any accidental read of `va` labels raises RuntimeError
            u_k_tr_only = u_k_full.copy()
            u_k_tr_only[va, :] = np.nan
            gate_k_tr_only = gate_k_full.astype(np.int8).copy()
            gate_k_tr_only[va, :] = -1

            # 3-fold family-grouped nested inner CV inside `tr`
            tr_fams = sorted(set(str(fams_arr[i]) for i in tr))
            rng_ii = np.random.default_rng(core.stream_seed(seed, "ii_folds", hold))
            perm_fams = list(tr_fams)
            rng_ii.shuffle(perm_fams)
            ii_map = {f: (idx % 3) for idx, f in enumerate(perm_fams)}

            ii_pinv = np.zeros((n_ctx, 8), dtype=np.float64)
            ii_shuf = np.zeros((n_ctx, 8), dtype=np.float64)

            for ii_h in range(3):
                itr = np.array([i for i in tr if ii_map[str(fams_arr[i])] != ii_h], dtype=np.int64)
                iva = np.array([i for i in tr if ii_map[str(fams_arr[i])] == ii_h], dtype=np.int64)
                if np.any(gate_k_tr_only[itr] < 0):
                    raise RuntimeError("Isolation violation in inner-inner training labels!")
                y_itr = (gate_k_tr_only[itr] == 0).reshape(-1)
                mod_ii = fit_hybrid_valguard(
                    X_all[itr].reshape(-1, 568),
                    X_aug[itr].reshape(-1, 652),
                    y_itr,
                    core.stream_seed(seed, "ii_valguard", hold, ii_h),
                )
                ii_pinv[iva] = predict_hybrid_valguard(
                    mod_ii,
                    X_all[iva].reshape(-1, 568),
                    X_aug[iva].reshape(-1, 652),
                    len(iva),
                    8,
                )

                rng_s = np.random.default_rng(core.stream_seed(null_seed, "ii_shuf_perm", hold, ii_h))
                y_itr_sh = rng_s.permutation(y_itr)
                mod_ii_sh = fit_hybrid_valguard(
                    X_all[itr].reshape(-1, 568),
                    X_aug[itr].reshape(-1, 652),
                    y_itr_sh,
                    core.stream_seed(null_seed, "ii_shuf_mod", hold, ii_h),
                )
                ii_shuf[iva] = predict_hybrid_valguard(
                    mod_ii_sh,
                    X_all[iva].reshape(-1, 568),
                    X_aug[iva].reshape(-1, 652),
                    len(iva),
                    8,
                )

            g_ref, h_ref, inv_ref = _eval_tr_subset(tr, ens2_base[r, tr], u_k_tr_only, gate_k_tr_only, fams_arr)
            inv_slot0 = int(np.sum(gate_k_tr_only[tr, 0] == 0))

            # Select tau_val on tr via nested inner CV
            best_tau_v = TAU_VAL_GRID[0]
            best_obj_v = (-1e9, -1e9, -1e9, -1e9)
            for tau in TAU_VAL_GRID:
                p_sub = apply_valguard_on_indices(tr, r, u_pbag, u_votes, u_raw, ii_pinv, tau)
                g_t, h_t, inv_t = _eval_tr_subset(tr, p_sub, u_k_tr_only, gate_k_tr_only, fams_arr)
                feas = 1 if (inv_t <= inv_slot0 and g_t >= 0.85 * g_ref and h_t <= h_ref) else 0
                obj = (feas, -inv_t, -h_t, g_t)
                if obj > best_obj_v:
                    best_obj_v = obj
                    best_tau_v = float(tau)
            taus_val[r, hold - 1] = best_tau_v

            # Select tau_shuf on tr via identical nested inner CV rule
            best_tau_s = TAU_VAL_GRID[0]
            best_obj_s = (-1e9, -1e9, -1e9, -1e9)
            for tau in TAU_VAL_GRID:
                p_sub = apply_valguard_on_indices(tr, r, u_pbag, u_votes, u_raw, ii_shuf, tau)
                g_t, h_t, inv_t = _eval_tr_subset(tr, p_sub, u_k_tr_only, gate_k_tr_only, fams_arr)
                feas = 1 if (inv_t <= inv_slot0 and g_t >= 0.85 * g_ref and h_t <= h_ref) else 0
                obj = (feas, -inv_t, -h_t, g_t)
                if obj > best_obj_s:
                    best_obj_s = obj
                    best_tau_s = float(tau)
            taus_shuf[r, hold - 1] = best_tau_s

            # Fit secondary harm-veto inside tr on top of inner-CV valguard picks
            tr_vg_picks_full = np.zeros(n_ctx, dtype=np.int32)
            tr_vg_picks_full[tr] = apply_valguard_on_indices(tr, r, u_pbag, u_votes, u_raw, ii_pinv, best_tau_v)
            F_harm = build_harm_features_for_picks(
                X_all, u_raw[r], u_bag[r], u_pinv[r], u_pbag[r], u_jref[r], tr_vg_picks_full
            )
            y_harm_tr_full = np.zeros(n_ctx, dtype=np.int32)
            for i in tr:
                pk = int(tr_vg_picks_full[i])
                if pk > 0 and (u_k_tr_only[i, pk] - u_k_tr_only[i, 0]) < -HARM_LOSS_THRESH:
                    y_harm_tr_full[i] = 1

            ii_harm_prob = np.zeros(n_ctx, dtype=np.float64)
            tr_sw = np.array([i for i in tr if tr_vg_picks_full[i] > 0], dtype=np.int64)
            for ii_h in range(3):
                itr_sw = np.array([i for i in tr_sw if ii_map[str(fams_arr[i])] != ii_h], dtype=np.int64)
                iva_sw = np.array([i for i in tr_sw if ii_map[str(fams_arr[i])] == ii_h], dtype=np.int64)
                if len(itr_sw) > 0 and len(set(y_harm_tr_full[itr_sw].tolist())) > 1:
                    clf_h_ii = fit_harm_clf(
                        F_harm[itr_sw],
                        y_harm_tr_full[itr_sw],
                        core.stream_seed(seed, "ii_harm", hold, ii_h),
                    )
                    if len(iva_sw) > 0:
                        ii_harm_prob[iva_sw] = clf_h_ii.predict_proba(F_harm[iva_sw])[:, 1]

            best_tau_h = TAU_HARM_GRID[-1]
            best_obj_h = (-1e9, -1e9, -1e9, -1e9)
            for tau_h in TAU_HARM_GRID:
                p_sub_h = tr_vg_picks_full[tr].copy()
                for idx_pos, i in enumerate(tr):
                    if p_sub_h[idx_pos] > 0 and ii_harm_prob[i] >= tau_h and ii_pinv[i, 0] <= best_tau_v:
                        p_sub_h[idx_pos] = 0
                g_t, h_t, inv_t = _eval_tr_subset(tr, p_sub_h, u_k_tr_only, gate_k_tr_only, fams_arr)
                feas = 1 if (inv_t <= inv_slot0 and g_t >= 0.85 * g_ref and h_t <= h_ref) else 0
                obj = (feas, -h_t, -inv_t, g_t)
                if obj > best_obj_h:
                    best_obj_h = obj
                    best_tau_h = float(tau_h)
            taus_harm[r, hold - 1] = best_tau_h

            # Fit and export outer-fold training models on `tr`
            y_tr = (gate_k_tr_only[tr] == 0).reshape(-1)
            mod_out = fit_hybrid_valguard(
                X_all[tr].reshape(-1, 568),
                X_aug[tr].reshape(-1, 652),
                y_tr,
                core.stream_seed(seed, "outer_valguard", hold),
            )
            vg_gbm_name = f"valguard_gbm_seed{seed}_fold{hold}.txt"
            vg_lr_name = f"valguard_lr_seed{seed}_fold{hold}.npz"
            vg_gbm_sha, vg_lr_sha = save_hybrid_valguard(
                mod_out, export_dir / vg_gbm_name, export_dir / vg_lr_name
            )

            rng_sh = np.random.default_rng(core.stream_seed(null_seed, "outer_shuf_perm", hold))
            y_tr_sh = rng_sh.permutation(y_tr)
            mod_sh = fit_hybrid_valguard(
                X_all[tr].reshape(-1, 568),
                X_aug[tr].reshape(-1, 652),
                y_tr_sh,
                core.stream_seed(null_seed, "outer_shuf_mod", hold),
            )
            sh_gbm_name = f"shufnull_gbm_seed{null_seed}_fold{hold}.txt"
            sh_lr_name = f"shufnull_lr_seed{null_seed}_fold{hold}.npz"
            sh_gbm_sha, sh_lr_sha = save_hybrid_valguard(
                mod_sh, export_dir / sh_gbm_name, export_dir / sh_lr_name
            )

            clf_h_out = fit_harm_clf(
                F_harm[tr_sw],
                y_harm_tr_full[tr_sw],
                core.stream_seed(seed, "outer_harm", hold),
            )
            hm_gbm_name = f"harmveto_gbm_seed{seed}_fold{hold}.txt"
            hm_gbm_path = export_dir / hm_gbm_name
            clf_h_out.booster_.save_model(str(hm_gbm_path))
            hm_gbm_sha = sha256_file(hm_gbm_path)

            outer_fold_models.append(
                {
                    "replicate": int(r),
                    "seed": int(seed),
                    "null_seed": int(null_seed),
                    "fold": int(hold),
                    "n_tr_contexts": int(len(tr)),
                    "n_va_contexts": int(len(va)),
                    "tau_val": float(best_tau_v),
                    "tau_shuf": float(best_tau_s),
                    "tau_harm": float(best_tau_h),
                    "valguard_gbm_file": vg_gbm_name,
                    "valguard_gbm_sha256": vg_gbm_sha,
                    "valguard_lr_file": vg_lr_name,
                    "valguard_lr_sha256": vg_lr_sha,
                    "shufnull_gbm_file": sh_gbm_name,
                    "shufnull_gbm_sha256": sh_gbm_sha,
                    "shufnull_lr_file": sh_lr_name,
                    "shufnull_lr_sha256": sh_lr_sha,
                    "harmveto_gbm_file": hm_gbm_name,
                    "harmveto_gbm_sha256": hm_gbm_sha,
                }
            )

        # Fit full-DL-Train frozen validity guard for seed `seed`
        y_all = (~gate_k_full).reshape(-1)
        mod_full = fit_hybrid_valguard(
            X_all.reshape(-1, 568),
            X_aug.reshape(-1, 652),
            y_all,
            core.stream_seed(seed, "full_dl_train_valguard"),
        )
        full_gbm_name = f"valguard_full_dl_train_gbm_seed{seed}.txt"
        full_lr_name = f"valguard_full_dl_train_lr_seed{seed}.npz"
        full_gbm_sha, full_lr_sha = save_hybrid_valguard(
            mod_full, export_dir / full_gbm_name, export_dir / full_lr_name
        )
        full_export_models.append(
            {
                "replicate": int(r),
                "seed": int(seed),
                "locked_tau_val": float(np.median(taus_val[r])),
                "fold_taus_val": [float(x) for x in taus_val[r]],
                "valguard_gbm_file": full_gbm_name,
                "valguard_gbm_sha256": full_gbm_sha,
                "valguard_lr_file": full_lr_name,
                "valguard_lr_sha256": full_lr_sha,
            }
        )

    outer_folds_sha256 = sha256_bytes(outer_fold_ids_all.tobytes())
    lock_payload: Dict[str, Any] = {
        "wave": "H295",
        "slug": "h295_label_free_validity_guard",
        "lane": "rd_argon",
        "seeds": list(H295_SEEDS),
        "null_seeds": list(H295_NULL_SEEDS),
        "tau_val_grid": list(TAU_VAL_GRID),
        "tau_harm_grid": list(TAU_HARM_GRID),
        "slot0_escape_floor": SLOT0_ESCAPE_FLOOR,
        "slot0_alt_max_pinv": SLOT0_ALT_MAX_PINV,
        "outer_fold_ids_sha256": outer_folds_sha256,
        "outer_fold_labels_read_before_lock": 0,
        "h287_ens_vote2_regression": reg_receipt,
        "outer_fold_models": outer_fold_models,
        "full_dl_train_export_models": full_export_models,
    }
    canonical_bytes = json.dumps(lock_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    lock_payload["fit_lock_content_sha256"] = sha256_bytes(canonical_bytes)
    lock_payload["elapsed_fit_seconds"] = round(time.time() - t0, 3)

    for path in (export_dir / "fit_lock.json", capsule_fit_lock_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(lock_payload, f, indent=2, sort_keys=True)
            f.write("\n")

    return lock_payload


def run_outer_score(
    oof_cache_path: Path,
    export_dir: Path,
    capsule_fit_lock_path: Path,
) -> Dict[str, Any]:
    """Stage 2 (`--mode outer_score`):
    1. Disable `.fit()` on both `lgb.LGBMClassifier` and `LogisticRegression`.
    2. Verify `fit_lock.json` (`fit_lock_content_sha256`) and verify every saved model file SHA256.
    3. Reconstruct `outer_fold_ids` and verify `outer_fold_ids_sha256`.
    4. Score held-out outer folds `va` across seeds 295101-295103 and null seeds 295201-295203.
    5. Run `analysis_h295.evaluate_panel` and `analysis_h295.evaluate_h295_verdict`.
    """
    t0 = time.time()

    def _forbidden_fit(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("H295 lock violation: .fit() is forbidden during outer_score!")

    lgb.LGBMClassifier.fit = _forbidden_fit  # type: ignore[method-assign]
    LogisticRegression.fit = _forbidden_fit  # type: ignore[method-assign]

    with capsule_fit_lock_path.open("r") as f:
        lock_data = json.load(f)

    check_copy = {
        k: v
        for k, v in lock_data.items()
        if k not in ("fit_lock_content_sha256", "elapsed_fit_seconds")
    }
    expected_content_sha = sha256_bytes(
        json.dumps(check_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if expected_content_sha != lock_data["fit_lock_content_sha256"]:
        raise RuntimeError(
            f"fit_lock_content_sha256 mismatch: {expected_content_sha} != {lock_data['fit_lock_content_sha256']}"
        )

    # Verify all 42 model files (36 outer-fold files + 6 full-DL-Train files)
    verified_files = 0
    for entry in lock_data["outer_fold_models"]:
        for f_key, s_key in (
            ("valguard_gbm_file", "valguard_gbm_sha256"),
            ("valguard_lr_file", "valguard_lr_sha256"),
            ("shufnull_gbm_file", "shufnull_gbm_sha256"),
            ("shufnull_lr_file", "shufnull_lr_sha256"),
            ("harmveto_gbm_file", "harmveto_gbm_sha256"),
        ):
            p = export_dir / entry[f_key]
            actual = sha256_file(p)
            if actual != entry[s_key]:
                raise RuntimeError(f"Model SHA256 mismatch on {p}: {actual} != {entry[s_key]}")
            verified_files += 1

    for entry in lock_data["full_dl_train_export_models"]:
        for f_key, s_key in (
            ("valguard_gbm_file", "valguard_gbm_sha256"),
            ("valguard_lr_file", "valguard_lr_sha256"),
        ):
            p = export_dir / entry[f_key]
            actual = sha256_file(p)
            if actual != entry[s_key]:
                raise RuntimeError(f"Full model SHA256 mismatch on {p}: {actual} != {entry[s_key]}")
            verified_files += 1

    z = np.load(oof_cache_path, allow_pickle=False)
    fams = [str(f) for f in z["fams"]]
    X_all = np.asarray(z["X"], dtype=np.float32)
    u_k = np.asarray(z["utility_k"], dtype=np.float64)
    gate_k = np.asarray(z["gate_k"], dtype=bool)
    u_pbag = np.asarray(z["u_pbag"], dtype=np.int32)
    u_jref = np.asarray(z["u_jref"], dtype=np.int32)
    u_votes = np.asarray(z["u_votes"], dtype=np.int32)
    u_raw = np.asarray(z["u_raw"], dtype=np.float64)
    u_bag = np.asarray(z["u_bag"], dtype=np.float64)
    u_pinv = np.asarray(z["u_p_inv"], dtype=np.float64)

    n_ctx = len(fams)
    X_aug = build_candidate_guard_features(X_all)

    outer_fold_ids_all = np.zeros((3, n_ctx), dtype=np.int32)
    for r in range(3):
        outer_fold_ids_all[r] = make_outer_family_folds(fams, H295_SEEDS[r], n_folds=4)
    if sha256_bytes(outer_fold_ids_all.tobytes()) != lock_data["outer_fold_ids_sha256"]:
        raise RuntimeError("outer_fold_ids_sha256 mismatch during outer_score!")

    ens2_base = np.zeros((3, n_ctx), dtype=np.int32)
    for r in range(3):
        for i in range(n_ctx):
            c0, _ = get_voted_candidates_for_context(u_pbag[r, :, i], u_votes[r, :, i], u_raw[r, :, i])
            ens2_base[r, i] = c0

    p_inv_oof = np.zeros((3, n_ctx, 8), dtype=np.float64)
    p_inv_shuf = np.zeros((3, n_ctx, 8), dtype=np.float64)
    p_harm_oof = np.zeros((3, n_ctx), dtype=np.float64)

    picks_vg = np.zeros((3, n_ctx), dtype=np.int32)
    picks_vghv = np.zeros((3, n_ctx), dtype=np.int32)
    picks_shuf = np.zeros((3, n_ctx), dtype=np.int32)

    by_r_fold = {(int(e["replicate"]), int(e["fold"])): e for e in lock_data["outer_fold_models"]}

    for r in range(3):
        fids = outer_fold_ids_all[r]
        for hold in (1, 2, 3, 4):
            entry = by_r_fold[(r, hold)]
            va = np.where(fids == hold)[0]
            tau_v = float(entry["tau_val"])
            tau_s = float(entry["tau_shuf"])
            tau_h = float(entry["tau_harm"])

            mod_vg = load_hybrid_valguard(
                export_dir / entry["valguard_gbm_file"],
                export_dir / entry["valguard_lr_file"],
            )
            p_inv_oof[r, va] = predict_hybrid_valguard(
                mod_vg,
                X_all[va].reshape(-1, 568),
                X_aug[va].reshape(-1, 652),
                len(va),
                8,
            )
            picks_vg[r, va] = apply_valguard_on_indices(
                va, r, u_pbag, u_votes, u_raw, p_inv_oof[r], tau_v
            )

            mod_sh = load_hybrid_valguard(
                export_dir / entry["shufnull_gbm_file"],
                export_dir / entry["shufnull_lr_file"],
            )
            p_inv_shuf[r, va] = predict_hybrid_valguard(
                mod_sh,
                X_all[va].reshape(-1, 568),
                X_aug[va].reshape(-1, 652),
                len(va),
                8,
            )
            picks_shuf[r, va] = apply_valguard_on_indices(
                va, r, u_pbag, u_votes, u_raw, p_inv_shuf[r], tau_s
            )

            # Secondary arm: apply frozen harmveto booster on top of valguard picks on `va`
            F_va_full = build_harm_features_for_picks(
                X_all, u_raw[r], u_bag[r], u_pinv[r], u_pbag[r], u_jref[r], picks_vg[r]
            )
            hm_booster = lgb.Booster(model_file=str(export_dir / entry["harmveto_gbm_file"]))
            p_harm_va = np.asarray(hm_booster.predict(F_va_full[va]), dtype=np.float64)
            p_harm_oof[r, va] = p_harm_va

            p_vghv_va = picks_vg[r, va].copy()
            for idx_pos, i in enumerate(va):
                if p_vghv_va[idx_pos] > 0 and p_harm_va[idx_pos] >= tau_h and p_inv_oof[r, i, 0] <= tau_v:
                    p_vghv_va[idx_pos] = 0
            picks_vghv[r, va] = p_vghv_va

    n_arms = len(ARM_NAMES)
    picks_panel = np.zeros((3, n_ctx, n_arms), dtype=np.int32)
    u_panel = np.zeros((3, n_ctx, n_arms), dtype=np.float64)

    for r in range(3):
        arm_list = [
            np.zeros(n_ctx, dtype=np.int32),  # SLOT0
            ens2_base[r],                     # ENS_VOTE2
            picks_vg[r],                      # ENS_VOTE2+VALGUARD
            picks_vghv[r],                    # ENS_VOTE2+VALGUARD+HARMVETO
            picks_shuf[r],                    # SHUFFLED_NULL
        ]
        for a_idx, p_arr in enumerate(arm_list):
            picks_panel[r, :, a_idx] = p_arr
            u_panel[r, :, a_idx] = u_k[np.arange(n_ctx), p_arr]

    panel_eval = stat.evaluate_panel(
        u_panel=u_panel,
        picks_panel=picks_panel,
        gate_k=gate_k,
        fams=fams,
        arm_names=ARM_NAMES,
        seeds=H295_SEEDS,
    )
    decision = stat.evaluate_h295_verdict(panel_eval)

    return {
        "verified_model_files": verified_files,
        "fit_lock_content_sha256": lock_data["fit_lock_content_sha256"],
        "fit_lock_sha256": sha256_file(capsule_fit_lock_path),
        "outer_cv_evaluation": panel_eval,
        "decision": decision,
        "picks_panel": picks_panel,
        "u_panel": u_panel,
        "p_inv_oof": p_inv_oof,
        "p_inv_shuf": p_inv_shuf,
        "p_harm_oof": p_harm_oof,
        "elapsed_score_seconds": round(time.time() - t0, 3),
    }
