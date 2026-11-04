"""H291 (h291_harm_veto_abstention) core implementation:
- Step 1: Bit-exact reproduction of H287 inner-CV cell (CLIP seed 287101 gain = +0.09355079434338156).
- Step 2 (Part A): 4-fold family-grouped inner CV on DL-Train (205 families, 1,219 contexts, seeds 291101-291103)
  with 644-dim features (568 H278 topk cols + 44 member/bag disagreement & slot-0 margin cols + 32 slot-diff cols),
  in-fold harm-veto classifier training, and 3-fold inner-inner CV threshold selection (zero outer-fold peeking).
- Step 3: Export frozen harm-veto boosters and write fit_lock.json BEFORE any Part B descriptive scoring.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np

import h265_core as core

H287_SEEDS: Tuple[int, int, int] = (287101, 287102, 287103)
H291_SEEDS: Tuple[int, int, int] = (291101, 291102, 291103)
TAU_GRID: Tuple[float, ...] = (0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30)
HARM_LOSS_THRESH: float = 0.01

PART_A_ARM_NAMES: Tuple[str, ...] = (
    "SLOT0",
    "CLIP",
    "ENS_VOTE3",
    "CLIP+HARMVETO",
    "ENS_VOTE2+HARMVETO",
    "PLACEBO",
    "NULL_LF",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_h287_regression_cell(
    oof_cache_path: Path,
    h287_panel_path: Path,
) -> Dict[str, Any]:
    """Step 1: Reproduce H287 inner-CV cell (CLIP seed 287101) bit-exact from published artifacts."""
    z = np.load(oof_cache_path, allow_pickle=False)
    fams = [str(f) for f in z["fams"]]
    u_k = np.asarray(z["utility_k"], dtype=np.float64)
    gate_k = np.asarray(z["gate_k"], dtype=bool)
    c_pbag = np.asarray(z["c_pbag"], dtype=np.int32)

    n_ctx = len(fams)
    fam_ids = sorted(set(fams))
    fams_arr = np.asarray(fams)
    fam_masks = [fams_arr == f for f in fam_ids]

    # Replicate r=0 corresponds to CLIP seed 287101 (native member m=0)
    picks_287101 = c_pbag[0, 0, :]
    u_287101 = u_k[np.arange(n_ctx), picks_287101].astype(np.float64)
    u_slot0 = u_k[:, 0].astype(np.float64)

    fam_u = np.array([float(u_287101[m].mean()) for m in fam_masks], dtype=np.float64)
    fam_u0 = np.array([float(u_slot0[m].mean()) for m in fam_masks], dtype=np.float64)
    fam_diff = fam_u - fam_u0

    gain = float(np.mean(fam_diff))
    harmed = int(np.sum(fam_diff < -0.01 - 1e-12))
    invalid = int(np.sum(~gate_k[np.arange(n_ctx), picks_287101]))
    changed = int(np.sum(picks_287101 != 0))

    h287_panel = np.load(h287_panel_path, allow_pickle=False)["panel"]
    # In H287 inner_cv_panel.npz, index 0 is SLOT0 and index 7 is CLIP
    ref_u = np.asarray(h287_panel[0, :, 7], dtype=np.float64)
    max_abs_diff = float(np.max(np.abs(u_287101 - ref_u)))
    expected_gain = float(
        np.mean(
            [
                float(h287_panel[0, m, 7].mean()) - float(h287_panel[0, m, 0].mean())
                for m in fam_masks
            ]
        )
    )
    gain_abs_diff = abs(gain - expected_gain)

    bit_exact = bool(
        max_abs_diff == 0.0
        and gain_abs_diff == 0.0
        and harmed == 31
        and invalid == 13
        and changed == 472
    )
    if not bit_exact:
        raise RuntimeError(
            f"Step 1 H287 regression check failed: max_abs_diff={max_abs_diff}, "
            f"gain={gain!r} vs {expected_gain!r}, harmed={harmed}, invalid={invalid}, changed={changed}"
        )
    return {
        "cell": "CLIP_seed_287101_inner_cv",
        "seed": 287101,
        "n_families": len(fam_ids),
        "n_contexts": n_ctx,
        "family_equal_mean_gain": gain,
        "expected_family_equal_mean_gain": expected_gain,
        "gain_abs_diff": gain_abs_diff,
        "utility_vector_max_abs_diff_vs_h287_panel": max_abs_diff,
        "harmed_families_gt_0p01": harmed,
        "raw_neg2_invalid_picks": invalid,
        "changed_contexts": changed,
        "bit_exact": bit_exact,
    }


def build_context_harm_features_from_arrays(
    X_all: np.ndarray,
    c_raw_r: np.ndarray,
    c_bag_r: np.ndarray,
    c_pinv_r: np.ndarray,
    c_pbag_r: np.ndarray,
    c_jref_r: np.ndarray,
    u_raw_r: np.ndarray,
    u_bag_r: np.ndarray,
    u_pinv_r: np.ndarray,
    u_pbag_r: np.ndarray,
    u_jref_r: np.ndarray,
    cand_picks: np.ndarray,
    use_clip: bool,
) -> np.ndarray:
    """Build the 644-column harm-veto feature matrix (n_ctx, 644) for proposed candidate slot `cand_picks[i]`:
    - 568 cols: H278 topk features of proposed candidate slot j (`X_all[i, j, :]`)
    - 44 cols: member disagreement stats (vote count, margin spread across 3 members and 24 bags,
      invalid-veto probability stats, slot-0 margin over best alternative, and member agreement counts)
    - 32 cols: candidate-minus-slot0 topk feature differences (`X_all[i, j, :32] - X_all[i, 0, :32]`).
    """
    n_ctx = int(X_all.shape[0])
    raw = c_raw_r if use_clip else u_raw_r       # (3, n_ctx, 8)
    bag = c_bag_r if use_clip else u_bag_r       # (3, 8, n_ctx, 8)
    pinv = c_pinv_r if use_clip else u_pinv_r    # (3, n_ctx, 8)
    pbag = c_pbag_r if use_clip else u_pbag_r    # (3, n_ctx)
    jref = c_jref_r if use_clip else u_jref_r    # (3, n_ctx)
    margins = np.array([0.10, 0.10, 0.10] if use_clip else [0.02, 0.10, 0.10], dtype=np.float64)

    m_marg = raw - raw[:, :, 0:1]
    b_marg = bag - bag[:, :, :, 0:1]
    b_marg_24 = b_marg.reshape(24, n_ctx, 8)
    b_votes = np.stack(
        [(b_marg[m] > margins[m]).sum(axis=0) for m in range(3)],
        axis=0,
    ).astype(np.float32)

    best_alt_raw = raw[:, :, 1:].max(axis=2)  # (3, n_ctx)
    slot0_marg = raw[:, :, 0] - best_alt_raw  # (3, n_ctx)

    rows: List[np.ndarray] = []
    for i in range(n_ctx):
        j = int(cand_picks[i]) if int(cand_picks[i]) > 0 else 1
        mm = m_marg[:, i, j]          # (3,)
        bm = b_marg_24[:, i, j]       # (24,)
        bv = b_votes[:, i, j]         # (3,)
        pi = pinv[:, i, j]            # (3,)
        pi0 = pinv[:, i, 0]           # (3,)
        s0m = slot0_marg[:, i]        # (3,)

        pbag_agree = sum(int(pbag[m, i]) == j for m in range(3))
        jref_agree = sum(int(jref[m, i]) == j for m in range(3))
        cpbag_agree = sum(int(c_pbag_r[m, i]) == j for m in range(3))
        upbag_agree = sum(int(u_pbag_r[m, i]) == j for m in range(3))
        n_nonzero_members = sum(int(pbag[m, i]) != 0 for m in range(3))

        extra = np.array(
            [
                *mm,
                float(mm.mean()),
                float(mm.std()),
                float(mm.min()),
                float(mm.max()),
                float(mm.max() - mm.min()),
                *bv,
                float(bv.sum()),
                float(bv.min()),
                float(bv.max()),
                float(bv.std()),
                float(bm.mean()),
                float(bm.std()),
                float(bm.min()),
                float(np.quantile(bm, 0.25)),
                float(np.median(bm)),
                float(np.quantile(bm, 0.75)),
                float(bm.max()),
                float((bm > 0.0).mean()),
                float((bm > 0.10).mean()),
                float((bm > 0.20).mean()),
                *pi,
                float(pi.mean()),
                float(pi.max()),
                float(pi.max() - pi.min()),
                float(pi0.mean()),
                float(pi0.max()),
                *s0m,
                float(s0m.mean()),
                float(s0m.min()),
                float(s0m.max() - s0m.min()),
                float(pbag_agree),
                float(jref_agree),
                float(cpbag_agree),
                float(upbag_agree),
                float(n_nonzero_members),
                *(X_all[i, j, :32] - X_all[i, 0, :32]),
            ],
            dtype=np.float32,
        )
        rows.append(np.concatenate([X_all[i, j, :], extra], axis=0))
    return np.stack(rows, axis=0)


def fit_harm_clf(X_tr: np.ndarray, y_tr: np.ndarray, rstate: int) -> lgb.LGBMClassifier:
    """Fit regularized shallow LightGBM binary classifier predicting P(switch from slot 0 loses > 0.01)."""
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
        random_state=int(rstate),
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    clf.fit(X_tr, y_tr.astype(int))
    return clf


def permute_labels_within_family(
    y: np.ndarray,
    fams: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute binary harm labels within multi-context families (and permute remaining singleton-family
    proposed switches together so the placebo classifier preserves the exact marginal harm rate while
    destroying feature-label correspondence)."""
    y_pl = np.asarray(y, dtype=np.int32).copy()
    fams_str = np.asarray([str(f) for f in fams])
    remainder: List[int] = []
    for f in sorted(set(fams_str.tolist())):
        loc = np.where(fams_str == f)[0]
        if len(loc) > 1 and len(set(y_pl[loc].tolist())) > 1:
            y_pl[loc] = rng.permutation(y_pl[loc])
        else:
            remainder.extend(loc.tolist())
    if remainder:
        rem_idx = np.asarray(remainder, dtype=np.int64)
        y_pl[rem_idx] = rng.permutation(y_pl[rem_idx])
    return y_pl


def compute_vote_picks(pbag_r: np.ndarray, min_agree: int) -> np.ndarray:
    """Compute per-context candidate pick requiring >= min_agree of 3 members to agree on j > 0."""
    n_ctx = int(pbag_r.shape[1])
    out = np.zeros(n_ctx, dtype=np.int32)
    for i in range(n_ctx):
        nz = [int(pbag_r[m, i]) for m in range(3) if int(pbag_r[m, i]) != 0]
        for cand in sorted(set(nz)):
            if nz.count(cand) >= int(min_agree):
                out[i] = int(cand)
                break
    return out


def _eval_subset_gain_harm(
    indices: np.ndarray,
    picks: np.ndarray,
    u_k: np.ndarray,
    fams_arr: np.ndarray,
) -> Tuple[float, int]:
    sub_fams = sorted(set(str(fams_arr[i]) for i in indices))
    u = u_k[indices, picks[indices]]
    diff = u - u_k[indices, 0]
    f_sub = fams_arr[indices]
    fg = np.array([float(np.mean(diff[f_sub == f])) for f in sub_fams], dtype=np.float64)
    return float(np.mean(fg)), int(np.sum(fg < -0.01 - 1e-12))


def run_part_a_and_export_locks(
    oof_cache_path: Path,
    export_dir: Path,
) -> Dict[str, Any]:
    """Run Part A (DL-Train 4-fold family-grouped inner CV across seeds 291101-291103),
    select thresholds strictly via 3-fold inner-inner CV inside each outer training split `tr`,
    and export frozen full-DL-Train harm-veto boosters + fit_lock.json to `export_dir`."""
    t0 = time.time()
    export_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(oof_cache_path, allow_pickle=False)
    cids = [str(c) for c in z["cids"]]
    fams = [str(f) for f in z["fams"]]
    fams_arr = np.asarray(fams)
    X_all = np.asarray(z["X"], dtype=np.float32)
    u_k = np.asarray(z["utility_k"], dtype=np.float64)
    fold_ids = np.asarray(z["fold_ids"], dtype=np.int32)

    c_pbag = np.asarray(z["c_pbag"], dtype=np.int32)
    u_pbag = np.asarray(z["u_pbag"], dtype=np.int32)
    c_jref = np.asarray(z["c_jref"], dtype=np.int32)
    u_jref = np.asarray(z["u_jref"], dtype=np.int32)
    c_raw = np.asarray(z["c_raw"], dtype=np.float64)
    u_raw = np.asarray(z["u_raw"], dtype=np.float64)
    c_bag = np.asarray(z["c_bag"], dtype=np.float64)
    u_bag = np.asarray(z["u_bag"], dtype=np.float64)
    c_pinv = np.asarray(z["c_p_inv"], dtype=np.float64)
    u_pinv = np.asarray(z["u_p_inv"], dtype=np.float64)

    n_ctx = len(fams)

    # Base proposals per replicate r in 0..2
    clip_prop = np.array([c_pbag[r, r, :] for r in range(3)], dtype=np.int32)
    clip2_prop = np.array([compute_vote_picks(c_pbag[r], min_agree=2) for r in range(3)], dtype=np.int32)
    ens2_prop = np.array([compute_vote_picks(u_pbag[r], min_agree=2) for r in range(3)], dtype=np.int32)
    ens3_prop = np.array([compute_vote_picks(u_pbag[r], min_agree=3) for r in range(3)], dtype=np.int32)

    def _run_harm_veto_arm(
        arm_tag: str,
        base_picks: np.ndarray,
        use_clip: bool,
        selector_mode: str,
    ) -> Dict[str, Any]:
        p_oof = np.zeros((3, n_ctx), dtype=np.float64)
        p_oof_pl = np.zeros((3, n_ctx), dtype=np.float64)
        taus = np.zeros((3, 4), dtype=np.float64)
        picks_hv = np.zeros((3, n_ctx), dtype=np.int32)
        picks_pl = np.zeros((3, n_ctx), dtype=np.int32)
        picks_nl = np.zeros((3, n_ctx), dtype=np.int32)
        exported_files: List[Dict[str, Any]] = []

        for r in range(3):
            F_r = build_context_harm_features_from_arrays(
                X_all=X_all,
                c_raw_r=c_raw[r],
                c_bag_r=c_bag[r],
                c_pinv_r=c_pinv[r],
                c_pbag_r=c_pbag[r],
                c_jref_r=c_jref[r],
                u_raw_r=u_raw[r],
                u_bag_r=u_bag[r],
                u_pinv_r=u_pinv[r],
                u_pbag_r=u_pbag[r],
                u_jref_r=u_jref[r],
                cand_picks=base_picks[r],
                use_clip=use_clip,
            )
            y_harm = np.array(
                [
                    1 if (base_picks[r, i] != 0 and (u_k[i, base_picks[r, i]] - u_k[i, 0]) < -HARM_LOSS_THRESH) else 0
                    for i in range(n_ctx)
                ],
                dtype=np.int32,
            )
            fids = fold_ids[r]
            rng_nl = np.random.default_rng(core.stream_seed(H291_SEEDS[r], arm_tag, "null_lf"))

            for hold in (1, 2, 3, 4):
                tr_all = np.where(fids != hold)[0]
                va_all = np.where(fids == hold)[0]
                tr_idx = np.where((fids != hold) & (base_picks[r] != 0))[0]
                va_idx = np.where((fids == hold) & (base_picks[r] != 0))[0]

                # 3-fold family-grouped inner-inner split inside `tr_all` (zero outer-fold `va` peeking)
                tr_fams = sorted(set(str(fams[i]) for i in tr_all))
                rng_ii = np.random.default_rng(core.stream_seed(H291_SEEDS[r], arm_tag, "ii", hold))
                perm_fams = list(tr_fams)
                rng_ii.shuffle(perm_fams)
                ii_map = {f: (idx % 3) for idx, f in enumerate(perm_fams)}
                ii_prob = np.zeros(n_ctx, dtype=np.float64)
                for ii_h in range(3):
                    itr = [i for i in tr_idx if ii_map[str(fams[i])] != ii_h]
                    iva = [i for i in tr_idx if ii_map[str(fams[i])] == ii_h]
                    clf_ii = fit_harm_clf(
                        F_r[itr],
                        y_harm[itr],
                        core.stream_seed(H291_SEEDS[r], arm_tag, hold, ii_h),
                    )
                    if len(iva) > 0:
                        ii_prob[iva] = clf_ii.predict_proba(F_r[iva])[:, 1]

                _, h_ens3_tr = _eval_subset_gain_harm(tr_all, ens3_prop[r], u_k, fams_arr)
                g_clip_tr, _ = _eval_subset_gain_harm(tr_all, clip_prop[r], u_k, fams_arr)

                best_tau = TAU_GRID[0]
                best_obj = (-1e9, -1e9, -1e9)
                for tau in TAU_GRID:
                    p_tr = np.zeros(n_ctx, dtype=np.int32)
                    for i in tr_all:
                        if base_picks[r, i] != 0 and ii_prob[i] < tau:
                            p_tr[i] = int(base_picks[r, i])
                    g_t, h_t = _eval_subset_gain_harm(tr_all, p_tr, u_k, fams_arr)
                    if selector_mode == "harm_cap":
                        feas = 1 if (h_t <= h_ens3_tr and g_t >= 0.02) else 0
                        obj = (feas, g_t if feas else -abs(h_t - h_ens3_tr), g_t)
                    else:
                        feas = 1 if (g_t >= 0.70 * g_clip_tr) else 0
                        obj = (feas, -h_t if feas else g_t, g_t)
                    if obj > best_obj:
                        best_obj = obj
                        best_tau = tau
                taus[r, hold - 1] = float(best_tau)

                clf_outer = fit_harm_clf(
                    F_r[tr_idx],
                    y_harm[tr_idx],
                    core.stream_seed(H291_SEEDS[r], arm_tag, hold),
                )
                if len(va_idx) > 0:
                    p_oof[r, va_idx] = clf_outer.predict_proba(F_r[va_idx])[:, 1]

                rng_pl = np.random.default_rng(core.stream_seed(H291_SEEDS[r], arm_tag, "placebo", hold))
                y_pl = permute_labels_within_family(y_harm[tr_idx], fams_arr[tr_idx], rng_pl)
                clf_pl = fit_harm_clf(
                    F_r[tr_idx],
                    y_pl,
                    core.stream_seed(H291_SEEDS[r], arm_tag, "pl_clf", hold),
                )
                if len(va_idx) > 0:
                    p_oof_pl[r, va_idx] = clf_pl.predict_proba(F_r[va_idx])[:, 1]

                kept = 0
                for i in va_idx:
                    if p_oof[r, i] < best_tau:
                        picks_hv[r, i] = int(base_picks[r, i])
                        kept += 1
                    if p_oof_pl[r, i] < best_tau:
                        picks_pl[r, i] = int(base_picks[r, i])
                if len(va_idx) > 0 and kept > 0:
                    chosen = rng_nl.choice(va_idx, size=min(kept, len(va_idx)), replace=False)
                    for i in chosen:
                        picks_nl[r, i] = int(base_picks[r, i])

            # Fit full-DL-Train harm-veto booster and placebo booster on all proposed switches of seed r
            all_prop_idx = np.where(base_picks[r] != 0)[0]
            clf_full = fit_harm_clf(
                F_r[all_prop_idx],
                y_harm[all_prop_idx],
                core.stream_seed(H291_SEEDS[r], arm_tag, "full_dl_train"),
            )
            fname = f"harm_veto_{arm_tag.lower()}_seed{H291_SEEDS[r]}.txt"
            fpath = export_dir / fname
            clf_full.booster_.save_model(str(fpath))
            exported_files.append(
                {
                    "arm": arm_tag,
                    "role": "harm_veto_full_dl_train",
                    "seed": int(H291_SEEDS[r]),
                    "locked_tau": float(np.median(taus[r])),
                    "fold_taus": [float(x) for x in taus[r]],
                    "n_train_proposed_switches": int(len(all_prop_idx)),
                    "n_train_harm_positive": int(np.sum(y_harm[all_prop_idx])),
                    "file": fname,
                    "bytes": int(fpath.stat().st_size),
                    "sha256": sha256_file(fpath),
                }
            )

            rng_pl_full = np.random.default_rng(
                core.stream_seed(H291_SEEDS[r], arm_tag, "placebo_full")
            )
            y_pl_full = permute_labels_within_family(
                y_harm[all_prop_idx], fams_arr[all_prop_idx], rng_pl_full
            )
            clf_pl_full = fit_harm_clf(
                F_r[all_prop_idx],
                y_pl_full,
                core.stream_seed(H291_SEEDS[r], arm_tag, "pl_clf_full"),
            )
            fname_pl = f"placebo_{arm_tag.lower()}_seed{H291_SEEDS[r]}.txt"
            fpath_pl = export_dir / fname_pl
            clf_pl_full.booster_.save_model(str(fpath_pl))
            exported_files.append(
                {
                    "arm": arm_tag,
                    "role": "placebo_full_dl_train",
                    "seed": int(H291_SEEDS[r]),
                    "locked_tau": float(np.median(taus[r])),
                    "fold_taus": [float(x) for x in taus[r]],
                    "n_train_proposed_switches": int(len(all_prop_idx)),
                    "n_train_harm_positive": int(np.sum(y_pl_full)),
                    "file": fname_pl,
                    "bytes": int(fpath_pl.stat().st_size),
                    "sha256": sha256_file(fpath_pl),
                }
            )

        return {
            "taus": taus,
            "p_oof": p_oof,
            "p_oof_pl": p_oof_pl,
            "picks_hv": picks_hv,
            "picks_pl": picks_pl,
            "picks_nl": picks_nl,
            "exported_files": exported_files,
        }

    # Run the two primary harm-veto arms and the auxiliary CLIP_VOTE2+HARMVETO diagnostic arm
    res_ens2 = _run_harm_veto_arm(
        arm_tag="ENS_VOTE2_HARMVETO",
        base_picks=ens2_prop,
        use_clip=False,
        selector_mode="harm_cap",
    )
    res_clip1 = _run_harm_veto_arm(
        arm_tag="CLIP_HARMVETO",
        base_picks=clip_prop,
        use_clip=True,
        selector_mode="gain70",
    )
    res_clip2_diag = _run_harm_veto_arm(
        arm_tag="CLIP_VOTE2_HARMVETO",
        base_picks=clip2_prop,
        use_clip=True,
        selector_mode="harm_cap",
    )

    # Assemble the 7-arm Part A panel:
    # ("SLOT0", "CLIP", "ENS_VOTE3", "CLIP+HARMVETO", "ENS_VOTE2+HARMVETO", "PLACEBO", "NULL_LF")
    # Note: PLACEBO and NULL_LF in the primary panel are paired with ENS_VOTE2+HARMVETO (our best-by-prereg
    # harm-cutting candidate), and we also record the CLIP+HARMVETO-paired PLACEBO and NULL_LF in diagnostics.
    n_arms = len(PART_A_ARM_NAMES)
    picks_panel = np.zeros((3, n_ctx, n_arms), dtype=np.int32)
    u_panel = np.zeros((3, n_ctx, n_arms), dtype=np.float64)

    for r in range(3):
        arm_picks_list = [
            np.zeros(n_ctx, dtype=np.int32),  # SLOT0
            clip_prop[r],                     # CLIP
            ens3_prop[r],                     # ENS_VOTE3
            res_clip1["picks_hv"][r],         # CLIP+HARMVETO
            res_ens2["picks_hv"][r],          # ENS_VOTE2+HARMVETO
            res_ens2["picks_pl"][r],          # PLACEBO (paired with ENS_VOTE2+HARMVETO)
            res_ens2["picks_nl"][r],          # NULL_LF (paired with ENS_VOTE2+HARMVETO)
        ]
        for a_idx, p_arr in enumerate(arm_picks_list):
            picks_panel[r, :, a_idx] = p_arr
            u_panel[r, :, a_idx] = u_k[np.arange(n_ctx), p_arr]

    # Auxiliary diagnostic panel for CLIP+HARMVETO controls and CLIP_VOTE2+HARMVETO
    diag_arm_names = (
        "SLOT0",
        "CLIP",
        "ENS_VOTE3",
        "CLIP_VOTE2",
        "ENS_VOTE2",
        "CLIP_VOTE2+HARMVETO",
        "CLIP+HARMVETO_PLACEBO",
        "CLIP+HARMVETO_NULL_LF",
    )
    diag_picks_panel = np.zeros((3, n_ctx, len(diag_arm_names)), dtype=np.int32)
    diag_u_panel = np.zeros((3, n_ctx, len(diag_arm_names)), dtype=np.float64)
    for r in range(3):
        d_list = [
            np.zeros(n_ctx, dtype=np.int32),
            clip_prop[r],
            ens3_prop[r],
            clip2_prop[r],
            ens2_prop[r],
            res_clip2_diag["picks_hv"][r],
            res_clip1["picks_pl"][r],
            res_clip1["picks_nl"][r],
        ]
        for a_idx, p_arr in enumerate(d_list):
            diag_picks_panel[r, :, a_idx] = p_arr
            diag_u_panel[r, :, a_idx] = u_k[np.arange(n_ctx), p_arr]

    all_exported = (
        res_ens2["exported_files"]
        + res_clip1["exported_files"]
        + res_clip2_diag["exported_files"]
    )

    return {
        "cids": cids,
        "fams": fams,
        "arm_names": list(PART_A_ARM_NAMES),
        "picks_panel": picks_panel,
        "u_panel": u_panel,
        "diag_arm_names": list(diag_arm_names),
        "diag_picks_panel": diag_picks_panel,
        "diag_u_panel": diag_u_panel,
        "inner_inner_selected_taus": {
            "ENS_VOTE2+HARMVETO": res_ens2["taus"].tolist(),
            "CLIP+HARMVETO": res_clip1["taus"].tolist(),
            "CLIP_VOTE2+HARMVETO": res_clip2_diag["taus"].tolist(),
        },
        "locked_part_b_taus": {
            "ENS_VOTE2+HARMVETO": [float(np.median(res_ens2["taus"][r])) for r in range(3)],
            "CLIP+HARMVETO": [float(np.median(res_clip1["taus"][r])) for r in range(3)],
            "CLIP_VOTE2+HARMVETO": [float(np.median(res_clip2_diag["taus"][r])) for r in range(3)],
        },
        "exported_files": all_exported,
        "elapsed_sec": float(time.time() - t0),
    }


def score_part_b_on_h286_even(
    prepared_h286_npz: Path,
    h287_export_dir: Path,
    h291_export_dir: Path,
    fit_lock_path: Path,
) -> Dict[str, Any]:
    """Step 4 (Part B, descriptive only, SEMI_FRESH, labeled reused):
    Verify fit_lock.json and all exported model SHA256 digests, disable all LightGBM .fit() methods,
    and score the frozen best-by-prereg arm (ENS_VOTE2+HARMVETO) and full Part A arm panel on
    H286 v2 200 EVEN r3175 pools (200 families, 200 contexts).
    """
    import bagged
    import veto

    t0 = time.time()
    fit_lock = json.loads(fit_lock_path.read_text(encoding="utf-8"))
    check_copy = dict(fit_lock)
    expected_content_sha = check_copy.pop("fit_lock_content_sha256")
    actual_content_sha = hashlib.sha256(
        json.dumps(check_copy, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if actual_content_sha != expected_content_sha:
        raise RuntimeError(
            f"fit_lock_content_sha256 mismatch: {actual_content_sha} != {expected_content_sha}"
        )

    # Verify every exported H291 model file sha256 against fit_lock.json
    model_meta_by_key: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for entry in fit_lock["exported_harm_veto_models"]:
        mpath = h291_export_dir / entry["file"]
        if sha256_file(mpath) != entry["sha256"]:
            raise RuntimeError(f"SHA256 mismatch on exported H291 model {mpath}")
        model_meta_by_key[(entry["arm"], entry["role"], int(entry["seed"]))] = entry

    # Verify H287 ensemble_spec.json and all 60 H287 member boosters
    h287_spec = json.loads((h287_export_dir / "ensemble_spec.json").read_text(encoding="utf-8"))
    for group_key in ("h284_bag_switch_members", "clip_bag_switch_members"):
        for mem in h287_spec[group_key]:
            for sub in (mem["ranker"], mem["veto"], *mem["bags"]):
                mpath = h287_export_dir / sub["file"]
                if sha256_file(mpath) != sub["sha256"]:
                    raise RuntimeError(f"SHA256 mismatch on H287 model {mpath}")

    # Hard-block any fitting during Part B
    def _forbidden_fit(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("Fitting is strictly forbidden during Part B descriptive scoring")

    lgb.LGBMRanker.fit = _forbidden_fit  # type: ignore[method-assign]
    lgb.LGBMClassifier.fit = _forbidden_fit  # type: ignore[method-assign]
    lgb.train = _forbidden_fit  # type: ignore[assignment]

    z = np.load(prepared_h286_npz, allow_pickle=False)
    cids = [str(c) for c in z["cids"]]
    fams = [str(f) for f in z["fams"]]
    X_all = np.asarray(z["X"], dtype=np.float32)          # (200, 8, 568)
    u_k = np.asarray(z["utility_k"], dtype=np.float64)    # (200, 8)
    gate_k = np.asarray(z["gate_k"], dtype=bool)          # (200, 8)
    n_ctx, K, d_feat = X_all.shape
    X_flat = X_all.reshape(n_ctx * K, d_feat)

    def _score_single_member(cell_info: Dict[str, Any]) -> Dict[str, Any]:
        r_path = h287_export_dir / cell_info["ranker"]["file"]
        v_path = h287_export_dir / cell_info["veto"]["file"]
        n_est = int(cell_info["selected_n_estimators"])
        raw_m = cell_info["strict_margin"]
        strict_margin = float("inf") if raw_m == "inf" else float(raw_m)
        tau = float(cell_info.get("selected_tau", veto.TAU))
        q_fixed = int(cell_info.get("q_fixed", bagged.Q_FIXED))

        ranker = lgb.Booster(model_file=str(r_path))
        gate = lgb.Booster(model_file=str(v_path))
        p_inv = np.asarray(gate.predict(X_flat), dtype=np.float64).reshape(n_ctx, K)
        raw_scores = np.asarray(ranker.predict(X_flat, num_iteration=n_est), dtype=np.float64).reshape(n_ctx, K)
        masked_scores = veto.mask(raw_scores, p_inv, tau=tau)
        j_ref = [core.select_from_top4_scores(masked_scores[i], strict_margin) for i in range(n_ctx)]

        bag_scores = []
        for b_info in cell_info["bags"]:
            bm = lgb.Booster(model_file=str(h287_export_dir / b_info["file"]))
            sb = np.asarray(bm.predict(X_flat, num_iteration=n_est), dtype=np.float64).reshape(n_ctx, K)
            bag_scores.append(sb)
        bag_scores_arr = np.stack(bag_scores, axis=0)
        votes = bagged.vote_counts(bag_scores_arr, j_ref, strict_margin)
        picks_bag = bagged.switch_by_votes(j_ref, votes, q_fixed)
        return {
            "p_inv": p_inv,
            "raw_scores": raw_scores,
            "bag_scores": bag_scores_arr,
            "j_ref": j_ref,
            "votes": votes,
            "picks_bag": picks_bag,
        }

    h284_members = [_score_single_member(c) for c in h287_spec["h284_bag_switch_members"]]
    clip_members = [_score_single_member(c) for c in h287_spec["clip_bag_switch_members"]]

    u_pinv = np.stack([m["p_inv"] for m in h284_members], axis=0)
    u_raw = np.stack([m["raw_scores"] for m in h284_members], axis=0)
    u_bag = np.stack([m["bag_scores"] for m in h284_members], axis=0)
    u_jref = np.asarray([m["j_ref"] for m in h284_members], dtype=np.int32)
    u_pbag = np.asarray([m["picks_bag"] for m in h284_members], dtype=np.int32)

    c_pinv = np.stack([m["p_inv"] for m in clip_members], axis=0)
    c_raw = np.stack([m["raw_scores"] for m in clip_members], axis=0)
    c_bag = np.stack([m["bag_scores"] for m in clip_members], axis=0)
    c_jref = np.asarray([m["j_ref"] for m in clip_members], dtype=np.int32)
    c_pbag = np.asarray([m["picks_bag"] for m in clip_members], dtype=np.int32)

    # Verify SLOT0 utility matches H286 published frozen_utility[r, :, 0] bit-exact
    # and R882_FIT (u_pbag) matches H290 published R882_FIT ([21, 7, 17] harmed, [2, 2, 4] raw -2, median gain 0.06299527974306189)
    h286_pan = np.load(
        prepared_h286_npz.parent / "eval-001-fresh_panels.npz", allow_pickle=False
    )["frozen_utility"]
    r882_harms = []
    r882_neg2 = []
    r882_gains = []
    for r in range(3):
        if float(np.max(np.abs(u_k[:, 0] - h286_pan[r, :, 0]))) > 1e-12:
            raise RuntimeError(f"SLOT0 utility mismatch vs H286 published panel on seed index {r}")
        u_r882 = u_k[np.arange(n_ctx), u_pbag[r]]
        diff_r882 = u_r882 - u_k[:, 0]
        r882_gains.append(float(np.mean(diff_r882)))
        r882_harms.append(int(np.sum(diff_r882 < -0.01 - 1e-12)))
        r882_neg2.append(int(np.sum(~gate_k[np.arange(n_ctx), u_pbag[r]])))
    if r882_harms != [21, 7, 17] or r882_neg2 != [2, 2, 4] or abs(float(np.median(r882_gains)) - 0.06299527974306189) > 1e-12:
        raise RuntimeError(
            f"R882_FIT mismatch vs H290 published reference: gains={r882_gains}, harms={r882_harms}, neg2={r882_neg2}"
        )

    clip_prop = np.array([c_pbag[r, :] for r in range(3)], dtype=np.int32)
    r882_prop = np.array([u_pbag[r, :] for r in range(3)], dtype=np.int32)
    ens3_1d = compute_vote_picks(u_pbag, min_agree=3)
    ens2_1d = compute_vote_picks(u_pbag, min_agree=2)
    clip2_1d = compute_vote_picks(c_pbag, min_agree=2)
    ens3_prop = np.stack([ens3_1d for _ in range(3)], axis=0)
    ens2_prop = np.stack([ens2_1d for _ in range(3)], axis=0)
    clip2_prop = np.stack([clip2_1d for _ in range(3)], axis=0)

    def _score_frozen_arm(
        arm_tag: str,
        base_picks: np.ndarray,
        use_clip: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        picks_hv = np.zeros((3, n_ctx), dtype=np.int32)
        picks_pl = np.zeros((3, n_ctx), dtype=np.int32)
        picks_nl = np.zeros((3, n_ctx), dtype=np.int32)
        probs_hv = np.zeros((3, n_ctx), dtype=np.float64)

        for r, hseed in enumerate(H291_SEEDS):
            F_r = build_context_harm_features_from_arrays(
                X_all=X_all,
                c_raw_r=c_raw,
                c_bag_r=c_bag,
                c_pinv_r=c_pinv,
                c_pbag_r=c_pbag,
                c_jref_r=c_jref,
                u_raw_r=u_raw,
                u_bag_r=u_bag,
                u_pinv_r=u_pinv,
                u_pbag_r=u_pbag,
                u_jref_r=u_jref,
                cand_picks=base_picks[r],
                use_clip=use_clip,
            )
            meta_hv = model_meta_by_key[(arm_tag, "harm_veto_full_dl_train", int(hseed))]
            meta_pl = model_meta_by_key[(arm_tag, "placebo_full_dl_train", int(hseed))]
            tau = float(meta_hv["locked_tau"])

            bo_hv = lgb.Booster(model_file=str(h291_export_dir / meta_hv["file"]))
            bo_pl = lgb.Booster(model_file=str(h291_export_dir / meta_pl["file"]))

            p_hv = bo_hv.predict(F_r)
            p_pl = bo_pl.predict(F_r)
            probs_hv[r] = p_hv

            prop_idx = np.where(base_picks[r] != 0)[0]
            kept = 0
            for i in prop_idx:
                if float(p_hv[i]) < tau:
                    picks_hv[r, i] = int(base_picks[r, i])
                    kept += 1
                if float(p_pl[i]) < tau:
                    picks_pl[r, i] = int(base_picks[r, i])

            rng_nl = np.random.default_rng(core.stream_seed(int(hseed), arm_tag, "part_b_null_lf"))
            if len(prop_idx) > 0 and kept > 0:
                chosen = rng_nl.choice(prop_idx, size=min(kept, len(prop_idx)), replace=False)
                for i in chosen:
                    picks_nl[r, i] = int(base_picks[r, i])

        return picks_hv, picks_pl, picks_nl, probs_hv

    ens2_hv, ens2_pl, ens2_nl, _ = _score_frozen_arm("ENS_VOTE2_HARMVETO", ens2_prop, use_clip=False)
    clip1_hv, clip1_pl, clip1_nl, _ = _score_frozen_arm("CLIP_HARMVETO", clip_prop, use_clip=True)
    clip2_hv, _, _, _ = _score_frozen_arm("CLIP_VOTE2_HARMVETO", clip2_prop, use_clip=True)

    part_b_arm_names = (
        "SLOT0",
        "CLIP",
        "ENS_VOTE3",
        "CLIP+HARMVETO",
        "ENS_VOTE2+HARMVETO",
        "PLACEBO",
        "NULL_LF",
        "R882_FIT",
        "ENS_VOTE2",
        "CLIP_VOTE2",
        "CLIP_VOTE2+HARMVETO",
    )
    n_arms = len(part_b_arm_names)
    picks_panel = np.zeros((3, n_ctx, n_arms), dtype=np.int32)
    u_panel = np.zeros((3, n_ctx, n_arms), dtype=np.float64)

    for r in range(3):
        a_list = [
            np.zeros(n_ctx, dtype=np.int32),  # SLOT0
            clip_prop[r],                     # CLIP
            ens3_prop[r],                     # ENS_VOTE3
            clip1_hv[r],                      # CLIP+HARMVETO
            ens2_hv[r],                       # ENS_VOTE2+HARMVETO (best-by-prereg arm)
            ens2_pl[r],                       # PLACEBO
            ens2_nl[r],                       # NULL_LF
            r882_prop[r],                     # R882_FIT (H284 BAG_SWITCH reference)
            ens2_prop[r],                     # ENS_VOTE2 (ungated)
            clip2_prop[r],                    # CLIP_VOTE2 (ungated)
            clip2_hv[r],                      # CLIP_VOTE2+HARMVETO
        ]
        for a_idx, p_arr in enumerate(a_list):
            picks_panel[r, :, a_idx] = p_arr
            u_panel[r, :, a_idx] = u_k[np.arange(n_ctx), p_arr]

    return {
        "cohort": "H286_v2_200_EVEN_r3175",
        "scope": "SEMI_FRESH_DESCRIPTIVE_ONLY",
        "reused": True,
        "fresh_claim": False,
        "fit_lock_verified": True,
        "fit_lock_content_sha256": expected_content_sha,
        "r882_fit_bit_exact_with_h286": True,
        "cids": cids,
        "fams": fams,
        "arm_names": list(part_b_arm_names),
        "picks_panel": picks_panel,
        "u_panel": u_panel,
        "elapsed_sec": float(time.time() - t0),
    }

