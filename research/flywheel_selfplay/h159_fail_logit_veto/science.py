"""H159 statistical, threshold-calibration, 5-fold cross-fit, and fail-logit veto functions."""

import math
import numpy as np

SEEDS = (159101, 159102, 159103)
FOLD_SEED = 159140
BOOT_SEED = 159141
GRID_T = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
ARMS = ("R882", "VETO_FIX", "VETO_CV")


def sigmoid(x):
    arr = np.asarray(x, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("nonfinite fail_logit")
    clipped = np.clip(arr, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def select_with_veto(total_cost, fail_logit, t):
    """Select argmin total_cost among candidates with sigmoid(fail_logit) < t; else fall back to plain argmin."""
    c = np.asarray(total_cost, dtype=np.float64)
    if c.ndim != 1 or len(c) == 0 or not np.isfinite(c).all():
        raise ValueError("invalid total_cost vector")
    probs = sigmoid(fail_logit)
    if probs.shape != c.shape:
        raise ValueError("fail_logit shape mismatch")
    plain = int(np.argmin(c))
    if t is None or float(t) >= 1.0:
        return dict(
            choice=plain,
            plain_choice=plain,
            changed=False,
            fallback=False,
            vetoed_count=0,
            winner_fail_prob=float(probs[plain]),
            plain_fail_prob=float(probs[plain]),
        )
    thresh = float(t)
    if not (0.0 < thresh < 1.0):
        raise ValueError(f"threshold t={t} outside (0, 1)")
    mask = probs < thresh
    vetoed_count = int((~mask).sum())
    if bool(np.any(mask)):
        valid_indices = np.flatnonzero(mask)
        choice = int(valid_indices[int(np.argmin(c[valid_indices]))])
        fallback = False
    else:
        choice = plain
        fallback = True
    return dict(
        choice=choice,
        plain_choice=plain,
        changed=bool(choice != plain),
        fallback=fallback,
        vetoed_count=vetoed_count,
        winner_fail_prob=float(probs[choice]),
        plain_fail_prob=float(probs[plain]),
    )


def folds(rows, seed=FOLD_SEED):
    fams = np.array(sorted({r["family_id"] for r in rows}))
    return [
        sorted(g.tolist())
        for g in np.array_split(np.random.default_rng(seed).permutation(fams), 5)
    ]


def family_means(values, families):
    v = np.asarray(values, dtype=np.float64)
    if len(v) != len(families) or len(v) == 0:
        raise ValueError("family_means length mismatch")
    if not np.isfinite(v).all():
        raise ValueError("nonfinite utility")
    order = sorted(set(families))
    idx = {f: i for i, f in enumerate(order)}
    sums = np.zeros(len(order), dtype=np.float64)
    counts = np.zeros(len(order), dtype=np.float64)
    for x, f in zip(v, families, strict=True):
        sums[idx[f]] += x
        counts[idx[f]] += 1.0
    return order, sums / counts


def evaluate_threshold_on_records(records, t):
    utilities = []
    choices = []
    invalids = []
    changed = []
    fallbacks = []
    vetoed_counts = []
    for rec in records:
        sel = select_with_veto(rec["total_cost"], rec["fail_logit"], t)
        idx = sel["choice"]
        valid = bool(rec["gate"][idx])
        util = float(-rec["fitness"][idx] if valid else -2.0)
        utilities.append(util)
        choices.append(idx)
        invalids.append(not valid)
        changed.append(sel["changed"])
        fallbacks.append(sel["fallback"])
        vetoed_counts.append(sel["vetoed_count"])
    fams = [rec["family_id"] for rec in records]
    order, f_means = family_means(utilities, fams)
    return dict(
        t=None if t is None else float(t),
        utilities=utilities,
        choices=choices,
        invalids=invalids,
        changed=changed,
        fallbacks=fallbacks,
        vetoed_counts=vetoed_counts,
        context_mean_utility=float(np.mean(utilities)),
        equal_family_mean_utility=float(np.mean(f_means)),
        family_order=order,
        family_utility_map=dict(zip(order, [float(x) for x in f_means], strict=True)),
        invalid_picks=int(sum(invalids)),
        changed_choices=int(sum(changed)),
        fallback_contexts=int(sum(fallbacks)),
        total_vetoed_candidates=int(sum(vetoed_counts)),
    )


def fit_threshold_on_families(family_utility_by_t, sampled_families, grid_t=GRID_T):
    """Choose t in grid_t maximizing equal-family utility on sampled_families; break exact ties toward larger t (more conservative)."""
    if not sampled_families:
        raise ValueError("sampled_families must be non-empty")
    scores = {}
    best_t = None
    best_score = -math.inf
    for t in sorted(grid_t):
        t_val = float(t)
        vals = [family_utility_by_t[t_val][f] for f in sampled_families]
        mean_u = float(np.mean(vals))
        scores[f"{t_val:.2f}"] = mean_u
        # >= breaks exact ties in favor of larger t (more conservative veto / closer to plain argmin).
        if mean_u >= best_score - 1e-15:
            best_score = mean_u
            best_t = t_val
    return dict(
        selected_t=float(best_t),
        selected_fit_equal_family_utility=float(best_score),
        grid_fit_equal_family_utility=scores,
    )


def elo_from_win_rate(p):
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return 1500.0 + 400.0 * math.log10(p / (1.0 - p))


def family_bootstrap_contrast(
    candidate_by_seed,
    reference_by_seed,
    families,
    seed=BOOT_SEED,
    draws=10_000,
    ties="reject",
):
    if len(candidate_by_seed) != len(reference_by_seed) or not candidate_by_seed:
        raise ValueError("seed count mismatch")
    if ties not in ("reject", "half"):
        raise ValueError("unknown tie policy")
    fam_order = None
    cand_f = []
    ref_f = []
    for c, r in zip(candidate_by_seed, reference_by_seed, strict=True):
        order_c, fc = family_means(c, families)
        order_r, fr = family_means(r, families)
        if order_c != order_r:
            raise ValueError("family order mismatch")
        if fam_order is None:
            fam_order = order_c
        cand_f.append(fc)
        ref_f.append(fr)
    cand_f = np.stack(cand_f)  # [S, F]
    ref_f = np.stack(ref_f)  # [S, F]
    diff = cand_f - ref_f  # [S, F]
    per_seed = diff.mean(axis=1)
    median_gain = float(np.median(per_seed))
    n_fam = len(fam_order)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_fam, size=(draws, n_fam))
    boot_seed_means = diff[:, idx].mean(axis=2)  # [S, draws]
    boot = np.median(boot_seed_means, axis=0)  # [draws]
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    p = float((np.sum(boot <= 0.0) + 1.0) / (draws + 1.0))

    med_seed_idx = int(np.argsort(per_seed)[len(per_seed) // 2])
    fam_d = diff[med_seed_idx]
    pos = np.sort(fam_d[fam_d > 0.0])[::-1]
    top3_share = float(pos[:3].sum() / pos.sum()) if pos.sum() > 0.0 else 1.0

    def win_rate(d):
        if ties == "reject":
            return float(np.mean(d > 0.0))
        return float(np.mean(d > 0.0) + 0.5 * np.mean(d == 0.0))

    elo = elo_from_win_rate(win_rate(fam_d))
    boot_d = fam_d[idx]
    if ties == "reject":
        w = (boot_d > 0.0).mean(axis=1)
    else:
        w = (boot_d > 0.0).mean(axis=1) + 0.5 * (boot_d == 0.0).mean(axis=1)
    elos = [elo_from_win_rate(x) for x in w]
    return dict(
        families=n_fam,
        per_seed_gain=[float(x) for x in per_seed],
        median_gain=median_gain,
        ci95=ci,
        p_one_sided=p,
        seeds_beating=int(np.sum(per_seed > 0.0)),
        median_seed=SEEDS[med_seed_idx],
        top3_positive_share=top3_share,
        elo=elo,
        elo_ci95=[float(np.quantile(elos, 0.025)), float(np.quantile(elos, 0.975))],
    )


def oracle_info(records, grid_t=GRID_T):
    """Descriptive-only audit of fail_logit probabilities and avoidable -2 cliff picks (never used for selection)."""
    all_probs = []
    winner_probs = []
    cliff_contexts = []
    for idx_ctx, rec in enumerate(records):
        c = np.asarray(rec["total_cost"], dtype=np.float64)
        probs = sigmoid(rec["fail_logit"])
        gate = np.asarray(rec["gate"], dtype=bool)
        fitness = np.asarray(rec["fitness"], dtype=np.float64)
        all_probs.append(probs)
        plain = int(np.argmin(c))
        winner_probs.append(float(probs[plain]))
        if not bool(gate[plain]):
            grid_avoided_at = []
            for t in sorted(grid_t):
                sel = select_with_veto(c, rec["fail_logit"], t)
                if bool(gate[sel["choice"]]):
                    grid_avoided_at.append(float(t))
            # Check if ANY threshold t in (0, 1) avoids the -2 pick:
            # As t varies across unique candidate fail_probs, check every distinct subset S(t) = {i : probs[i] < t}.
            any_t_avoided = False
            best_any_t = None
            for cand_p in sorted(set(probs.tolist())):
                if 0.0 < cand_p < 1.0:
                    sel = select_with_veto(c, rec["fail_logit"], cand_p)
                    if bool(gate[sel["choice"]]):
                        any_t_avoided = True
                        best_any_t = float(cand_p)
                        break
            valid_idx = np.flatnonzero(gate)
            if len(valid_idx) > 0:
                best_valid = int(valid_idx[int(np.argmin(c[valid_idx]))])
                best_valid_prob = float(probs[best_valid])
                best_valid_cost = float(c[best_valid])
                best_valid_utility = float(-fitness[best_valid])
            else:
                best_valid = None
                best_valid_prob = None
                best_valid_cost = None
                best_valid_utility = None
            cliff_contexts.append(
                dict(
                    context_index=idx_ctx,
                    context_sha256=rec["context_sha256"],
                    family_id=rec["family_id"],
                    plain_choice=plain,
                    plain_cost=float(c[plain]),
                    plain_fail_prob=float(probs[plain]),
                    best_valid_choice=best_valid,
                    best_valid_cost=best_valid_cost,
                    best_valid_fail_prob=best_valid_prob,
                    best_valid_utility=best_valid_utility,
                    avoided_on_grid=bool(len(grid_avoided_at) > 0),
                    grid_thresholds_avoiding=grid_avoided_at,
                    avoided_at_any_t=any_t_avoided,
                    smallest_any_t_avoiding=best_any_t,
                )
            )
    flat_probs = np.concatenate(all_probs) if all_probs else np.zeros(1)
    win_arr = (
        np.asarray(winner_probs, dtype=np.float64) if winner_probs else np.zeros(1)
    )
    return dict(
        descriptive_only=True,
        used_for_selection=False,
        total_contexts=len(records),
        total_candidates=int(flat_probs.size),
        candidate_fail_prob_quantiles=dict(
            min=float(np.min(flat_probs)),
            p50=float(np.quantile(flat_probs, 0.50)),
            p90=float(np.quantile(flat_probs, 0.90)),
            p99=float(np.quantile(flat_probs, 0.99)),
            max=float(np.max(flat_probs)),
            count_gte_0_50=int(np.sum(flat_probs >= 0.50)),
            fraction_gte_0_50=float(np.mean(flat_probs >= 0.50)),
        ),
        r882_winner_fail_prob_quantiles=dict(
            min=float(np.min(win_arr)),
            p50=float(np.quantile(win_arr, 0.50)),
            p90=float(np.quantile(win_arr, 0.90)),
            p99=float(np.quantile(win_arr, 0.99)),
            max=float(np.max(win_arr)),
            count_gte_0_50=int(np.sum(win_arr >= 0.50)),
        ),
        r882_invalid_picks=len(cliff_contexts),
        avoidable_on_grid_count=int(
            sum(1 for c in cliff_contexts if c["avoided_on_grid"])
        ),
        avoidable_at_any_t_count=int(
            sum(1 for c in cliff_contexts if c["avoided_at_any_t"])
        ),
        cliff_contexts=cliff_contexts,
    )


def analyze_h159(
    train_records,
    val_records,
    seeds=SEEDS,
    fold_seed=FOLD_SEED,
    boot_seed=BOOT_SEED,
    grid_t=GRID_T,
):
    """Run full H159 evaluation for R882, VETO_FIX (primary on DL-Val), VETO_CV (secondary cross-fit on DL-Train), and ORACLE_INFO."""
    r882_train = evaluate_threshold_on_records(train_records, None)
    r882_val = evaluate_threshold_on_records(val_records, None)

    train_by_t = {
        float(t): evaluate_threshold_on_records(train_records, t) for t in grid_t
    }
    val_by_t = {float(t): evaluate_threshold_on_records(val_records, t) for t in grid_t}
    train_fam_util_by_t = {
        float(t): train_by_t[float(t)]["family_utility_map"] for t in grid_t
    }

    all_train_families = r882_train["family_order"]
    n_train_fams = len(all_train_families)

    # 1. VETO_FIX: point fit on all 205 TRAIN families + 3 family-resampling bootstrap seeds 159101-159103
    veto_fix_point_fit = fit_threshold_on_families(
        train_fam_util_by_t, all_train_families, grid_t=grid_t
    )
    veto_fix_seeds = []
    veto_fix_val_utils_by_seed = []
    veto_fix_val_invalids_by_seed = []
    veto_fix_val_changed_by_seed = []
    for s in seeds:
        rng = np.random.default_rng(s)
        sampled_fams = [
            all_train_families[i]
            for i in rng.integers(0, n_train_fams, size=n_train_fams)
        ]
        fit_res = fit_threshold_on_families(
            train_fam_util_by_t, sampled_fams, grid_t=grid_t
        )
        t_sel = fit_res["selected_t"]
        val_eval = val_by_t[t_sel]
        veto_fix_seeds.append(
            dict(
                seed=int(s),
                selected_t=t_sel,
                fit_equal_family_utility=fit_res["selected_fit_equal_family_utility"],
                grid_fit_equal_family_utility=fit_res["grid_fit_equal_family_utility"],
                val_context_mean_utility=val_eval["context_mean_utility"],
                val_equal_family_mean_utility=val_eval["equal_family_mean_utility"],
                val_invalid_picks=val_eval["invalid_picks"],
                val_changed_choices=val_eval["changed_choices"],
                val_fallback_contexts=val_eval["fallback_contexts"],
            )
        )
        veto_fix_val_utils_by_seed.append(val_eval["utilities"])
        veto_fix_val_invalids_by_seed.append(val_eval["invalid_picks"])
        veto_fix_val_changed_by_seed.append(val_eval["changed_choices"])

    val_families = [r["family_id"] for r in val_records]
    veto_fix_contrast = family_bootstrap_contrast(
        veto_fix_val_utils_by_seed,
        [r882_val["utilities"] for _ in seeds],
        val_families,
        seed=boot_seed,
    )

    # 2. VETO_CV: 5-fold cross-fit on 1,219 / 205 TRAIN families (fold_seed = 159140)
    fold_groups = folds(train_records, seed=fold_seed)
    fold_of_fam = {f: k for k, g in enumerate(fold_groups) for f in g}

    # Deterministic 5-fold out-of-fold evaluation
    cv_det_folds = []
    cv_det_utilities = [0.0] * len(train_records)
    cv_det_invalids = [False] * len(train_records)
    cv_det_changed = [False] * len(train_records)
    for k, held_fams in enumerate(fold_groups):
        held_set = set(held_fams)
        fit_fams = [f for f in all_train_families if f not in held_set]
        fit_res = fit_threshold_on_families(
            train_fam_util_by_t, fit_fams, grid_t=grid_t
        )
        t_k = fit_res["selected_t"]
        t_eval = train_by_t[t_k]
        held_indices = [
            i for i, r in enumerate(train_records) if r["family_id"] in held_set
        ]
        for i in held_indices:
            cv_det_utilities[i] = t_eval["utilities"][i]
            cv_det_invalids[i] = t_eval["invalids"][i]
            cv_det_changed[i] = t_eval["changed"][i]
        held_fam_gains = [
            t_eval["family_utility_map"][f] - r882_train["family_utility_map"][f]
            for f in held_fams
        ]
        cv_det_folds.append(
            dict(
                fold=k,
                fit_families=len(fit_fams),
                held_families=len(held_fams),
                held_contexts=len(held_indices),
                selected_t=t_k,
                fit_equal_family_utility=fit_res["selected_fit_equal_family_utility"],
                held_equal_family_gain_vs_r882=float(np.mean(held_fam_gains)),
                held_invalid_picks=int(
                    sum(t_eval["invalids"][i] for i in held_indices)
                ),
                held_changed_choices=int(
                    sum(t_eval["changed"][i] for i in held_indices)
                ),
            )
        )

    # 3-seed resampled 5-fold cross-fit
    veto_cv_utils_by_seed = []
    veto_cv_invalids_by_seed = []
    veto_cv_changed_by_seed = []
    veto_cv_seed_details = []
    for s in seeds:
        u_seed = [0.0] * len(train_records)
        inv_seed = [False] * len(train_records)
        chg_seed = [False] * len(train_records)
        fold_t_picks = []
        for k, held_fams in enumerate(fold_groups):
            held_set = set(held_fams)
            fit_fams = [f for f in all_train_families if f not in held_set]
            rng = np.random.default_rng(int(s) * 10 + k)
            sampled_fit = [
                fit_fams[i] for i in rng.integers(0, len(fit_fams), size=len(fit_fams))
            ]
            fit_res = fit_threshold_on_families(
                train_fam_util_by_t, sampled_fit, grid_t=grid_t
            )
            t_k = fit_res["selected_t"]
            fold_t_picks.append(t_k)
            t_eval = train_by_t[t_k]
            for i, r in enumerate(train_records):
                if fold_of_fam[r["family_id"]] == k:
                    u_seed[i] = t_eval["utilities"][i]
                    inv_seed[i] = t_eval["invalids"][i]
                    chg_seed[i] = t_eval["changed"][i]
        veto_cv_utils_by_seed.append(u_seed)
        veto_cv_invalids_by_seed.append(int(sum(inv_seed)))
        veto_cv_changed_by_seed.append(int(sum(chg_seed)))
        _, f_means_seed = family_means(u_seed, [r["family_id"] for r in train_records])
        veto_cv_seed_details.append(
            dict(
                seed=int(s),
                fold_selected_t=fold_t_picks,
                train_context_mean_utility=float(np.mean(u_seed)),
                train_equal_family_mean_utility=float(np.mean(f_means_seed)),
                train_invalid_picks=int(sum(inv_seed)),
                train_changed_choices=int(sum(chg_seed)),
            )
        )

    train_families = [r["family_id"] for r in train_records]
    veto_cv_contrast = family_bootstrap_contrast(
        veto_cv_utils_by_seed,
        [r882_train["utilities"] for _ in seeds],
        train_families,
        seed=boot_seed,
    )

    # Check whether the rule is neutral or changes choices
    grid_changed_val = {
        f"{t:.2f}": val_by_t[float(t)]["changed_choices"] for t in grid_t
    }
    grid_changed_train = {
        f"{t:.2f}": train_by_t[float(t)]["changed_choices"] for t in grid_t
    }
    total_changed_Fix_CV = int(
        sum(veto_fix_val_changed_by_seed)
        + val_by_t[veto_fix_point_fit["selected_t"]]["changed_choices"]
        + sum(veto_cv_changed_by_seed)
        + sum(cv_det_changed)
    )
    total_changed_any_grid_t = int(
        sum(grid_changed_val.values()) + sum(grid_changed_train.values())
    )

    gates = dict(
        margin_gt_0_001=bool(veto_fix_contrast["median_gain"] > 0.001),
        lcb95_gt_0=bool(veto_fix_contrast["ci95"][0] > 0.0),
        two_of_three_seeds=bool(veto_fix_contrast["seeds_beating"] >= 2),
        p_lt_0_05=bool(veto_fix_contrast["p_one_sided"] < 0.05),
        non_increasing_invalid=bool(
            max(veto_fix_val_invalids_by_seed) <= r882_val["invalid_picks"]
        ),
    )

    if total_changed_Fix_CV == 0:
        verdict = "RULE_NEUTRAL"
        r882_verdict = "NO_ADVANCE"
    elif all(gates.values()):
        verdict = "ADVANCE"
        r882_verdict = "ADVANCE"
    else:
        verdict = "NO_ADVANCE"
        r882_verdict = "NO_ADVANCE"

    return dict(
        verdict=verdict,
        r882_verdict=r882_verdict,
        rule_neutral=bool(total_changed_Fix_CV == 0),
        rule_neutral_across_entire_grid=bool(total_changed_any_grid_t == 0),
        primary_gates=gates,
        r882_baseline=dict(
            val_context_mean_utility=r882_val["context_mean_utility"],
            val_equal_family_mean_utility=r882_val["equal_family_mean_utility"],
            val_invalid_picks=r882_val["invalid_picks"],
            train_context_mean_utility=r882_train["context_mean_utility"],
            train_equal_family_mean_utility=r882_train["equal_family_mean_utility"],
            train_invalid_picks=r882_train["invalid_picks"],
        ),
        primary_veto_fix_dl_val=dict(
            point_fit_t=veto_fix_point_fit["selected_t"],
            point_fit_train_equal_family_utility=veto_fix_point_fit[
                "selected_fit_equal_family_utility"
            ],
            point_fit_grid_train_equal_family_utility=veto_fix_point_fit[
                "grid_fit_equal_family_utility"
            ],
            point_fit_val_context_mean_utility=val_by_t[
                veto_fix_point_fit["selected_t"]
            ]["context_mean_utility"],
            point_fit_val_equal_family_mean_utility=val_by_t[
                veto_fix_point_fit["selected_t"]
            ]["equal_family_mean_utility"],
            point_fit_val_invalid_picks=val_by_t[veto_fix_point_fit["selected_t"]][
                "invalid_picks"
            ],
            point_fit_val_changed_choices=val_by_t[veto_fix_point_fit["selected_t"]][
                "changed_choices"
            ],
            seeds=veto_fix_seeds,
            vs_r882=veto_fix_contrast,
        ),
        secondary_veto_cv_dl_train=dict(
            deterministic_folds=cv_det_folds,
            deterministic_context_mean_utility=float(np.mean(cv_det_utilities)),
            deterministic_equal_family_mean_utility=float(
                np.mean(family_means(cv_det_utilities, train_families)[1])
            ),
            deterministic_invalid_picks=int(sum(cv_det_invalids)),
            deterministic_changed_choices=int(sum(cv_det_changed)),
            seeds=veto_cv_seed_details,
            vs_r882=veto_cv_contrast,
        ),
        grid_profile=dict(
            dl_val={
                f"{t:.2f}": dict(
                    context_mean_utility=val_by_t[float(t)]["context_mean_utility"],
                    equal_family_mean_utility=val_by_t[float(t)][
                        "equal_family_mean_utility"
                    ],
                    invalid_picks=val_by_t[float(t)]["invalid_picks"],
                    changed_choices=val_by_t[float(t)]["changed_choices"],
                    fallback_contexts=val_by_t[float(t)]["fallback_contexts"],
                    total_vetoed_candidates=val_by_t[float(t)][
                        "total_vetoed_candidates"
                    ],
                )
                for t in grid_t
            },
            dl_train={
                f"{t:.2f}": dict(
                    context_mean_utility=train_by_t[float(t)]["context_mean_utility"],
                    equal_family_mean_utility=train_by_t[float(t)][
                        "equal_family_mean_utility"
                    ],
                    invalid_picks=train_by_t[float(t)]["invalid_picks"],
                    changed_choices=train_by_t[float(t)]["changed_choices"],
                    fallback_contexts=train_by_t[float(t)]["fallback_contexts"],
                    total_vetoed_candidates=train_by_t[float(t)][
                        "total_vetoed_candidates"
                    ],
                )
                for t in grid_t
            },
        ),
        oracle_info=dict(
            dl_val=oracle_info(val_records, grid_t=grid_t),
            dl_train=oracle_info(train_records, grid_t=grid_t),
        ),
    )
