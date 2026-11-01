"""Pure-NumPy fold, inner-CV lambda selection, bootstrap, Holm, Elo, and pseudoreplication utilities for H172."""

import hashlib
import math

import numpy as np

SEEDS = (172101, 172102, 172103)
FOLD_SEED = 172140
BOOT_SEED = 172141
ARMS = ("CTRL", "REFIT_CV", "REFIT_BAG", "REFIT_SHUF")
TREATMENT_ARMS = ("REFIT_CV", "REFIT_BAG")
ALL_ARMS = ("R882", "CTRL", "REFIT_CV", "REFIT_BAG", "REFIT_SHUF")

LAMBDA_GRID = (10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1)
LAMBDA_LABELS = ("1e4", "1e3", "1e2", "1e1", "1", "1e-1", "inf")
LAMBDA_BY_LABEL = {
    "1e4": 10000.0,
    "1e3": 1000.0,
    "1e2": 100.0,
    "1e1": 10.0,
    "1": 1.0,
    "1e-1": 0.1,
    "inf": math.inf,
}
FINITE_LAMBDA_LABELS = ("1e4", "1e3", "1e2", "1e1", "1", "1e-1")

BAG_COUNT = 8
LBFGS_MAX_ITER = 500
LBFGS_GRAD_TOL = 1e-6

H132_OUTPUT_TENSORS = (
    "adapter_10tok.mm_prior_out.2.weight",
    "adapter_10tok.mm_prior_out.2.bias",
    "adapter_10tok.traj_delta_head.2.weight",
    "adapter_10tok.traj_delta_head.2.bias",
    "adapter_10tok.parent_delta_net.3.weight",
    "adapter_10tok.parent_delta_net.3.bias",
)
ACTIVE_RANKING_WEIGHT_TENSORS = (
    "adapter_10tok.mm_prior_out.2.weight",
    "adapter_10tok.traj_delta_head.2.weight",
)
H132_OUTPUT_TOTAL_PARAMS = 99
H132_OUTPUT_WEIGHT_PARAMS = 96
H132_OUTPUT_BIAS_PARAMS = 3
ACTIVE_RANKING_WEIGHT_PARAMS = 48


def stream_seed(*parts):
    raw = ":".join(str(x) for x in parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def folds(rows, seed=FOLD_SEED):
    fams = sorted({r["family_id"] for r in rows})
    if len(fams) != 205:
        raise ValueError(f"expected 205 families, got {len(fams)}")
    perm = list(np.random.default_rng(seed).permutation(fams))
    return [sorted(perm[i * 41 : (i + 1) * 41]) for i in range(5)]


def partition(rows, outer, groups=None):
    """Partition 205-family TRAIN into 164-family FIT split (4 folds) and 41-family ASSESS split (1 fold)."""
    groups = folds(rows) if groups is None else groups
    a_set = set(groups[outer])
    f_set = {f for idx, g in enumerate(groups) if idx != outer for f in g}
    assert len(f_set) == 164 and len(a_set) == 41
    assert not (f_set & a_set)
    fit = [r for r in rows if r["family_id"] in f_set]
    assess = [r for r in rows if r["family_id"] in a_set]
    assert len(fit) + len(assess) == len(rows)
    return fit, assess


def inner_cv_folds(fit_rows, seed, outer, n_inner=5):
    """Deterministically partition the 164 fit-split families into 5 disjoint inner family folds."""
    fit_fams = sorted({r["family_id"] for r in fit_rows})
    if len(fit_fams) != 164:
        raise ValueError(f"expected 164 fit families, got {len(fit_fams)}")
    rng = np.random.default_rng(stream_seed("inner_cv", seed, outer))
    perm = list(rng.permutation(fit_fams))
    splits = [sorted(list(chunk)) for chunk in np.array_split(perm, n_inner)]
    result = []
    seen_val = set()
    for m in range(n_inner):
        val_set = set(splits[m])
        train_set = set(fit_fams) - val_set
        assert not (train_set & val_set)
        seen_val |= val_set
        itr = [r for r in fit_rows if r["family_id"] in train_set]
        iva = [r for r in fit_rows if r["family_id"] in val_set]
        assert len(itr) + len(iva) == len(fit_rows)
        result.append((itr, iva, sorted(val_set)))
    assert seen_val == set(fit_fams)
    return result


def family_bootstrap_context_weights(fit_rows, seed, outer, bag_idx):
    """Draw a family-cluster bootstrap resample of fit_rows families and return normalized context weights."""
    fit_fams = sorted({r["family_id"] for r in fit_rows})
    rng = np.random.default_rng(stream_seed("bag_families", seed, outer, bag_idx))
    drawn = rng.choice(len(fit_fams), size=len(fit_fams), replace=True)
    counts = np.bincount(drawn, minlength=len(fit_fams))
    fam_mult = {fam: int(counts[i]) for i, fam in enumerate(fit_fams)}
    raw_w = np.asarray(
        [float(fam_mult[r["family_id"]]) for r in fit_rows], dtype=np.float64
    )
    total = float(raw_w.sum())
    if total <= 0.0:
        raise RuntimeError("empty bootstrap weight sum")
    return raw_w / total, fam_mult


def permute_targets_within_family(fitness, gate, seed, outer, cid):
    """Permute candidate targets (fitness, gate) within a fit-split context (hence within its family)."""
    f_arr = np.asarray(fitness, dtype=np.float32)
    g_arr = np.asarray(gate, dtype=bool)
    if f_arr.shape != g_arr.shape or f_arr.ndim != 1:
        raise ValueError("1D matching fitness and gate arrays required")
    rng = np.random.default_rng(stream_seed("shuf_placebo", seed, outer, cid))
    perm = rng.permutation(f_arr.shape[0])
    return f_arr[perm].copy(), g_arr[perm].copy(), perm


def family_vector(rows, values):
    buckets = {}
    for row, v in zip(rows, values, strict=True):
        buckets.setdefault(row["family_id"], []).append(float(v))
    fams = sorted(buckets)
    return fams, np.asarray([np.mean(buckets[f]) for f in fams], dtype=np.float64)


def equal_family_mean(rows, values):
    return float(family_vector(rows, values)[1].mean())


def bootstrap_family(values, draws, seed):
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    idx = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    means = values[idx].mean(axis=1)
    return dict(
        delta=float(values.mean()),
        ci_low_95=float(np.quantile(means, 0.025)),
        ci_high_95=float(np.quantile(means, 0.975)),
        p_one_sided=float((1 + np.sum(means <= 0.0)) / (draws + 1)),
        draws=int(draws),
    )


def bootstrap_median_of_seed_gains(fam_by_seed, draws, seed):
    """Shared-family bootstrap of the median over seeds of per-seed equal-family mean gains."""
    mat = np.asarray(fam_by_seed, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("expected 2D array (n_seeds, n_families)")
    per_seed = mat.mean(axis=1)
    point = float(np.median(per_seed))
    n_fams = mat.shape[1]
    idx = np.random.default_rng(seed).integers(0, n_fams, size=(draws, n_fams))
    boot_seed_means = mat[:, idx].mean(axis=2)  # (n_seeds, draws)
    boot_medians = np.median(boot_seed_means, axis=0)  # (draws,)
    return dict(
        delta=point,
        ci_low_95=float(np.quantile(boot_medians, 0.025)),
        ci_high_95=float(np.quantile(boot_medians, 0.975)),
        p_one_sided=float((1 + np.sum(boot_medians <= 0.0)) / (draws + 1)),
        draws=int(draws),
        per_seed_equal_family_delta=[float(x) for x in per_seed],
        positive_seeds=int(np.sum(per_seed > 0.0)),
    )


def select_lambda(baseline_values, candidate_values_by_label, fit_rows, seed, outer):
    """Select lambda on the 164-family fit split using strict, unconstrained, and finite-best rules."""
    base_fams, base_vec = family_vector(fit_rows, baseline_values)
    candidates = {
        "inf": dict(
            delta=0.0,
            ci_low_95=0.0,
            ci_high_95=0.0,
            p_one_sided=1.0,
            draws=1000,
            eligible_strict=False,
            eligible_unconstrained=False,
        )
    }
    best_strict_label, best_strict_gain = "inf", 0.0
    best_uncon_label, best_uncon_gain = "inf", 0.0
    best_finite_label, best_finite_gain = None, -math.inf

    for idx, label in enumerate(FINITE_LAMBDA_LABELS):
        fams, cand_vec = family_vector(fit_rows, candidate_values_by_label[label])
        assert fams == base_fams
        stats = bootstrap_family(
            cand_vec - base_vec,
            draws=1000,
            seed=stream_seed("inner_boot", seed, outer, idx),
        )
        eligible_strict = stats["delta"] > 0.001 + 1e-12 and stats["ci_low_95"] > 0.0
        eligible_uncon = stats["delta"] > 1e-12
        candidates[label] = dict(
            **stats,
            eligible_strict=bool(eligible_strict),
            eligible_unconstrained=bool(eligible_uncon),
        )
        if eligible_strict and stats["delta"] > best_strict_gain + 1e-12:
            best_strict_label, best_strict_gain = label, stats["delta"]
        if eligible_uncon and stats["delta"] > best_uncon_gain + 1e-12:
            best_uncon_label, best_uncon_gain = label, stats["delta"]
        if stats["delta"] > best_finite_gain + 1e-12:
            best_finite_label, best_finite_gain = label, stats["delta"]

    return dict(
        labels=list(LAMBDA_LABELS),
        selected_lambda=best_strict_label,
        selected_lambda_value=LAMBDA_BY_LABEL[best_strict_label],
        selected_gain=float(best_strict_gain),
        unconstrained_selected_lambda=best_uncon_label,
        unconstrained_selected_lambda_value=LAMBDA_BY_LABEL[best_uncon_label],
        unconstrained_selected_gain=float(best_uncon_gain),
        best_finite_lambda=best_finite_label,
        best_finite_lambda_value=LAMBDA_BY_LABEL[best_finite_label],
        best_finite_gain=float(best_finite_gain),
        candidates=candidates,
    )


def holm(p_by_arm):
    ordered = sorted(p_by_arm.items(), key=lambda kv: kv[1])
    m = len(ordered)
    running = 0.0
    out = {}
    for i, (arm, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * float(p)))
        out[arm] = running
    return out


def _mm_bradley_terry_batch(win_matrices, max_iter=400, tol=1e-10, prior=0.01):
    """Vectorized canonical H145/H148 MM Bradley-Terry solver over batch of win matrices (..., K, K)."""
    w = np.asarray(win_matrices, dtype=np.float64).copy()
    orig_shape = w.shape[:-2]
    k = w.shape[-1]
    w = w.reshape(-1, k, k)
    b = w.shape[0]
    off = ~np.eye(k, dtype=bool)
    w[:, off] += prior
    n_ij = w + np.swapaxes(w, 1, 2)
    diag_n = np.diagonal(n_ij, axis1=1, axis2=2)
    wins_i = w.sum(axis=2)
    strength = np.ones((b, k), dtype=np.float64)
    for _ in range(max_iter):
        denom = (n_ij / (strength[:, :, None] + strength[:, None, :])).sum(axis=2) - (
            diag_n / (2.0 * strength)
        )
        nxt = np.where(denom > 0.0, wins_i / denom, strength)
        nxt = np.clip(nxt / np.mean(nxt, axis=1, keepdims=True), 1e-6, 1e6)
        if np.max(np.abs(nxt - strength)) < tol:
            strength = nxt
            break
        strength = nxt
    elo = 1500.0 + 400.0 * np.log10(strength / strength[:, :1])
    return elo.reshape(*orig_shape, k)


def _mm_bradley_terry(win_matrix, max_iter=400, tol=1e-10, prior=0.01):
    """Canonical H145/H148 MM Bradley-Terry solver anchored at arm 0 (R882) = 1500.0."""
    return _mm_bradley_terry_batch(win_matrix, max_iter=max_iter, tol=tol, prior=prior)


def _family_win_tables(rows, seed_panel, tie_margin=1e-4):
    """Per-family pairwise win tables (F, K, K) for a single seed panel (C, K)."""
    k = seed_panel.shape[1]
    by_fam = {}
    for idx, r in enumerate(rows):
        by_fam.setdefault(r["family_id"], []).append(idx)
    fams = sorted(by_fam)
    fam_wins = np.zeros((len(fams), k, k), dtype=np.float64)
    for fi, fam in enumerate(fams):
        sub = seed_panel[by_fam[fam]]  # (n_c, k)
        diff = sub[:, :, None] - sub[:, None, :]
        wins = (diff > tie_margin).astype(np.float64) + 0.5 * (
            np.abs(diff) <= tie_margin
        ).astype(np.float64)
        np.fill_diagonal(wins[0], 0.0)
        w = wins.mean(axis=0)
        np.fill_diagonal(w, 0.0)
        fam_wins[fi] = w
    return fam_wins


def elo_ratings(rows, panel, draws=1000, seed=BOOT_SEED, arm_names=ALL_ARMS):
    """Canonical H145/H148 400-step MM Bradley-Terry Elo anchored at R882 = 1500.0."""
    panel = np.asarray(panel, dtype=np.float64)
    n_seeds = panel.shape[0]
    fam_tables = np.stack(
        [_family_win_tables(rows, panel[si]) for si in range(n_seeds)], axis=0
    )  # (S, F, K, K)
    seed_elos = _mm_bradley_terry_batch(fam_tables.sum(axis=1))  # (S, K)
    point = np.median(seed_elos, axis=0)

    n_fams = fam_tables.shape[1]
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, n_fams, size=(draws, n_fams))  # (D, F)
    # Sum resampled family win tables per (draw, seed): (S, D, K, K)
    boot_win_mats = fam_tables[:, ix].sum(axis=2)  # (S, D, K, K)
    boot_seed_elos = _mm_bradley_terry_batch(boot_win_mats)  # (S, D, K)
    boot = np.median(boot_seed_elos, axis=0)  # (D, K)
    return {
        arm: dict(
            elo=float(point[i]),
            elo_by_seed=[float(seed_elos[si, i]) for si in range(n_seeds)],
            elo_delta_vs_r882=float(point[i] - 1500.0),
            ci_low_95=float(np.quantile(boot[:, i], 0.025)),
            ci_high_95=float(np.quantile(boot[:, i], 0.975)),
        )
        for i, arm in enumerate(arm_names)
    }


def summarize(rows, panel, minus2_counts=None, drift_by_arm=None, seed=BOOT_SEED):
    panel = np.asarray(panel, dtype=np.float64)
    n_seeds = panel.shape[0]
    fams = None
    fam_panel = []
    for si in range(n_seeds):
        seed_cols = []
        for ai in range(len(ALL_ARMS)):
            f_ids, vec = family_vector(rows, panel[si, :, ai])
            if fams is None:
                fams = f_ids
            else:
                assert fams == f_ids
            seed_cols.append(vec)
        fam_panel.append(np.stack(seed_cols, axis=-1))
    fam_panel = np.stack(fam_panel, axis=0)  # (n_seeds, n_fams, n_arms)

    r882_fam = fam_panel[:, :, 0]
    ctrl_fam = fam_panel[:, :, 1]
    cv_fam = fam_panel[:, :, 2]
    shuf_fam = fam_panel[:, :, 4]
    elo = elo_ratings(rows, panel, draws=1000, seed=seed + 17)

    arms = {}
    p_raw_all4 = {}
    for idx, arm in enumerate(ARMS, 1):
        arm_fam = fam_panel[:, :, idx]
        stats = bootstrap_median_of_seed_gains(
            arm_fam - r882_fam, draws=10000, seed=seed + idx
        )
        vs_ctrl = bootstrap_median_of_seed_gains(
            arm_fam - ctrl_fam, draws=10000, seed=seed + 100 + idx
        )
        vs_cv = bootstrap_median_of_seed_gains(
            arm_fam - cv_fam, draws=10000, seed=seed + 200 + idx
        )
        vs_shuf = bootstrap_median_of_seed_gains(
            arm_fam - shuf_fam, draws=10000, seed=seed + 300 + idx
        )
        per_seed_raw_means = panel[:, :, idx].mean(axis=1)
        per_seed_fam_means = arm_fam.mean(axis=1)
        entry = dict(
            **stats,
            raw_context_mean=float(np.median(per_seed_raw_means)),
            equal_family_mean=float(np.median(per_seed_fam_means)),
            per_seed_equal_family_mean=[float(x) for x in per_seed_fam_means],
            elo=elo[arm]["elo"],
            elo_by_seed=elo[arm]["elo_by_seed"],
            elo_delta_vs_r882=elo[arm]["elo_delta_vs_r882"],
            elo_ci_low_95=elo[arm]["ci_low_95"],
            elo_ci_high_95=elo[arm]["ci_high_95"],
            delta_vs_ctrl=vs_ctrl["delta"],
            ci_low_95_vs_ctrl=vs_ctrl["ci_low_95"],
            ci_high_95_vs_ctrl=vs_ctrl["ci_high_95"],
            p_one_sided_vs_ctrl=vs_ctrl["p_one_sided"],
            per_seed_delta_vs_ctrl=vs_ctrl["per_seed_equal_family_delta"],
            delta_vs_refit_cv=vs_cv["delta"],
            ci_low_95_vs_refit_cv=vs_cv["ci_low_95"],
            ci_high_95_vs_refit_cv=vs_cv["ci_high_95"],
            p_one_sided_vs_refit_cv=vs_cv["p_one_sided"],
            per_seed_delta_vs_refit_cv=vs_cv["per_seed_equal_family_delta"],
            delta_vs_refit_shuf=vs_shuf["delta"],
            ci_low_95_vs_refit_shuf=vs_shuf["ci_low_95"],
            ci_high_95_vs_refit_shuf=vs_shuf["ci_high_95"],
            p_one_sided_vs_refit_shuf=vs_shuf["p_one_sided"],
            per_seed_delta_vs_refit_shuf=vs_shuf["per_seed_equal_family_delta"],
        )
        if minus2_counts is not None and arm in minus2_counts:
            entry["minus2_invalid_counts"] = minus2_counts[arm]
        if drift_by_arm is not None and arm in drift_by_arm:
            entry["head_l2_drift"] = drift_by_arm[arm]
        arms[arm] = entry
        p_raw_all4[arm] = stats["p_one_sided"]

    adjusted_all4 = holm(p_raw_all4)
    adjusted_treat2 = holm({a: p_raw_all4[a] for a in TREATMENT_ARMS})
    adjusted_3 = holm(
        {a: p_raw_all4[a] for a in ("REFIT_CV", "REFIT_BAG", "REFIT_SHUF")}
    )

    shuf_entry = arms["REFIT_SHUF"]
    placebo_pass = bool(
        shuf_entry["delta"] > 0.001 + 1e-12
        and shuf_entry["ci_low_95"] > 0.0
        and shuf_entry["p_one_sided"] < 0.05
    )

    for arm, p_adj4 in adjusted_all4.items():
        arms[arm]["p_holm_all4"] = p_adj4
        arms[arm]["p_holm_3"] = adjusted_3.get(arm, p_adj4)
        arms[arm]["p_holm"] = adjusted_treat2.get(arm, p_adj4)
        arms[arm]["advance"] = bool(
            arm in TREATMENT_ARMS
            and not placebo_pass
            and arms[arm]["delta"] > 0.001 + 1e-12
            and arms[arm]["ci_low_95"] > 0.0
            and arms[arm]["p_holm"] < 0.05
        )

    out = dict(
        family_count=len(fams),
        context_count=int(panel.shape[1]),
        r882_raw_context_mean=float(np.median(panel[:, :, 0].mean(axis=1))),
        r882_equal_family_mean=float(np.median(r882_fam.mean(axis=1))),
        r882_elo=1500.0,
        placebo_advance_leak_flag=placebo_pass,
        arms=arms,
    )
    if minus2_counts is not None and "R882" in minus2_counts:
        out["r882_minus2_invalid_counts"] = minus2_counts["R882"]
    return out


def pseudoreplication_report(cell_records):
    """Check whether the 3 seeds produce identical selected lambdas, head weights, or context choices per outer fold."""
    by_outer = {}
    for rec in cell_records:
        by_outer.setdefault(rec["outer_fold"], []).append(rec)
    report = {}
    for arm in ("REFIT_CV", "REFIT_BAG", "REFIT_SHUF"):
        same_lambda_folds = 0
        same_weights_folds = 0
        same_choices_folds = 0
        same_uncon_choices_folds = 0
        same_finite_choices_folds = 0
        fold_details = {}
        for outer, recs in sorted(by_outer.items()):
            recs = sorted(recs, key=lambda r: r["seed"])
            lams = [r["decisions"][arm]["selected_lambda"] for r in recs]
            w_shas = [r["head_sha256"][arm]["strict"] for r in recs]
            ch_strict = [tuple(r["assess_choices"][arm]["strict"]) for r in recs]
            ch_uncon = [tuple(r["assess_choices"][arm]["unconstrained"]) for r in recs]
            ch_finite = [tuple(r["assess_choices"][arm]["best_finite"]) for r in recs]
            sl = len(set(lams)) == 1
            sw = len(set(w_shas)) == 1
            sc = len(set(ch_strict)) == 1
            su = len(set(ch_uncon)) == 1
            sf = len(set(ch_finite)) == 1
            same_lambda_folds += int(sl)
            same_weights_folds += int(sw)
            same_choices_folds += int(sc)
            same_uncon_choices_folds += int(su)
            same_finite_choices_folds += int(sf)
            fold_details[f"A{outer}"] = dict(
                selected_lambdas=lams,
                unique_selected_lambdas=len(set(lams)),
                unique_strict_weight_shas=len(set(w_shas)),
                unique_strict_choice_vectors=len(set(ch_strict)),
                unique_unconstrained_choice_vectors=len(set(ch_uncon)),
                unique_best_finite_choice_vectors=len(set(ch_finite)),
            )
        report[arm] = dict(
            outer_folds_total=len(by_outer),
            identical_selected_lambda_folds=same_lambda_folds,
            identical_strict_weights_folds=same_weights_folds,
            identical_strict_choices_folds=same_choices_folds,
            identical_unconstrained_choices_folds=same_uncon_choices_folds,
            identical_best_finite_choices_folds=same_finite_choices_folds,
            all_seeds_identical_strict_choices=bool(
                same_choices_folds == len(by_outer)
            ),
            all_seeds_identical_best_finite_choices=bool(
                same_finite_choices_folds == len(by_outer)
            ),
            by_outer_fold=fold_details,
        )
    return report
