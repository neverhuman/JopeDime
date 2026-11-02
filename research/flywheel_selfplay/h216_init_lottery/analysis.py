"""H216 random-init lottery with B-locked strict-margin selection and paired-family inference."""

import hashlib
import numpy as np
from parents import numerical

BASE = numerical()
SEEDS = (216101, 216102, 216103)
CANDIDATE_COUNT = 16
FOLD_SEED = 183140
BOOT_SEED = 216141
ARMS = ("R882", "LOTTERY16", "PLACEBO", "ORACLE_A")
PRIMARY_ARM = "LOTTERY16"
HOLM_CONTRASTS = ("LOTTERY16_minus_R882", "LOTTERY16_minus_PLACEBO")
DRAWS = 10_000
B_MARGIN = 0.005
B_QUANTILE = 0.10
HARM_FAMILY_THRESHOLD = -0.01
HARM_FRACTION_CAP = 0.10

stream_seed = BASE.stream_seed
context_utility = BASE.context_utility
family_win_matrix = BASE.family_win_matrix
bt_ratings = BASE.bt_ratings
holm_adjust = BASE.holm_adjust
paired_family_bootstrap = BASE.paired_family_bootstrap


def family_folds(family_ids):
    return BASE.family_folds(family_ids, seed=FOLD_SEED)


def partition(family_ids, outer, folds=None):
    return BASE.partition(
        family_ids, outer, family_folds(family_ids) if folds is None else folds
    )


def candidate_seeds(seed):
    if seed not in SEEDS:
        raise ValueError(f"unknown H216 replicate seed: {seed}")
    offset = 216200 + 100 * SEEDS.index(seed)
    return tuple(offset + i for i in range(1, CANDIDATE_COUNT + 1))


def all_candidate_seeds():
    return tuple(s for seed in SEEDS for s in candidate_seeds(seed))


def validate_candidate_init_hashes(init_hashes, reference_hash):
    expected_seeds = all_candidate_seeds()
    if tuple(init_hashes.keys()) != expected_seeds:
        raise ValueError("candidate init hash inventory must match all 48 H216 seeds")
    values = list(init_hashes.values())
    if (
        len(values) != len(expected_seeds)
        or len(set(values)) != len(expected_seeds)
        or any(
            not isinstance(v, str) or len(v) != 64 or v == reference_hash
            for v in values
        )
    ):
        raise ValueError("48 distinct non-reference constructor init hashes required")
    return True


def family_equal_vector(rows, context_utilities):
    fams = sorted({r["family_id"] for r in rows})
    if not fams:
        raise ValueError("nonempty rows required")
    means = []
    for fam in fams:
        vals = [
            float(context_utilities[r["context_sha256"]])
            for r in rows
            if r["family_id"] == fam
        ]
        if not vals or not np.isfinite(vals).all():
            raise ValueError("finite context utilities required for every family")
        means.append(float(np.mean(vals)))
    return fams, np.asarray(means, dtype=np.float64)


def _select_from_delta_matrix(
    delta_b,
    seed,
    outer,
    *,
    draws=DRAWS,
    margin=B_MARGIN,
    quantile=B_QUANTILE,
    role_tag="LOTTERY16",
):
    delta = np.asarray(delta_b, dtype=np.float64)
    if (
        delta.ndim != 2
        or delta.shape[0] != CANDIDATE_COUNT
        or delta.shape[1] < 1
        or not np.isfinite(delta).all()
    ):
        raise ValueError("expected finite (16, n_B_families) gain matrix")
    n_fam = delta.shape[1]
    boot_rng = np.random.default_rng(stream_seed("H216", "B_BOOT", seed, outer))
    ix_b = boot_rng.integers(0, n_fam, size=(draws, n_fam))
    candidates = []
    eligible = []
    for k in range(CANDIDATE_COUNT):
        boot_gains = delta[k, ix_b].mean(axis=1)
        point_gain = float(delta[k].mean())
        boot_med = float(np.median(boot_gains))
        q10 = float(np.quantile(boot_gains, quantile))
        ci95 = np.quantile(boot_gains, [0.025, 0.975]).tolist()
        is_ok = bool(boot_med > margin and q10 > margin)
        rec = dict(
            candidate_index=k + 1,
            init_seed=candidate_seeds(seed)[k],
            b_point_family_gain=point_gain,
            b_boot_median_gain=boot_med,
            b_boot_q10_gain=q10,
            b_boot_ci95=ci95,
            eligible=is_ok,
        )
        candidates.append(rec)
        if is_ok:
            eligible.append(rec)
    if eligible:
        chosen = max(
            eligible,
            key=lambda r: (
                r["b_boot_median_gain"],
                r["b_boot_q10_gain"],
                -r["candidate_index"],
            ),
        )
        selected_index = int(chosen["candidate_index"])
        selected_label = f"CAND{selected_index}"
        fallback = False
    else:
        selected_index = 0
        selected_label = "R882"
        fallback = True
    return dict(
        role=role_tag,
        seed=int(seed),
        outer_fold=int(outer),
        b_family_count=int(n_fam),
        b_bootstrap_draws=int(draws),
        margin=float(margin),
        quantile=float(quantile),
        eligible_count=len(eligible),
        selected_candidate_index=selected_index,
        selected_label=selected_label,
        fallback_to_r882=fallback,
        candidates=candidates,
    )


def select_on_b(
    r882_b_fam,
    cand_b_fam,
    seed,
    outer,
    *,
    draws=DRAWS,
    margin=B_MARGIN,
    quantile=B_QUANTILE,
):
    base = np.asarray(r882_b_fam, dtype=np.float64)
    cand = np.asarray(cand_b_fam, dtype=np.float64)
    if (
        base.ndim != 1
        or cand.shape != (CANDIDATE_COUNT, base.shape[0])
        or not np.isfinite(base).all()
        or not np.isfinite(cand).all()
    ):
        raise ValueError("invalid B-split family utility arrays")
    delta_b = cand - base[None, :]
    return _select_from_delta_matrix(
        delta_b,
        seed,
        outer,
        draws=draws,
        margin=margin,
        quantile=quantile,
        role_tag="LOTTERY16",
    )


def shuffle_b_family_gains(delta_b, seed, outer):
    delta = np.asarray(delta_b, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[0] != CANDIDATE_COUNT:
        raise ValueError("expected (16, n_B_families) gain matrix")
    rng = np.random.default_rng(stream_seed("H216", "PLACEBO_SHUFFLE", seed, outer))
    n_fam = delta.shape[1]
    shuffled = np.empty_like(delta)
    for k in range(CANDIDATE_COUNT):
        fam_perm = rng.permutation(n_fam)
        shuffled[k] = delta[k, fam_perm]
    for f in range(n_fam):
        cand_perm = rng.permutation(CANDIDATE_COUNT)
        shuffled[:, f] = shuffled[cand_perm, f]
    return shuffled


def select_placebo_on_b(
    r882_b_fam,
    cand_b_fam,
    seed,
    outer,
    *,
    draws=DRAWS,
    margin=B_MARGIN,
    quantile=B_QUANTILE,
):
    base = np.asarray(r882_b_fam, dtype=np.float64)
    cand = np.asarray(cand_b_fam, dtype=np.float64)
    if (
        base.ndim != 1
        or cand.shape != (CANDIDATE_COUNT, base.shape[0])
        or not np.isfinite(base).all()
        or not np.isfinite(cand).all()
    ):
        raise ValueError("invalid B-split family utility arrays")
    delta_b = cand - base[None, :]
    shuffled_delta = shuffle_b_family_gains(delta_b, seed, outer)
    out = _select_from_delta_matrix(
        shuffled_delta,
        seed,
        outer,
        draws=draws,
        margin=margin,
        quantile=quantile,
        role_tag="PLACEBO",
    )
    out["b_family_shuffled"] = True
    out["shuffle_digest_sha256"] = hashlib.sha256(shuffled_delta.tobytes()).hexdigest()
    return out


def select_oracle_on_a(r882_a_fam, cand_a_fam, seed, outer):
    base = np.asarray(r882_a_fam, dtype=np.float64)
    cand = np.asarray(cand_a_fam, dtype=np.float64)
    if (
        base.ndim != 1
        or cand.shape != (CANDIDATE_COUNT, base.shape[0])
        or not np.isfinite(base).all()
        or not np.isfinite(cand).all()
    ):
        raise ValueError("invalid A-split family utility arrays")
    base_mean = float(base.mean())
    cand_means = cand.mean(axis=1)
    best_k = int(np.argmax(cand_means))
    best_cand_mean = float(cand_means[best_k])
    if best_cand_mean > base_mean:
        selected_index = best_k + 1
        selected_label = f"CAND{selected_index}"
        best_gain = best_cand_mean - base_mean
    else:
        selected_index = 0
        selected_label = "R882"
        best_gain = 0.0
    return dict(
        role="ORACLE_A",
        diagnostic_only=True,
        cannot_win=True,
        seed=int(seed),
        outer_fold=int(outer),
        a_family_count=int(base.shape[0]),
        selected_candidate_index=selected_index,
        selected_label=selected_label,
        oracle_a_family_gain=float(best_gain),
        candidate_a_family_gains=(cand_means - base_mean).tolist(),
    )


def _ci(x):
    return np.quantile(x, [0.025, 0.975], axis=0).tolist()


def summarize(context_panel, family_ids, b_selections=None, draws=DRAWS):
    panel = np.asarray(context_panel, dtype=np.float64)
    if (
        panel.shape != (len(SEEDS), len(family_ids), len(ARMS))
        or not np.isfinite(panel).all()
    ):
        raise ValueError("complete three-seed x context x four-arm panel required")
    fams = sorted(set(family_ids))
    n_fams = len(fams)
    masks = [np.asarray(family_ids) == f for f in fams]
    fp = np.stack([panel[:, mask, :].mean(axis=1) for mask in masks], axis=1)
    ix = np.random.default_rng(BOOT_SEED).integers(0, n_fams, (draws, n_fams))
    out = dict(
        arms={},
        gain_vs_r882={},
        contrasts={},
        elo_vs_r882={},
        assessed_contexts=len(family_ids),
        assessed_families=n_fams,
        family_bootstrap_draws=draws,
        seed_ids=list(SEEDS),
        arm_names=list(ARMS),
        primary_arm=PRIMARY_ARM,
        holm_family=list(HOLM_CONTRASTS),
    )
    for j, arm in enumerate(ARMS):
        mean = fp[:, :, j].mean(axis=1)
        boot = np.median(fp[:, ix, j].mean(axis=2), axis=0)
        out["arms"][arm] = dict(
            per_seed_equal_family_utility=mean.tolist(),
            median_over_seeds=float(np.median(mean)),
            ci95=_ci(boot),
            minus2_count_by_seed=(panel[:, :, j] == -2.0).sum(axis=1).tolist(),
            diagnostic_only=(arm == "ORACLE_A"),
        )
    pairs = [
        ("LOTTERY16_minus_R882", "LOTTERY16", "R882"),
        ("PLACEBO_minus_R882", "PLACEBO", "R882"),
        ("ORACLE_A_minus_R882", "ORACLE_A", "R882"),
        ("LOTTERY16_minus_PLACEBO", "LOTTERY16", "PLACEBO"),
        ("ORACLE_A_minus_LOTTERY16", "ORACLE_A", "LOTTERY16"),
    ]
    for name, a, b in pairs:
        delta = fp[:, :, ARMS.index(a)] - fp[:, :, ARMS.index(b)]
        per_seed = delta.mean(axis=1)
        point = float(np.median(per_seed))
        boot = paired_family_bootstrap(delta, ix)
        fam_mean_delta = delta.mean(axis=0)
        harmed_count = int((fam_mean_delta < HARM_FAMILY_THRESHOLD).sum())
        strict_neg_count = int((fam_mean_delta < 0.0).sum())
        p_direct = float((1 + (boot <= 0.0).sum()) / (draws + 1))
        p_centered = float((1 + (boot - point >= point).sum()) / (draws + 1))
        record = dict(
            per_seed=per_seed.tolist(),
            median_over_seeds=point,
            ci95=_ci(boot),
            p_one_sided_direct=p_direct,
            p_one_sided_centered=p_centered,
            p_one_sided=max(p_direct, p_centered),
            harmed_families_count=harmed_count,
            harmed_families_fraction=float(harmed_count / n_fams),
            strict_negative_families_count=strict_neg_count,
            strict_negative_families_fraction=float(strict_neg_count / n_fams),
            positive_seeds=int((per_seed > 0.0).sum()),
            diagnostic_only=(a == "ORACLE_A" or b == "ORACLE_A"),
        )
        out["contrasts"][name] = record
        if b == "R882":
            out["gain_vs_r882"][a] = record
    raw_holm_ps = [out["contrasts"][c]["p_one_sided"] for c in HOLM_CONTRASTS]
    adj_ps = holm_adjust(raw_holm_ps)
    for c, adj_p in zip(HOLM_CONTRASTS, adj_ps, strict=True):
        out["contrasts"][c]["holm_m2_adjusted_p"] = float(adj_p)
    out["gain_vs_r882"]["LOTTERY16"]["holm_m2_adjusted_p"] = out["contrasts"][
        "LOTTERY16_minus_R882"
    ]["holm_m2_adjusted_p"]
    wins = np.stack(
        [family_win_matrix(panel[s], family_ids) for s in range(len(SEEDS))]
    )
    ratings = bt_ratings(wins.sum(axis=1))
    elo_boot = []
    for start in range(0, draws, 100):
        resampled = np.take(wins, ix[start : start + 100], axis=1).sum(axis=2)
        elo_boot.append(np.median(bt_ratings(resampled), axis=0))
    elo_ci = _ci(np.concatenate(elo_boot, axis=0))
    for j, a in enumerate(ARMS):
        out["elo_vs_r882"][a] = dict(
            per_seed=ratings[:, j].tolist(),
            median_over_seeds=float(np.median(ratings[:, j])),
            ci95=[elo_ci[0][j], elo_ci[1][j]],
        )
    if b_selections is not None:
        out["b_selection_summary"] = b_selections
    return decide(out)


def decide(summary):
    primary = summary["gain_vs_r882"]["LOTTERY16"]
    vs_placebo = summary["contrasts"]["LOTTERY16_minus_PLACEBO"]
    advance = bool(
        primary["median_over_seeds"] > 0.0
        and primary["ci95"][0] > 0.0
        and primary["p_one_sided"] < 0.05
        and primary["holm_m2_adjusted_p"] < 0.05
        and primary["harmed_families_fraction"] <= HARM_FRACTION_CAP
    )
    primary["advance_candidate"] = advance
    primary["beats_placebo_holm_m2"] = bool(
        vs_placebo["median_over_seeds"] > 0.0
        and vs_placebo["ci95"][0] > 0.0
        and vs_placebo["holm_m2_adjusted_p"] < 0.05
    )
    summary["gain_vs_r882"]["PLACEBO"]["advance_candidate"] = False
    summary["gain_vs_r882"]["ORACLE_A"]["advance_candidate"] = False
    summary["verdict"] = "CANDIDATE" if advance else "NO_ADVANCE"
    summary["serving_promotion"] = False
    summary["promotion_requires_H177_H178_qualification"] = True
    return summary
