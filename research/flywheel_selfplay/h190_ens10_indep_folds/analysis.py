"""H190 ENS10 confirmation with independent family-fold partitions per outer seed (190141, 190142, 190143).

Imports the exact H177 pooled cross-fit decision rule (`H177_DECISION_RULE`, `pooled_b_select`, `StageOrderGuard`,
`verify_pooled_b_coverage`, `resolve_h177_verdict`, `holm`, `family_matrix`, `bt_ratings`, `family_wins`, `DRAWS`, `EPS`)
directly from `research.flywheel_selfplay.h177_pooled_selector.analysis` without reimplementing the selector math.
"""

import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _load_sibling_module(mod_name, rel_path):
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    target = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


h177_analysis = _load_sibling_module(
    "h177_pooled_selector_analysis",
    "research/flywheel_selfplay/h177_pooled_selector/analysis.py",
)

H177_DECISION_RULE = h177_analysis.H177_DECISION_RULE
pooled_b_select = h177_analysis.pooled_b_select
resolve_h177_verdict = h177_analysis.resolve_h177_verdict
holm = h177_analysis.holm
family_matrix = h177_analysis.family_matrix
bt_ratings = h177_analysis.bt_ratings
family_wins = h177_analysis.family_wins
DRAWS = h177_analysis.DRAWS
EPS = h177_analysis.EPS

SEEDS = (190101, 190102, 190103)
FOLD_SEEDS = (190141, 190142, 190143)
FOLD_SEED_BY_SEED = {
    190101: 190141,
    190102: 190142,
    190103: 190143,
}
PRIOR_WAVE_FOLD_SEEDS = (177140, 179140, 182140, 185140)
BOOT_SEED = 190150

DEFAULT_ENSEMBLE_K = 10
FALLBACK_ENSEMBLE_K = 5
MEMBER_UPDATES = 40
RENS_MEMBER_UPDATES = 0
PILOT_UPDATES = 4
WALL_CAP_SECONDS = 9_000
EVAL_RESERVE_SECONDS = 900

MEMBER_SEEDS_BY_SEED = {
    190101: (
        190201,
        190202,
        190203,
        190204,
        190205,
        190206,
        190207,
        190208,
        190209,
        190210,
    ),
    190102: (
        190301,
        190302,
        190303,
        190304,
        190305,
        190306,
        190307,
        190308,
        190309,
        190310,
    ),
    190103: (
        190401,
        190402,
        190403,
        190404,
        190405,
        190406,
        190407,
        190408,
        190409,
        190410,
    ),
}

PRIOR_ENSEMBLE_MEMBER_SEEDS = (
    set(range(179201, 179206))
    | set(range(179301, 179306))
    | set(range(179401, 179406))
    | set(range(182201, 182221))
    | set(range(182301, 182321))
    | set(range(182401, 182421))
    | set(range(185201, 185206))
    | set(range(185301, 185306))
    | set(range(185401, 185406))
)

CANDIDATES = ("ENS10", "RENS10")
ALL_ARMS = ("R882",) + CANDIDATES
HOLM_MULTIPLICITY = 2
MARGIN = 0.001

PRESPECIFIED_CANDIDATE_STEP = {
    "ENS10": MEMBER_UPDATES,
    "RENS10": RENS_MEMBER_UPDATES,
}

H177_SLOT_BY_ARM = {
    "ENS10": "ACC4",
    "RENS10": "ACC16",
}


def stream_seed(*parts):
    return h177_analysis.stream_seed(*parts)


def fold_seed_for(outer_seed):
    if outer_seed not in FOLD_SEED_BY_SEED:
        raise ValueError(f"unknown outer seed: {outer_seed}")
    return FOLD_SEED_BY_SEED[outer_seed]


def member_seeds_for(outer_seed, k=DEFAULT_ENSEMBLE_K):
    if outer_seed not in MEMBER_SEEDS_BY_SEED:
        raise ValueError(f"unknown outer seed: {outer_seed}")
    if k not in (DEFAULT_ENSEMBLE_K, FALLBACK_ENSEMBLE_K):
        raise ValueError(
            f"ensemble k must be {DEFAULT_ENSEMBLE_K} or {FALLBACK_ENSEMBLE_K}"
        )
    return MEMBER_SEEDS_BY_SEED[outer_seed][:k]


def validate_member_seed_disjointness():
    all_h190 = []
    for s in SEEDS:
        all_h190.extend(MEMBER_SEEDS_BY_SEED[s])
    if len(all_h190) != 30 or len(set(all_h190)) != 30:
        raise ValueError("H190 requires 30 mutually distinct member seeds")
    if set(all_h190) & PRIOR_ENSEMBLE_MEMBER_SEEDS:
        raise ValueError("H190 member seeds overlap prior H179/H182/H185 member seeds")
    return True


def validate_member_init_hashes(hashes, k=DEFAULT_ENSEMBLE_K):
    """Require k distinct non-empty constructor-produced adapter hashes for the ensemble members."""
    values = tuple(hashes)
    if (
        k not in (DEFAULT_ENSEMBLE_K, FALLBACK_ENSEMBLE_K)
        or len(values) != k
        or any(not isinstance(v, str) or not v for v in values)
        or len(set(values)) != k
    ):
        raise ValueError(f"{k} distinct fresh-member initialization hashes required")
    return values


def validate_rens_matches_ens_inits(rens_hashes, ens_init_hashes, k=DEFAULT_ENSEMBLE_K):
    """Prove RENS10 scores the exact same k untrained member initializations that ENS10 trains from."""
    r_vals = validate_member_init_hashes(rens_hashes, k=k)
    e_vals = validate_member_init_hashes(ens_init_hashes, k=k)
    if r_vals != e_vals:
        raise ValueError(
            "RENS member initializations do not match ENS member initializations"
        )
    return True


def validate_cell_init_hashes(r882_hash, member_hashes, k=DEFAULT_ENSEMBLE_K):
    """Require r882 and all k ENS/RENS member inits to be mutually distinct."""
    m_vals = validate_member_init_hashes(member_hashes, k=k)
    all_hashes = (r882_hash,) + m_vals
    if (
        any(not isinstance(v, str) or not v for v in all_hashes)
        or len(set(all_hashes)) != k + 1
    ):
        raise ValueError(
            f"r882 and the {k} member initializations must have {k + 1} distinct hashes"
        )
    return all_hashes


def score_ensemble(member_scores, k=DEFAULT_ENSEMBLE_K):
    """Average per-candidate cost tensors across the k ensemble members before taking argmin."""
    scores = np.asarray(member_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != k or not np.isfinite(scores).all():
        raise ValueError(
            f"expected finite ({k}, n_candidates) member score array, got {scores.shape}"
        )
    return scores.mean(axis=0)


def folds(rows, fold_seed):
    fams = sorted({r["family_id"] for r in rows})
    if len(fams) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(int(fold_seed)).permutation(205)
    return [sorted(fams[i] for i in group) for group in np.array_split(order, 5)]


def partition(rows, outer, fold_seed):
    if not 0 <= int(outer) < 5:
        raise ValueError("outer fold must be in 0..4")
    groups = folds(rows, fold_seed=int(fold_seed))
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    return (
        [r for r in rows if r["family_id"] not in a | b],
        [r for r in rows if r["family_id"] in b],
        [r for r in rows if r["family_id"] in a],
    )


def verify_pooled_b_coverage_for_seed(rows, fold_seed):
    """Invoke H177's verify_pooled_b_coverage directly under a specific fold_seed."""
    prev_seed = h177_analysis.FOLD_SEED
    try:
        h177_analysis.FOLD_SEED = int(fold_seed)
        proof = h177_analysis.verify_pooled_b_coverage(rows)
    finally:
        h177_analysis.FOLD_SEED = prev_seed
    return dict(**proof, fold_seed=int(fold_seed))


def verify_independent_fold_partitions(rows):
    """Verify each of the 3 outer seeds has a valid 205-family 3/1/1 partition AND that all 3 partitions differ."""
    if len(set(FOLD_SEEDS)) != 3 or (set(FOLD_SEEDS) & set(PRIOR_WAVE_FOLD_SEEDS)):
        raise ValueError(
            "H190 fold seeds must be 3 distinct seeds disjoint from prior waves"
        )
    by_seed = {}
    partitions_by_seed = {}
    for s in SEEDS:
        fseed = fold_seed_for(s)
        by_seed[str(s)] = verify_pooled_b_coverage_for_seed(rows, fseed)
        partitions_by_seed[s] = folds(rows, fseed)

    # Check that all 3 partitions differ from each other on every fold 0..4 and differ from prior waves
    pairwise_jaccard = {}
    for i, s1 in enumerate(SEEDS):
        for s2 in SEEDS[i + 1 :]:
            g1, g2 = partitions_by_seed[s1], partitions_by_seed[s2]
            if g1 == g2:
                raise ValueError(f"fold partitions for {s1} and {s2} are identical")
            jacc = []
            for outer in range(5):
                set1, set2 = set(g1[outer]), set(g2[outer])
                if set1 == set2:
                    raise ValueError(
                        f"outer fold {outer} is identical between {s1} and {s2}"
                    )
                jacc.append(len(set1 & set2) / float(len(set1 | set2)))
            pairwise_jaccard[f"{s1}_vs_{s2}"] = dict(
                mean_fold_jaccard=float(np.mean(jacc)),
                max_fold_jaccard=float(np.max(jacc)),
                per_fold_jaccard=jacc,
            )

    for prior_fseed in PRIOR_WAVE_FOLD_SEEDS:
        g_prior = folds(rows, prior_fseed)
        for s in SEEDS:
            if partitions_by_seed[s] == g_prior:
                raise ValueError(
                    f"H190 seed {s} partition matches prior fold_seed {prior_fseed}"
                )

    return dict(
        independent_partitions_verified=True,
        fold_seeds_by_seed={str(k): int(v) for k, v in FOLD_SEED_BY_SEED.items()},
        per_seed_coverage=by_seed,
        pairwise_fold_jaccard=pairwise_jaccard,
    )


class StageOrderGuard(h177_analysis.StageOrderGuard):
    """Reuses H177's StageOrderGuard to enforce immutable candidate steps and fsynced pooled_b_selection_lock.json before Stage 2."""

    def __init__(self, expected_steps=None, ensemble_k=DEFAULT_ENSEMBLE_K):
        steps = dict(expected_steps or PRESPECIFIED_CANDIDATE_STEP)
        if set(steps) != set(CANDIDATES):
            raise ValueError(
                f"expected steps for {list(CANDIDATES)}, got {sorted(steps)}"
            )
        if (
            steps["ENS10"] != MEMBER_UPDATES
            or steps["RENS10"] != RENS_MEMBER_UPDATES
            or ensemble_k not in (DEFAULT_ENSEMBLE_K, FALLBACK_ENSEMBLE_K)
        ):
            raise ValueError(
                "invalid prespecified candidate steps or ensemble_k for H190"
            )
        self.expected_steps = steps
        self.ensemble_k = int(ensemble_k)
        self.b_read_started = False
        self.b_locked_path = None

    def mark_b_locked(self, lock_path):
        p = Path(lock_path)
        if not p.is_file():
            raise RuntimeError("pooled_b_selection_lock.json does not exist on disk")
        payload = json.loads(p.read_text(encoding="utf-8"))
        if (
            payload.get("wave") != "H190"
            or payload.get("pooled_b_families") != 205
            or payload.get("pooled_b_contexts") != 1219
            or payload.get("locked_before_a_confirmation") is not True
            or set(payload.get("decisions", {})) != set(CANDIDATES)
            or payload.get("fold_seeds_by_seed")
            != {str(k): int(v) for k, v in FOLD_SEED_BY_SEED.items()}
        ):
            raise RuntimeError("invalid pooled_b_selection_lock.json payload")
        self.b_locked_path = p

    def require_b_locked(self):
        if self.b_locked_path is None or not self.b_locked_path.is_file():
            raise RuntimeError(
                "A-family or DL-Val confirmation forbidden before pooled_b_selection_lock.json is written and fsynced"
            )
        payload = json.loads(self.b_locked_path.read_text(encoding="utf-8"))
        if (
            payload.get("wave") != "H190"
            or payload.get("pooled_b_families") != 205
            or set(payload.get("decisions", {})) != set(CANDIDATES)
        ):
            raise RuntimeError("corrupted pooled_b_selection_lock.json")
        return payload


def batch_plan(cids, orders, seed, updates=MEMBER_UPDATES):
    """H156/H182/H185 FIT-only minibatch sampler: 8 FIT contexts, 4 top-fitness revealed + 20 random unrevealed candidates."""
    cids = sorted(cids)
    if len(cids) < 8 or updates < 1:
        raise ValueError(
            "batch_plan requires at least 8 FIT contexts and positive updates"
        )
    plan = []
    for step in range(1, updates + 1):
        stream = stream_seed(seed, step)
        rng = np.random.default_rng(stream)
        chosen = [cids[i] for i in rng.choice(len(cids), 8, replace=False)]
        indexes = []
        for cid in chosen:
            order = np.asarray(orders[cid], dtype=np.int64)
            if len(order) < 24:
                raise ValueError("at least 24 candidates required per FIT context")
            indexes.append(
                np.concatenate([order[:4], rng.choice(order[4:], 20, replace=False)])
            )
        plan.append((stream, chosen, indexes))
    return plan


def execution_plan(
    seconds_per_update,
    cap_seconds=WALL_CAP_SECONDS,
    reserve_seconds=EVAL_RESERVE_SECONDS,
):
    """Timing-only pilot rule: keep K=10 (ENS10 + RENS10, 6,000 updates) if projected <= 2.5h (9,000s); else cut to K=5 (ENS5 + RENS5, 3,000 updates)."""
    if not math.isfinite(seconds_per_update) or seconds_per_update <= 0.0:
        raise ValueError("positive finite seconds_per_update required")
    total_default = DEFAULT_ENSEMBLE_K * MEMBER_UPDATES * 5 * len(SEEDS)
    proj_default = total_default * float(seconds_per_update) + reserve_seconds
    if proj_default <= cap_seconds:
        return dict(
            outer_folds=5,
            seeds=len(SEEDS),
            active_ensemble_k=DEFAULT_ENSEMBLE_K,
            member_updates=MEMBER_UPDATES,
            rens_member_updates=RENS_MEMBER_UPDATES,
            total_optimizer_updates=total_default,
            projected_seconds=proj_default,
            cap_seconds=cap_seconds,
            fallback_triggered=False,
        )
    total_fallback = FALLBACK_ENSEMBLE_K * MEMBER_UPDATES * 5 * len(SEEDS)
    proj_fallback = total_fallback * float(seconds_per_update) + reserve_seconds
    return dict(
        outer_folds=5,
        seeds=len(SEEDS),
        active_ensemble_k=FALLBACK_ENSEMBLE_K,
        member_updates=MEMBER_UPDATES,
        rens_member_updates=RENS_MEMBER_UPDATES,
        total_optimizer_updates=total_fallback,
        projected_seconds=proj_fallback,
        cap_seconds=cap_seconds,
        fallback_triggered=True,
    )


def pooled_b_select_h190(
    rows,
    r882_values,
    candidate_panels_by_arm,
    seed=BOOT_SEED,
    prespecified_steps=None,
):
    """Delegate Stage-1 205-family pooled-B selection directly to H177_DECISION_RULE.select (`h177_analysis.pooled_b_select`).

    Maps R882 to CTRL/OC15 and (ENS10, RENS10) to (ACC4, ACC16) so `h177_analysis.pooled_b_select` computes the exact
    3-seed within-family mean, 10,000-draw paired family bootstrap, and axis=0 median-seed CI, then applies `h177_analysis.holm`
    across the m=2 registered candidate arms (ENS10, RENS10).
    """
    if not isinstance(candidate_panels_by_arm, dict) or set(
        candidate_panels_by_arm
    ) != set(CANDIDATES):
        raise ValueError(f"exact 2 candidate arms required: {list(CANDIDATES)}")
    steps = dict(prespecified_steps or PRESPECIFIED_CANDIDATE_STEP)
    base_arr = np.asarray(r882_values, dtype=np.float64)
    mapped_panels = {
        "CTRL": base_arr,
        H177_SLOT_BY_ARM["ENS10"]: np.asarray(
            candidate_panels_by_arm["ENS10"], dtype=np.float64
        ),
        H177_SLOT_BY_ARM["RENS10"]: np.asarray(
            candidate_panels_by_arm["RENS10"], dtype=np.float64
        ),
        "OC15": base_arr,
    }
    raw_h177 = H177_DECISION_RULE.select(rows, base_arr, mapped_panels, seed=seed)

    # Also compute one-sided p-value for the median-over-seeds functional from the exact same 10,000 draws
    families = [r["family_id"] for r in rows]
    base_3 = (
        np.broadcast_to(base_arr[None, :], (3, len(rows)))
        if base_arr.ndim == 1
        else base_arr
    )
    fm_base = np.stack([family_matrix(v, families)[1] for v in base_3])
    n_fams = fm_base.shape[1]
    ix = np.random.default_rng(seed).integers(0, n_fams, (DRAWS, n_fams))
    p_med_by_arm = {}
    for arm in CANDIDATES:
        arr = np.asarray(candidate_panels_by_arm[arm], dtype=np.float64)
        fm_arm = np.stack([family_matrix(v, families)[1] for v in arr])
        diff = fm_arm - fm_base
        boot_med = np.median(diff[:, ix].mean(axis=2), axis=0)
        p_med_by_arm[arm] = float((1 + (boot_med <= 0.0).sum()) / (DRAWS + 1))

    pvals_m2 = {
        arm: float(raw_h177[H177_SLOT_BY_ARM[arm]]["p_one_sided"]) for arm in CANDIDATES
    }
    holm_m2 = holm(pvals_m2)
    holm_m2_med = holm(p_med_by_arm)

    decisions = {}
    for arm in CANDIDATES:
        slot = H177_SLOT_BY_ARM[arm]
        d = dict(raw_h177[slot])
        cand_step = int(steps[arm])
        unadj_ok = bool(d["unadjusted_admitted"])
        h_p = float(holm_m2[arm])
        h_p_med = float(holm_m2_med[arm])
        admitted_m2 = bool(unadj_ok and h_p < 0.05)
        decisions[arm] = dict(
            prespecified_candidate_step=cand_step,
            mean_gain=float(d["mean_gain"]),
            ci95=list(d["ci95"]),
            per_seed_gain=list(d["per_seed_gain"]),
            median_seed_gain=float(d["median_seed_gain"]),
            median_seed_ci95=list(d["median_seed_ci95"]),
            positive_seeds=int(d["positive_seeds"]),
            p_one_sided=float(d["p_one_sided"]),
            p_one_sided_median_seed=float(p_med_by_arm[arm]),
            holm_multiplicity=HOLM_MULTIPLICITY,
            holm_p=h_p,
            holm_p_median_seed=h_p_med,
            holm_p_m4_with_ctrl=float(d["holm_p"]),
            unadjusted_admitted=unadj_ok,
            admitted=admitted_m2,
            admitted_m4_with_ctrl=bool(d["admitted"]),
            selected_step=cand_step if admitted_m2 else 0,
            selected_step_unadjusted=cand_step if unadj_ok else 0,
            h177_slot=slot,
        )
    return decisions


def partition_variance_decomposition(rows, panel):
    """Quantify the between-seed (independent fold-partition + member-init/batch) variance component."""
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (3, len(rows), len(ALL_ARMS)) or not np.isfinite(values).all():
        raise ValueError(f"expected (3, {len(rows)}, {len(ALL_ARMS)}) finite panel")
    families = [r["family_id"] for r in rows]
    fm = np.stack([family_matrix(v, families)[1] for v in values])  # (3, n_fams, 3)

    out = {}
    pairs = [
        ("ENS10_minus_R882", ALL_ARMS.index("ENS10"), 0),
        ("RENS10_minus_R882", ALL_ARMS.index("RENS10"), 0),
        ("ENS10_minus_RENS10", ALL_ARMS.index("ENS10"), ALL_ARMS.index("RENS10")),
    ]
    for name, col_a, col_b in pairs:
        diff = fm[:, :, col_a] - fm[:, :, col_b]  # (3, n_fams)
        per_seed = diff.mean(axis=1)  # (3,)
        seed_avg_fam = diff.mean(axis=0)  # (n_fams,)
        between_seed_var = float(np.var(per_seed, ddof=1))
        between_seed_sd = float(np.std(per_seed, ddof=1))
        between_seed_range = float(np.max(per_seed) - np.min(per_seed))
        between_family_var = float(np.var(seed_avg_fam, ddof=1))
        within_fam_between_seed_var = float(np.mean(np.var(diff, axis=0, ddof=1)))
        within_fam_between_seed_sd = float(math.sqrt(within_fam_between_seed_var))
        out[name] = dict(
            per_seed_gain=per_seed.tolist(),
            between_seed_sd_ddof1=between_seed_sd,
            between_seed_var_ddof1=between_seed_var,
            between_seed_range=between_seed_range,
            between_seed_se_of_mean=float(between_seed_sd / math.sqrt(3.0)),
            between_family_var_ddof1=between_family_var,
            within_family_between_seed_var_ddof1=within_fam_between_seed_var,
            within_family_between_seed_sd_ddof1=within_fam_between_seed_sd,
        )
    return out


def summarize_panel(
    rows, panel, minus2_by_arm=None, agreement_by_arm=None, seed=BOOT_SEED
):
    """Summarize a 3-policy panel (R882, ENS10, RENS10) of shape (3, N_ctx, 3).

    Computes both:
    - 3-seed within-family mean gain + 10,000-draw paired family bootstrap 95% CI (H177 primary functional), and
    - median-over-seeds gain + 10,000-draw paired family bootstrap 95% CI reducing over seeds along axis=0,
    plus Holm m=2 adjusted p-values across (ENS10, RENS10), diagnostic contrast (ENS10-RENS10),
    partition-variance decomposition, and canonical H148 400-iteration MM Bradley-Terry Elo vs r882=1500.0.
    """
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (3, len(rows), len(ALL_ARMS)) or not np.isfinite(values).all():
        raise ValueError(
            f"complete 3-seed {len(ALL_ARMS)}-policy panel required, got {values.shape}"
        )
    families = [r["family_id"] for r in rows]
    fm = np.stack([family_matrix(v, families)[1] for v in values])  # (3, n_fams, 3)
    n_fams = fm.shape[1]
    ix = np.random.default_rng(seed).integers(0, n_fams, (DRAWS, n_fams))
    means = fm.mean(axis=1)  # (3, 3)

    arms_summary = {}
    for j, arm in enumerate(ALL_ARMS):
        fam_draws_by_seed = fm[:, :, j][:, ix].mean(axis=2)  # (3, DRAWS)
        boot_seed_avg = fm[:, :, j].mean(axis=0)[ix].mean(axis=1)  # (DRAWS,)
        boot_med_seed = np.median(
            fam_draws_by_seed, axis=0
        )  # (DRAWS,) — axis=0 over seeds!
        entry = dict(
            per_seed_equal_family_utility=means[:, j].tolist(),
            seed_averaged_equal_family_utility=float(means[:, j].mean()),
            seed_averaged_ci95=np.quantile(boot_seed_avg, [0.025, 0.975]).tolist(),
            median_seed_equal_family_utility=float(np.median(means[:, j])),
            median_seed_ci95=np.quantile(boot_med_seed, [0.025, 0.975]).tolist(),
            per_seed_context_utility=values[:, :, j].mean(axis=1).tolist(),
            minus2_contexts_per_seed=(
                list(minus2_by_arm[arm])
                if minus2_by_arm is not None and arm in minus2_by_arm
                else [int(np.count_nonzero(values[s, :, j] == -2.0)) for s in range(3)]
            ),
        )
        if agreement_by_arm is not None and arm in agreement_by_arm:
            entry["choice_agreement_with_r882_per_seed"] = list(agreement_by_arm[arm])
        arms_summary[arm] = entry

    contrasts = {}

    def add_contrast(name, col_a, col_b, is_diagnostic):
        diff = fm[:, :, col_a] - fm[:, :, col_b]  # (3, n_fams)
        per_seed = diff.mean(axis=1)  # (3,)
        seed_avg_fam = diff.mean(axis=0)  # (n_fams,)
        boot_avg = seed_avg_fam[ix].mean(axis=1)  # (DRAWS,)
        boot_by_seed = diff[:, ix].mean(axis=2)  # (3, DRAWS)
        boot_med = np.median(boot_by_seed, axis=0)  # (DRAWS,)
        contrasts[name] = dict(
            diagnostic_only=bool(is_diagnostic),
            per_seed_gain=per_seed.tolist(),
            between_seed_sd_ddof1=float(np.std(per_seed, ddof=1)),
            between_seed_range=float(np.max(per_seed) - np.min(per_seed)),
            mean_gain=float(seed_avg_fam.mean()),
            ci95=np.quantile(boot_avg, [0.025, 0.975]).tolist(),
            p_one_sided=float((1 + (boot_avg <= 0.0).sum()) / (DRAWS + 1)),
            median_gain=float(np.median(per_seed)),
            median_seed_ci95=np.quantile(boot_med, [0.025, 0.975]).tolist(),
            p_one_sided_median_seed=float((1 + (boot_med <= 0.0).sum()) / (DRAWS + 1)),
            positive_seeds=int((per_seed > 0.0).sum()),
        )

    for col, arm in enumerate(CANDIDATES, 1):
        add_contrast(f"{arm}_minus_R882", col, 0, is_diagnostic=False)

    add_contrast(
        "ENS10_minus_RENS10",
        ALL_ARMS.index("ENS10"),
        ALL_ARMS.index("RENS10"),
        is_diagnostic=True,
    )

    adjusted_m2 = holm(
        {arm: contrasts[f"{arm}_minus_R882"]["p_one_sided"] for arm in CANDIDATES}
    )
    adjusted_m2_med = holm(
        {
            arm: contrasts[f"{arm}_minus_R882"]["p_one_sided_median_seed"]
            for arm in CANDIDATES
        }
    )
    for arm in CANDIDATES:
        c = contrasts[f"{arm}_minus_R882"]
        c["holm_multiplicity"] = HOLM_MULTIPLICITY
        c["holm_p"] = float(adjusted_m2[arm])
        c["holm_p_median_seed"] = float(adjusted_m2_med[arm])
        c["admitted_pooled_rule"] = bool(
            c["mean_gain"] > MARGIN + EPS and c["ci95"][0] > 0.0 and c["holm_p"] < 0.05
        )

    fw = np.stack([family_wins(v, families) for v in values])  # (3, n_fams, 3, 3)
    elo_by_seed = bt_ratings(fw.sum(axis=1))  # (3, 3)
    elo_seed_avg = bt_ratings(fw.mean(axis=0).sum(axis=0))  # (3,)
    ei = np.random.default_rng(seed + 1).integers(0, n_fams, (1000, n_fams))
    elo_boot_seed_avg = bt_ratings(fw.mean(axis=0)[ei].sum(axis=1))  # (1000, 3)
    elo_boot_by_seed = np.stack(
        [bt_ratings(fw[s][ei].sum(axis=1)) for s in range(3)], axis=0
    )  # (3, 1000, 3)
    elo_boot_med_seed = np.median(elo_boot_by_seed, axis=0)  # (1000, 3)

    elo_summary = {}
    for j, arm in enumerate(ALL_ARMS):
        elo_summary[arm] = dict(
            per_seed=elo_by_seed[:, j].tolist(),
            median_over_seeds=float(np.median(elo_by_seed[:, j])),
            median_seed_ci95=np.quantile(
                elo_boot_med_seed[:, j], [0.025, 0.975]
            ).tolist(),
            seed_averaged_elo=float(elo_seed_avg[j]),
            seed_averaged_ci95=np.quantile(
                elo_boot_seed_avg[:, j], [0.025, 0.975]
            ).tolist(),
        )

    return dict(
        contexts=len(rows),
        families=n_fams,
        arms=arms_summary,
        contrasts=contrasts,
        partition_variance=partition_variance_decomposition(rows, values),
        elo_vs_r882=elo_summary,
        utility_bootstrap_draws=DRAWS,
        elo_bootstrap_draws=1000,
    )


def pairwise_member_disagreement(choices_3d, k=DEFAULT_ENSEMBLE_K):
    """Compute pairwise member choice disagreement rate across (3, N_ctx, k) integer choice arrays."""
    arr = np.asarray(choices_3d, dtype=np.int64)
    if arr.ndim != 3 or arr.shape[0] != 3 or arr.shape[2] < k:
        raise ValueError(f"expected (3, N_ctx, >={k}) choice array, got {arr.shape}")
    sub = arr[:, :, :k]
    n_pairs = k * (k - 1) // 2
    per_seed = []
    for s in range(3):
        dis = 0
        for i in range(k):
            for j in range(i + 1, k):
                dis += int(np.count_nonzero(sub[s, :, i] != sub[s, :, j]))
        per_seed.append(float(dis / (sub.shape[1] * n_pairs)))
    return dict(
        k=int(k),
        mean_over_seeds=float(np.mean(per_seed)),
        median_over_seeds=float(np.median(per_seed)),
        per_seed=per_seed,
    )


def resolve_h190_verdict(
    pooled_b_decisions,
    confirm_a_summary,
    h178_receipt,
    expected_h177_analysis_sha256=None,
):
    """Apply H190's two-stage confirmatory rule and delegate final H178-gated lift resolution to `resolve_h177_verdict`.

    Per hwave_t0835_h190.md:
    - ADVANCE only if:
      (1) admitted on Stage-1 pooled B (`mean_gain > 0.001 + 1e-12`, `ci95[0] > 0`, Holm m=2 `holm_p < 0.05`), AND
      (2) on Stage-2 pooled A, gain > 0 (`mean_gain > 1e-12` and `median_gain > 1e-12`) with 3/3 independent-partition seeds
          positive (`positive_seeds == 3`) AND median CI lower > 0 (`median_seed_ci95[0] > 0.0` and `ci95[0] > 0.0`),
      (3) plus H178 receipt gate via `resolve_h177_verdict`. Ties (<= +0.001 on B or <= 0 on A) reject.
    """
    per_arm_qualification = {}
    qualifying_arms = []
    for arm in CANDIDATES:
        b_dec = pooled_b_decisions[arm]
        a_con = confirm_a_summary["contrasts"][f"{arm}_minus_R882"]
        b_admitted = bool(b_dec["admitted"])
        a_mean_pos = bool(float(a_con["mean_gain"]) > EPS)
        a_med_pos = bool(float(a_con["median_gain"]) > EPS)
        a_all_seeds_pos = bool(int(a_con["positive_seeds"]) == 3)
        a_med_ci_pos = bool(float(a_con["median_seed_ci95"][0]) > 0.0)
        a_mean_ci_pos = bool(float(a_con["ci95"][0]) > 0.0)
        a_confirmed = bool(
            a_mean_pos
            and a_med_pos
            and a_all_seeds_pos
            and a_med_ci_pos
            and a_mean_ci_pos
        )
        qualifies = bool(b_admitted and a_confirmed)
        if qualifies:
            qualifying_arms.append(arm)
        per_arm_qualification[arm] = dict(
            pooled_b_admitted=b_admitted,
            pooled_b_mean_gain=float(b_dec["mean_gain"]),
            pooled_b_ci95=list(b_dec["ci95"]),
            pooled_b_median_seed_gain=float(b_dec["median_seed_gain"]),
            pooled_b_median_seed_ci95=list(b_dec["median_seed_ci95"]),
            pooled_b_holm_p_m2=float(b_dec["holm_p"]),
            pooled_a_mean_gain=float(a_con["mean_gain"]),
            pooled_a_median_gain=float(a_con["median_gain"]),
            pooled_a_ci95=list(a_con["ci95"]),
            pooled_a_median_seed_ci95=list(a_con["median_seed_ci95"]),
            pooled_a_positive_seeds=int(a_con["positive_seeds"]),
            pooled_a_median_ci_lower_gt_0=a_med_ci_pos,
            pooled_a_confirmed=a_confirmed,
            qualifies_for_advance=qualifies,
        )

    candidate_admitted = len(qualifying_arms) > 0
    verdict_info = resolve_h177_verdict(
        candidate_admitted,
        h178_receipt=h178_receipt,
        expected_analysis_sha256=expected_h177_analysis_sha256,
    )
    return dict(
        **verdict_info,
        qualifying_arms=qualifying_arms,
        per_arm_qualification=per_arm_qualification,
        serving_promotion=False,
        adoption_note=(
            "An ADVANCE is not adopted by H190; if verdict == ADVANCE, flag @flywheel_gp_fused in AGENT_CHAT "
            "for promotion review with DOPE checksum verification."
        ),
    )
