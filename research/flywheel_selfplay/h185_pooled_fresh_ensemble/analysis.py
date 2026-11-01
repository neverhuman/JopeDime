"""H185 power-aware pooled confirmatory selector for fresh-init and ensemble candidates (FRESH160, ENS5, RENS5 vs R882).

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

SEEDS = (185101, 185102, 185103)
MEMBER_SEEDS_BY_SEED = {
    185101: (185201, 185202, 185203, 185204, 185205),
    185102: (185301, 185302, 185303, 185304, 185305),
    185103: (185401, 185402, 185403, 185404, 185405),
}
FOLD_SEED = 185140
BOOT_SEED = 185141

CANDIDATES = ("FRESH160", "ENS5", "RENS5")
ALL_ARMS = ("R882",) + CANDIDATES
HOLM_MULTIPLICITY = 3
MARGIN = 0.001

FRESH160_DEFAULT_UPDATES = 160
FRESH160_FALLBACK_UPDATES = 80
ENS5_MEMBER_UPDATES = 40
RENS5_MEMBER_UPDATES = 0
PILOT_UPDATES = 4
WALL_CAP_SECONDS = 10_800
EVAL_RESERVE_SECONDS = 1_200

PRESPECIFIED_CANDIDATE_STEP = {
    "FRESH160": FRESH160_DEFAULT_UPDATES,
    "ENS5": ENS5_MEMBER_UPDATES,
    "RENS5": RENS5_MEMBER_UPDATES,
}

# Exact mapping of the 3 H185 candidate arms onto the 3 non-CTRL treatment slots of H177's pooled_b_select
# so `h177_analysis.pooled_b_select` executes its built-in 3-treatment Holm adjustment (`admitted_treatments3`,
# `holm_p_treatments3`) and 10,000-draw paired family bootstrap verbatim.
H177_SLOT_BY_ARM = {
    "FRESH160": "ACC4",
    "ENS5": "ACC16",
    "RENS5": "OC15",
}


def stream_seed(*parts):
    return h177_analysis.stream_seed(*parts)


def member_seeds_for(outer_seed):
    if outer_seed not in MEMBER_SEEDS_BY_SEED:
        raise ValueError(f"unknown outer seed: {outer_seed}")
    return MEMBER_SEEDS_BY_SEED[outer_seed]


def fresh160_init_seed(outer_seed, outer_fold):
    if outer_seed not in SEEDS or not 0 <= int(outer_fold) < 5:
        raise ValueError("invalid outer_seed or outer_fold")
    return stream_seed(outer_seed, outer_fold, "FRESH160", "fresh-native-init-v1")


def validate_member_init_hashes(hashes):
    """Require five distinct non-empty constructor-produced adapter hashes for the five ensemble members."""
    values = tuple(hashes)
    if (
        len(values) != 5
        or any(not isinstance(v, str) or not v for v in values)
        or len(set(values)) != 5
    ):
        raise ValueError("five distinct fresh-member initialization hashes required")
    return values


def validate_rens5_matches_ens5_inits(rens5_hashes, ens5_init_hashes):
    """Prove RENS5 scores the exact same 5 untrained member initializations that ENS5 trains from."""
    r_vals = validate_member_init_hashes(rens5_hashes)
    e_vals = validate_member_init_hashes(ens5_init_hashes)
    if r_vals != e_vals:
        raise ValueError(
            "RENS5 member initializations do not match ENS5 member initializations"
        )
    return True


def validate_cell_init_hashes(r882_hash, fresh160_hash, member_hashes):
    """Require r882, FRESH160 init, and all 5 ENS5/RENS5 member inits to be mutually distinct."""
    m_vals = validate_member_init_hashes(member_hashes)
    all_hashes = (r882_hash, fresh160_hash) + m_vals
    if (
        any(not isinstance(v, str) or not v for v in all_hashes)
        or len(set(all_hashes)) != 7
    ):
        raise ValueError(
            "r882, FRESH160, and the 5 member initializations must have 7 distinct hashes"
        )
    return all_hashes


def score_ensemble(member_scores, k=5):
    """Average per-candidate cost tensors across the k ensemble members before taking argmin."""
    scores = np.asarray(member_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != k or not np.isfinite(scores).all():
        raise ValueError(
            f"expected finite ({k}, n_candidates) member score array, got {scores.shape}"
        )
    return scores.mean(axis=0)


def folds(rows, fold_seed=FOLD_SEED):
    fams = sorted({r["family_id"] for r in rows})
    if len(fams) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(fold_seed).permutation(205)
    return [sorted(fams[i] for i in group) for group in np.array_split(order, 5)]


def partition(rows, outer, fold_seed=FOLD_SEED):
    if not 0 <= int(outer) < 5:
        raise ValueError("outer fold must be in 0..4")
    groups = folds(rows, fold_seed=fold_seed)
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    return (
        [r for r in rows if r["family_id"] not in a | b],
        [r for r in rows if r["family_id"] in b],
        [r for r in rows if r["family_id"] in a],
    )


def verify_pooled_b_coverage(rows, fold_seed=FOLD_SEED):
    """Invoke H177's verify_pooled_b_coverage directly under H185's fresh fold_seed=185140."""
    prev_seed = h177_analysis.FOLD_SEED
    try:
        h177_analysis.FOLD_SEED = int(fold_seed)
        proof = h177_analysis.verify_pooled_b_coverage(rows)
    finally:
        h177_analysis.FOLD_SEED = prev_seed
    return dict(**proof, fold_seed=int(fold_seed))


class StageOrderGuard(h177_analysis.StageOrderGuard):
    """Reuses H177's StageOrderGuard to enforce immutable candidate steps and fsynced pooled_b_selection_lock.json before Stage 2."""

    def __init__(self, expected_steps=None):
        steps = dict(expected_steps or PRESPECIFIED_CANDIDATE_STEP)
        if set(steps) != set(CANDIDATES):
            raise ValueError(
                f"expected steps for {list(CANDIDATES)}, got {sorted(steps)}"
            )
        if (
            not isinstance(steps["FRESH160"], int)
            or steps["FRESH160"]
            not in (FRESH160_DEFAULT_UPDATES, FRESH160_FALLBACK_UPDATES)
            or steps["ENS5"] != ENS5_MEMBER_UPDATES
            or steps["RENS5"] != RENS5_MEMBER_UPDATES
        ):
            raise ValueError("invalid prespecified candidate steps for H185 arms")
        self.expected_steps = steps
        self.b_read_started = False
        self.b_locked_path = None

    def mark_b_locked(self, lock_path):
        p = Path(lock_path)
        if not p.is_file():
            raise RuntimeError("pooled_b_selection_lock.json does not exist on disk")
        payload = json.loads(p.read_text(encoding="utf-8"))
        if (
            payload.get("wave") != "H185"
            or payload.get("pooled_b_families") != 205
            or payload.get("pooled_b_contexts") != 1219
            or payload.get("locked_before_a_confirmation") is not True
            or set(payload.get("decisions", {})) != set(CANDIDATES)
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
            payload.get("wave") != "H185"
            or payload.get("pooled_b_families") != 205
            or set(payload.get("decisions", {})) != set(CANDIDATES)
        ):
            raise RuntimeError("corrupted pooled_b_selection_lock.json")
        return payload


def batch_plan(cids, orders, seed, updates=40):
    """H156/H179/H180 FIT-only minibatch sampler: 8 FIT contexts, 4 top-fitness revealed + 20 random unrevealed candidates."""
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
    """Timing-only pilot rule: keep FRESH160=160 updates (5,400 total updates) if <= 3h; else drop FRESH160 to 80 updates (4,200 total updates). Never fewer than 3 seeds or 5 folds."""
    if not math.isfinite(seconds_per_update) or seconds_per_update <= 0.0:
        raise ValueError("positive finite seconds_per_update required")
    total_default = (
        (FRESH160_DEFAULT_UPDATES + 5 * ENS5_MEMBER_UPDATES) * 5 * len(SEEDS)
    )
    proj_default = total_default * float(seconds_per_update) + reserve_seconds
    if proj_default <= cap_seconds:
        return dict(
            outer_folds=5,
            seeds=len(SEEDS),
            fresh160_updates=FRESH160_DEFAULT_UPDATES,
            ens5_member_updates=ENS5_MEMBER_UPDATES,
            rens5_member_updates=RENS5_MEMBER_UPDATES,
            total_optimizer_updates=total_default,
            projected_seconds=proj_default,
            fallback_triggered=False,
        )
    total_fallback = (
        (FRESH160_FALLBACK_UPDATES + 5 * ENS5_MEMBER_UPDATES) * 5 * len(SEEDS)
    )
    proj_fallback = total_fallback * float(seconds_per_update) + reserve_seconds
    return dict(
        outer_folds=5,
        seeds=len(SEEDS),
        fresh160_updates=FRESH160_FALLBACK_UPDATES,
        ens5_member_updates=ENS5_MEMBER_UPDATES,
        rens5_member_updates=RENS5_MEMBER_UPDATES,
        total_optimizer_updates=total_fallback,
        projected_seconds=proj_fallback,
        fallback_triggered=True,
    )


def pooled_b_select_h185(
    rows,
    r882_values,
    candidate_panels_by_arm,
    seed=BOOT_SEED,
    prespecified_steps=None,
):
    """Delegate Stage-1 205-family pooled-B selection directly to H177_DECISION_RULE.select (`h177_analysis.pooled_b_select`).

    Maps R882 to CTRL and (FRESH160, ENS5, RENS5) to the 3 non-CTRL slots (ACC4, ACC16, OC15) so that
    `h177_analysis.pooled_b_select` computes the exact 3-seed within-family mean, 10,000-draw paired family bootstrap,
    axis=0 median-seed CI, and m=3 Holm-Bonferroni adjustment (`holm_p_treatments3`, `admitted_treatments3`) without
    reimplementing any selector logic.
    """
    if not isinstance(candidate_panels_by_arm, dict) or set(
        candidate_panels_by_arm
    ) != set(CANDIDATES):
        raise ValueError(f"exact 3 candidate arms required: {list(CANDIDATES)}")
    steps = dict(prespecified_steps or PRESPECIFIED_CANDIDATE_STEP)
    mapped_panels = {
        "CTRL": np.asarray(r882_values, dtype=np.float64),
        H177_SLOT_BY_ARM["FRESH160"]: np.asarray(
            candidate_panels_by_arm["FRESH160"], dtype=np.float64
        ),
        H177_SLOT_BY_ARM["ENS5"]: np.asarray(
            candidate_panels_by_arm["ENS5"], dtype=np.float64
        ),
        H177_SLOT_BY_ARM["RENS5"]: np.asarray(
            candidate_panels_by_arm["RENS5"], dtype=np.float64
        ),
    }
    raw_h177 = H177_DECISION_RULE.select(rows, r882_values, mapped_panels, seed=seed)
    decisions = {}
    for arm in CANDIDATES:
        slot = H177_SLOT_BY_ARM[arm]
        d = dict(raw_h177[slot])
        cand_step = int(steps[arm])
        admitted_m3 = bool(d["admitted_treatments3"])
        unadj_ok = bool(d["unadjusted_admitted"])
        decisions[arm] = dict(
            prespecified_candidate_step=cand_step,
            mean_gain=float(d["mean_gain"]),
            ci95=list(d["ci95"]),
            per_seed_gain=list(d["per_seed_gain"]),
            median_seed_gain=float(d["median_seed_gain"]),
            median_seed_ci95=list(d["median_seed_ci95"]),
            positive_seeds=int(d["positive_seeds"]),
            p_one_sided=float(d["p_one_sided"]),
            holm_multiplicity=HOLM_MULTIPLICITY,
            holm_p=float(d["holm_p_treatments3"]),
            holm_p_m4_with_ctrl=float(d["holm_p"]),
            unadjusted_admitted=unadj_ok,
            admitted=admitted_m3,
            admitted_m4_with_ctrl=bool(d["admitted"]),
            selected_step=cand_step if admitted_m3 else 0,
            selected_step_unadjusted=cand_step if unadj_ok else 0,
            h177_slot=slot,
        )
    return decisions


def summarize_panel(
    rows, panel, minus2_by_arm=None, agreement_by_arm=None, seed=BOOT_SEED
):
    """Summarize a 4-policy panel (R882, FRESH160, ENS5, RENS5) of shape (3, N_ctx, 4).

    Computes both:
    - 3-seed within-family mean gain + 10,000-draw paired family bootstrap 95% CI (H177 primary functional), and
    - median-over-seeds gain + 10,000-draw paired family bootstrap 95% CI reducing over seeds along axis=0 (H176 #1512 / H179 / H180 functional),
    plus Holm m=3 adjusted p-values across (FRESH160, ENS5, RENS5), diagnostic contrasts (ENS5-RENS5, FRESH160-RENS5, ENS5-FRESH160),
    and canonical H148 400-iteration MM Bradley-Terry Elo vs r882=1500.0 with 1,000-draw family bootstrap CI.
    """
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (3, len(rows), len(ALL_ARMS)) or not np.isfinite(values).all():
        raise ValueError(
            f"complete 3-seed {len(ALL_ARMS)}-policy panel required, got {values.shape}"
        )
    families = [r["family_id"] for r in rows]
    fm = np.stack([family_matrix(v, families)[1] for v in values])  # (3, n_fams, 4)
    n_fams = fm.shape[1]
    ix = np.random.default_rng(seed).integers(0, n_fams, (DRAWS, n_fams))
    means = fm.mean(axis=1)  # (3, 4)

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
        # Paired family bootstrap per seed -> (3, DRAWS), then median over seeds along axis=0 -> (DRAWS,)
        boot_by_seed = diff[:, ix].mean(axis=2)  # (3, DRAWS)
        boot_med = np.median(boot_by_seed, axis=0)  # (DRAWS,)
        contrasts[name] = dict(
            diagnostic_only=bool(is_diagnostic),
            per_seed_gain=per_seed.tolist(),
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

    # Prespecified diagnostic contrasts (mechanism decomposition; not used for admission)
    add_contrast(
        "ENS5_minus_RENS5",
        ALL_ARMS.index("ENS5"),
        ALL_ARMS.index("RENS5"),
        is_diagnostic=True,
    )
    add_contrast(
        "FRESH160_minus_RENS5",
        ALL_ARMS.index("FRESH160"),
        ALL_ARMS.index("RENS5"),
        is_diagnostic=True,
    )
    add_contrast(
        "ENS5_minus_FRESH160",
        ALL_ARMS.index("ENS5"),
        ALL_ARMS.index("FRESH160"),
        is_diagnostic=True,
    )

    adjusted_m3 = holm(
        {arm: contrasts[f"{arm}_minus_R882"]["p_one_sided"] for arm in CANDIDATES}
    )
    adjusted_m3_med = holm(
        {
            arm: contrasts[f"{arm}_minus_R882"]["p_one_sided_median_seed"]
            for arm in CANDIDATES
        }
    )
    for arm in CANDIDATES:
        c = contrasts[f"{arm}_minus_R882"]
        c["holm_multiplicity"] = HOLM_MULTIPLICITY
        c["holm_p"] = float(adjusted_m3[arm])
        c["holm_p_median_seed"] = float(adjusted_m3_med[arm])
        c["admitted_pooled_rule"] = bool(
            c["mean_gain"] > MARGIN + EPS and c["ci95"][0] > 0.0 and c["holm_p"] < 0.05
        )

    fw = np.stack([family_wins(v, families) for v in values])  # (3, n_fams, 4, 4)
    elo_by_seed = bt_ratings(fw.sum(axis=1))  # (3, 4)
    elo_seed_avg = bt_ratings(fw.mean(axis=0).sum(axis=0))  # (4,)
    ei = np.random.default_rng(seed + 1).integers(0, n_fams, (1000, n_fams))
    elo_boot_seed_avg = bt_ratings(fw.mean(axis=0)[ei].sum(axis=1))  # (1000, 4)
    # Also compute median-over-seeds Elo bootstrap CI (axis=0 over seeds)
    elo_boot_by_seed = np.stack(
        [bt_ratings(fw[s][ei].sum(axis=1)) for s in range(3)], axis=0
    )  # (3, 1000, 4)
    elo_boot_med_seed = np.median(elo_boot_by_seed, axis=0)  # (1000, 4)

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
        elo_vs_r882=elo_summary,
        utility_bootstrap_draws=DRAWS,
        elo_bootstrap_draws=1000,
    )


def resolve_h185_verdict(
    pooled_b_decisions,
    confirm_a_summary,
    h178_receipt,
    expected_h177_analysis_sha256=None,
):
    """Apply H185's two-stage confirmatory rule and delegate final H178-gated lift resolution to `resolve_h177_verdict`.

    Per hwave_t0700_h185_tokagy.md:
    - An arm qualifies iff:
      (1) it is admitted on Stage-1 pooled B (`mean_gain > 0.001 + 1e-12`, `ci95[0] > 0`, Holm m=3 `holm_p < 0.05`), AND
      (2) on Stage-2 pooled A, its equal-family gain vs R882 is strictly positive (`mean_gain > 1e-12` and `median_gain > 1e-12`, ties reject)
          with all 3/3 seeds positive (`positive_seeds == 3`).
    - Then `resolve_h177_verdict(candidate_admitted, h178_receipt, expected_analysis_sha256)` checks the H178 calibration receipt.
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
        a_confirmed = bool(a_mean_pos and a_med_pos and a_all_seeds_pos)
        qualifies = bool(b_admitted and a_confirmed)
        if qualifies:
            qualifying_arms.append(arm)
        per_arm_qualification[arm] = dict(
            pooled_b_admitted=b_admitted,
            pooled_b_mean_gain=float(b_dec["mean_gain"]),
            pooled_b_ci95=list(b_dec["ci95"]),
            pooled_b_holm_p_m3=float(b_dec["holm_p"]),
            pooled_a_mean_gain=float(a_con["mean_gain"]),
            pooled_a_median_gain=float(a_con["median_gain"]),
            pooled_a_ci95=list(a_con["ci95"]),
            pooled_a_median_seed_ci95=list(a_con["median_seed_ci95"]),
            pooled_a_positive_seeds=int(a_con["positive_seeds"]),
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
        adoption_note="An ADVANCE is not adopted by H185; if verdict == ADVANCE, flag @flywheel_gp_fused in AGENT_CHAT for promotion review with DOPE checksum verification.",
    )
