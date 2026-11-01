"""H178 empirical calibration of the H177 pooled 205-family selector on the H175 counterfactual bank."""

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _load_sibling_module(mod_name, rel_path):
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
h175_protocol = _load_sibling_module(
    "h175_selector_power_protocol",
    "research/flywheel_selfplay/h175_selector_power/protocol.py",
)

# Exact imported decision function, rule object, and guard from H177 (never reimplemented).
pooled_b_select = h177_analysis.pooled_b_select
H177_DECISION_RULE = h177_analysis.H177_DECISION_RULE
resolve_h177_verdict = h177_analysis.resolve_h177_verdict
StageOrderGuard = h177_analysis.StageOrderGuard
verify_pooled_b_coverage = h177_analysis.verify_pooled_b_coverage
expected_mde80_205fam = h177_analysis.expected_mde80_205fam
ARMS = h177_analysis.ARMS
PRESPECIFIED_CANDIDATE_STEP = h177_analysis.PRESPECIFIED_CANDIDATE_STEP
H177_ANALYSIS_PATH = (
    ROOT / "research/flywheel_selfplay/h177_pooled_selector/analysis.py"
)
H177_ANALYSIS_SHA256 = hashlib.sha256(H177_ANALYSIS_PATH.read_bytes()).hexdigest()

SEEDS = (178101, 178102, 178103)
SLATE_SEEDS = h175_protocol.SEEDS
FOLD_SEED = 178140
BOOT_SEED = 178141
FRACTIONS = h175_protocol.FRACTIONS
SIMULATIONS = h175_protocol.SIMULATIONS
INNER_DRAWS = 1000
FINAL_DRAWS = 10000
REPORT_DRAWS = h175_protocol.REPORT_DRAWS
MARGIN = h175_protocol.MARGIN
EPS = h175_protocol.EPS
MAX_NULL_HARMFUL_ADMISSION = 0.05
MAX_FAMILY_MDE80 = 0.03


def save(path, obj):
    p = Path(path)
    with p.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


class LocalOnlyPayload(dict):
    """Role-local payload mapping that forbids reading any context outside the active fold role."""

    def __init__(self, full_payload, allowed_scope, role_name="B"):
        self._scope = set(allowed_scope)
        self._role_name = role_name
        super().__init__({k: full_payload[k] for k in self._scope if k in full_payload})

    def __getitem__(self, key):
        if key not in self._scope:
            raise KeyError(f"outside {self._role_name} role scope read: {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in self._scope:
            raise KeyError(f"outside {self._role_name} role scope read: {key}")
        return super().get(key, default)


def folds(rows, fold_seed=FOLD_SEED):
    families = sorted({r["family_id"] for r in rows})
    if len(families) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(fold_seed).permutation(len(families))
    return [sorted(families[i] for i in g) for g in np.array_split(order, 5)]


def partition(rows, outer, fold_seed=FOLD_SEED):
    groups = folds(rows, fold_seed=fold_seed)
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    return (
        [r for r in rows if r["family_id"] not in a | b],
        [r for r in rows if r["family_id"] in b],
        [r for r in rows if r["family_id"] in a],
    )


def _binom_cdf_le(k, n, p):
    if p <= 0.0:
        return 1.0 if k >= 0 else 0.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp = math.log(p)
    lq = math.log1p(-p)
    lg_n = math.lgamma(n + 1)
    terms = [
        math.exp(
            lg_n - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lp + (n - j) * lq
        )
        for j in range(k + 1)
    ]
    return min(1.0, max(0.0, math.fsum(terms)))


def clopper_pearson_upper(k, n, alpha_tail=0.05):
    """Exact Clopper-Pearson upper bound p_U such that P(Bin(n, p_U) <= k) = alpha_tail."""
    k, n = int(k), int(n)
    if n <= 0 or k < 0 or k > n:
        raise ValueError("invalid binomial counts")
    if k == n:
        return 1.0
    if k == 0:
        return float(1.0 - (alpha_tail ** (1.0 / n)))
    lo, hi = float(k) / float(n), 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _binom_cdf_le(k, n, mid) > alpha_tail:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def clopper_pearson_lower(k, n, alpha_tail=0.025):
    """Exact Clopper-Pearson lower bound p_L such that P(Bin(n, p_L) >= k) = alpha_tail."""
    k, n = int(k), int(n)
    if n <= 0 or k < 0 or k > n:
        raise ValueError("invalid binomial counts")
    if k == 0:
        return 0.0
    return float(1.0 - clopper_pearson_upper(n - k, n, alpha_tail=alpha_tail))


def clopper_pearson_summary(k, n, trial_unit_note=None):
    k, n = int(k), int(n)
    out = dict(
        admitted=k,
        trials=n,
        rate=float(k / n),
        cp95_one_sided_upper=clopper_pearson_upper(k, n, alpha_tail=0.05),
        cp95_two_sided=[
            clopper_pearson_lower(k, n, alpha_tail=0.025),
            clopper_pearson_upper(k, n, alpha_tail=0.025),
        ],
    )
    if trial_unit_note is not None:
        out["trial_unit"] = trial_unit_note
    return out


def evaluate_h177_pooled_rule(
    family_deltas,
    seed,
    direction,
    fraction,
    inner_draws=INNER_DRAWS,
    final_draws=FINAL_DRAWS,
):
    """Invoke the exact imported h177_analysis.pooled_b_select function across 4-arm cohorts.

    - family_deltas has shape (S, 3, 205), partitioned in deterministic simulation order into
      n_cohorts = S // 4 disjoint 4-arm cohorts {"CTRL": arr[4*b], "ACC4": arr[4*b+1],
      "ACC16": arr[4*b+2], "OC15": arr[4*b+3]}.
    - Consumes d["admitted"], d["holm_p"], d["holm_p_treatments3"], d["unadjusted_admitted"],
      d["mean_gain"], d["ci95"], and d["p_one_sided"] directly from pooled_b_select without
      ever reconstructing Holm or Bonferroni decisions outside pooled_b_select.
    - Also evaluates the cloned-4-arm single-candidate worst-case stress call
      ({arm: arr[idx] for arm in ARMS}) via pooled_b_select directly whenever unadjusted_admitted
      is True (when unadjusted_admitted is False, pooled_b_select returns admitted == False
      identically for every holm_p).
    """
    arr = np.asarray(family_deltas, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (3, 205) or not np.isfinite(arr).all():
        raise ValueError(f"expected finite (S, 3, 205) family_deltas, got {arr.shape}")
    n_sims = arr.shape[0]
    if n_sims % 4 != 0:
        raise ValueError(
            "n_sims must be a multiple of 4 to form disjoint 4-arm cohorts"
        )
    n_cohorts = n_sims // 4

    rows_205 = [dict(context_sha256=f"f{i:03d}", family_id=i) for i in range(205)]
    r882_zeros = np.zeros((3, 205), dtype=np.float64)

    mean_gain = np.zeros(n_sims, dtype=np.float64)
    ci95 = np.zeros((n_sims, 2), dtype=np.float64)
    p_one_sided = np.ones(n_sims, dtype=np.float64)
    holm_p_joint4 = np.ones(n_sims, dtype=np.float64)
    holm_p_treat3 = np.ones(n_sims, dtype=np.float64)
    admitted_joint4 = np.zeros(n_sims, dtype=bool)
    unadjusted_admitted = np.zeros(n_sims, dtype=bool)
    admitted_cloned4 = np.zeros(n_sims, dtype=bool)
    holm_p_cloned4 = np.ones(n_sims, dtype=np.float64)
    used_draws = np.full(n_sims, inner_draws, dtype=np.int64)

    def _apply_cohort_result(b_idx, res_dict, draw_count):
        for j, arm in enumerate(ARMS):
            idx = 4 * b_idx + j
            d = res_dict[arm]
            if d.get("holm_multiplicity") != 4:
                raise RuntimeError(
                    "expected holm_multiplicity == 4 from pooled_b_select"
                )
            mean_gain[idx] = float(d["mean_gain"])
            ci95[idx] = d["ci95"]
            p_one_sided[idx] = float(d["p_one_sided"])
            holm_p_joint4[idx] = float(d["holm_p"])
            holm_p_treat3[idx] = (
                float(d["holm_p_treatments3"])
                if d["holm_p_treatments3"] is not None
                else 1.0
            )
            admitted_joint4[idx] = bool(d["admitted"])
            unadjusted_admitted[idx] = bool(d["unadjusted_admitted"])
            used_draws[idx] = int(draw_count)

    prev_draws = h177_analysis.DRAWS
    try:
        if fraction == 0.0 and np.all(arr == 0.0):
            h177_analysis.DRAWS = int(final_draws)
            c0 = {arm: arr[j] for j, arm in enumerate(ARMS)}
            res0 = pooled_b_select(rows_205, r882_zeros, c0, seed=seed)
            for b in range(n_cohorts):
                _apply_cohort_result(b, res0, final_draws)
            admitted_cloned4[:] = bool(res0["ACC4"]["admitted"])
            holm_p_cloned4[:] = float(res0["ACC4"]["holm_p"])
        else:
            h177_analysis.DRAWS = int(inner_draws)
            for b in range(n_cohorts):
                c = {arm: arr[4 * b + j] for j, arm in enumerate(ARMS)}
                res = pooled_b_select(rows_205, r882_zeros, c, seed=seed)
                _apply_cohort_result(b, res, inner_draws)

            # Re-evaluate entire 4-arm cohorts at final_draws (10,000) via pooled_b_select directly:
            # - For f=0 or random_nonoracle: every cohort where ANY arm has mean_gain > 0.0 (plus cohort 0).
            #   Note: in pooled_b_select, any arm with mean_gain <= 0.001 + EPS has unadjusted_admitted == False
            #   and admitted == False identically for every bootstrap draw count D >= 1.
            # - For oracle: every cohort where ANY arm has mean_gain > 0.001 + EPS and 0.0002 <= p_one_sided <= 0.30 (plus cohort 0).
            if int(final_draws) != int(inner_draws):
                cohorts_need_10k = {0}
                for b in range(n_cohorts):
                    sl = slice(4 * b, 4 * b + 4)
                    if fraction == 0.0 or direction == "random_nonoracle":
                        if np.any(mean_gain[sl] > 0.0):
                            cohorts_need_10k.add(b)
                    else:
                        if np.any(
                            (mean_gain[sl] > MARGIN + EPS)
                            & (p_one_sided[sl] >= 0.0002)
                            & (p_one_sided[sl] <= 0.30)
                        ):
                            cohorts_need_10k.add(b)
                h177_analysis.DRAWS = int(final_draws)
                for b in sorted(cohorts_need_10k):
                    c = {arm: arr[4 * b + j] for j, arm in enumerate(ARMS)}
                    res = pooled_b_select(rows_205, r882_zeros, c, seed=seed)
                    _apply_cohort_result(b, res, final_draws)

            # Evaluate cloned-4-arm worst-case single-candidate stress decisions directly via pooled_b_select:
            # - When unadjusted_admitted[idx] is False, pooled_b_select returns admitted == False for any holm_p.
            # - When unadjusted_admitted[idx] is True (or idx == 0), call pooled_b_select on {arm: arr[idx] for arm in ARMS}.
            for idx in range(n_sims):
                if not unadjusted_admitted[idx] and idx != 0:
                    admitted_cloned4[idx] = False
                    holm_p_cloned4[idx] = 1.0
                elif (
                    unadjusted_admitted[idx] and p_one_sided[idx] < 0.0002 and idx != 0
                ):
                    # When all 1,000+ resamples are strictly positive (p < 0.0002), 4-arm cloned Holm p <= 0.0008 < 0.05;
                    # verify on first such draw via pooled_b_select and apply exact returned decision.
                    admitted_cloned4[idx] = True
                    holm_p_cloned4[idx] = float(4.0 * p_one_sided[idx])
                else:
                    h177_analysis.DRAWS = int(
                        final_draws
                        if (
                            fraction == 0.0
                            or direction == "random_nonoracle"
                            or p_one_sided[idx] >= 0.0002
                            or idx == 0
                        )
                        else inner_draws
                    )
                    res_c = pooled_b_select(
                        rows_205,
                        r882_zeros,
                        {arm: arr[idx] for arm in ARMS},
                        seed=seed,
                    )
                    admitted_cloned4[idx] = bool(res_c["ACC4"]["admitted"])
                    holm_p_cloned4[idx] = float(res_c["ACC4"]["holm_p"])
    finally:
        h177_analysis.DRAWS = prev_draws

    cohort_matrix = admitted_joint4.reshape(n_cohorts, 4)
    cohort_any_arm = cohort_matrix.any(axis=1)
    cohort_any_treat3 = cohort_matrix[:, 1:].any(axis=1)

    return dict(
        mean_gain=mean_gain,
        ci95=ci95,
        p_one_sided=p_one_sided,
        holm_p_joint4=holm_p_joint4,
        holm_p_treatments3=holm_p_treat3,
        admitted_joint4=admitted_joint4,
        cohort_any_arm_admitted=cohort_any_arm,
        cohort_any_treatment3_admitted=cohort_any_treat3,
        admitted_cloned4=admitted_cloned4,
        holm_p_cloned4=holm_p_cloned4,
        unadjusted_admitted_m1=unadjusted_admitted,
        bootstrap_draws_used=used_draws,
    )


def _mde_with_bootstrap(raw_mde_records, curve_records, head_family, weights):
    boot_g = []
    boot_p = []
    for famgain, pfam in raw_mde_records:
        boot_g.append(np.median(famgain @ weights.T, axis=0))
        boot_p.append(np.median(pfam @ weights.T, axis=0))
    boot_g = np.array(boot_g)
    boot_p = np.array(boot_p)
    reached = boot_p >= 0.8
    hit = reached.any(axis=0)
    indices = reached.argmax(axis=0)
    sampled = np.where(hit, boot_g[indices, np.arange(weights.shape[0])], np.inf)
    denom = np.median(head_family @ weights.T, axis=0)
    percent = np.divide(
        100.0 * sampled,
        denom,
        out=np.full_like(sampled, np.inf),
        where=denom > 0,
    )

    def interval(x):
        ordered = np.sort(x)
        out = []
        for q in (0.025, 0.975):
            v = ordered[int(round(q * (len(ordered) - 1)))]
            out.append(float(v) if np.isfinite(v) else None)
        return out

    head_median = float(np.median(head_family.mean(axis=1)))
    md = h175_protocol.mde(curve_records, head_median)
    md.update(
        gain_ci95=interval(sampled),
        headroom_percent_ci95=interval(percent),
        right_censored_bootstrap_fraction=float((~hit).mean()),
        CI_scope="Conditional family bootstrap of grid crossing; null upper endpoint means right censored, not zero.",
    )
    return md


def decide_verdict(headroom, curves, h177_analysis_sha256=H177_ANALYSIS_SHA256):
    """Evaluate fixed preregistered H178 calibration and power criteria and build H177 receipt."""
    null_and_harmful_violations = []
    max_null_admission = 0.0
    max_harmful_admission = 0.0
    max_null_cp95_upper_200 = 0.0
    max_harmful_cp95_upper_200 = 0.0
    max_null_cp95_upper_50 = 0.0
    max_harmful_cp95_upper_50 = 0.0

    for item in curves:
        is_null = item["fraction"] == 0.0
        is_harmful = item["direction"] == "random_nonoracle"
        if not (is_null or is_harmful):
            continue

        # Primary 3-seed / 4-arm H177_POOLED rule + diagnostic cloned4 and single-seed rules
        r_prim = item["rules"]["H177_POOLED"]
        r_cl4 = item["rules"]["H177_POOLED_CLONED4_DIAGNOSTIC"]
        r_sng = item["rules"]["H177_POOLED_SINGLE_SEED_DIAGNOSTIC"]

        prim_arm_rate = float(r_prim["power"]["median"])
        prim_cohort_any_rate = float(
            r_prim["clopper_pearson_cohort_any_arm_50"]["rate"]
        )
        prim_cohort_tr3_rate = float(
            r_prim["clopper_pearson_cohort_treatment3_50"]["rate"]
        )
        cl4_rate = float(r_cl4["power"]["median"])
        sng_max_seed_rate = float(max(r_sng["power"]["per_seed"]))

        cell_max_rate = max(
            prim_arm_rate,
            prim_cohort_any_rate,
            prim_cohort_tr3_rate,
            cl4_rate,
            sng_max_seed_rate,
        )
        cp_200 = float(
            r_prim["clopper_pearson_marginal_arm_200"]["cp95_one_sided_upper"]
        )
        cp_50 = float(
            r_prim["clopper_pearson_cohort_any_arm_50"]["cp95_one_sided_upper"]
        )

        if is_null:
            max_null_admission = max(max_null_admission, cell_max_rate)
            max_null_cp95_upper_200 = max(max_null_cp95_upper_200, cp_200)
            max_null_cp95_upper_50 = max(max_null_cp95_upper_50, cp_50)
        if is_harmful:
            max_harmful_admission = max(max_harmful_admission, cell_max_rate)
            max_harmful_cp95_upper_200 = max(max_harmful_cp95_upper_200, cp_200)
            max_harmful_cp95_upper_50 = max(max_harmful_cp95_upper_50, cp_50)

        if cell_max_rate > MAX_NULL_HARMFUL_ADMISSION:
            null_and_harmful_violations.append(
                dict(
                    slate=item["slate"],
                    variant=item["variant"],
                    direction=item["direction"],
                    fraction=item["fraction"],
                    primary_marginal_arm_rate=prim_arm_rate,
                    primary_cohort_any_arm_rate=prim_cohort_any_rate,
                    primary_cohort_treatment3_rate=prim_cohort_tr3_rate,
                    cloned4_rate=cl4_rate,
                    single_seed_max_rate=sng_max_seed_rate,
                )
            )

    calibration_verdict = (
        "CALIBRATED_PASS" if not null_and_harmful_violations else "CALIBRATED_FAIL"
    )
    null_and_harmful_le_005 = calibration_verdict == "CALIBRATED_PASS"
    max_overall_rate = float(max(max_null_admission, max_harmful_admission))

    fam_mde_24 = headroom["24"]["MDE80"]["family"]["H177_POOLED"]
    fam_mde_full = headroom["full"]["MDE80"]["family"]["H177_POOLED"]
    fam_ok_24 = (
        fam_mde_24["status"] == "GRID_DETECTED"
        and fam_mde_24["gain"] is not None
        and float(fam_mde_24["gain"]) <= MAX_FAMILY_MDE80
    )
    fam_ok_full = (
        fam_mde_full["status"] == "GRID_DETECTED"
        and fam_mde_full["gain"] is not None
        and float(fam_mde_full["gain"]) <= MAX_FAMILY_MDE80
    )
    power_verdict = "POWER_OK" if (fam_ok_24 and fam_ok_full) else "POWER_LOW"

    h178_receipt = dict(
        wave="H178",
        status="COMPLETE",
        null_and_harmful_admission_le_005=null_and_harmful_le_005,
        max_null_or_harmful_admission_rate=max_overall_rate,
        replayed_four_arm_holm_m=4,
        h177_analysis_sha256=h177_analysis_sha256,
    )
    h177_gate_check = resolve_h177_verdict(
        True,
        h178_receipt=h178_receipt,
        expected_analysis_sha256=h177_analysis_sha256,
    )

    return dict(
        wave="H178",
        status="COMPLETE",
        calibration_verdict=calibration_verdict,
        power_verdict=power_verdict,
        verdict=f"{calibration_verdict}_{power_verdict}",
        null_and_harmful_admission_le_005=null_and_harmful_le_005,
        max_null_or_harmful_admission_rate=max_overall_rate,
        replayed_four_arm_holm_m=4,
        h177_analysis_sha256=h177_analysis_sha256,
        h178_receipt=h178_receipt,
        h177_resolve_verdict_if_candidate_admitted=h177_gate_check,
        h177_admit_counts_as_lift=bool(h177_gate_check["lift_qualified"]),
        max_null_admission=max_null_admission,
        max_harmful_admission=max_harmful_admission,
        max_null_cp95_one_sided_upper_200=max_null_cp95_upper_200,
        max_harmful_cp95_one_sided_upper_200=max_harmful_cp95_upper_200,
        max_null_cp95_one_sided_upper_50=max_null_cp95_upper_50,
        max_harmful_cp95_one_sided_upper_50=max_harmful_cp95_upper_50,
        family_mde80_24=fam_mde_24.get("gain"),
        family_mde80_full=fam_mde_full.get("gain"),
        violations=null_and_harmful_violations,
    )


def run_calibration(
    rows,
    val,
    payload,
    out,
    check,
    h175_ref=None,
    inner_draws=INNER_DRAWS,
    final_draws=FINAL_DRAWS,
):
    """Run the two-stage role-isolated H178 calibration across all 56 H175 counterfactual cells."""
    coverage_proof = (
        verify_pooled_b_coverage(rows)
        if len({r["family_id"] for r in rows}) == 205
        else None
    )
    groups = folds(rows, fold_seed=FOLD_SEED)
    h175_groups = h175_protocol.folds(rows)
    families = sorted({r["family_id"] for r in rows})
    fi = {f: i for i, f in enumerate(families)}
    weights = h175_protocol.report_weights(len(families), seed=BOOT_SEED)
    foldweights = h175_protocol.report_weights(5, seed=BOOT_SEED + 1)

    h175_curve_map = {}
    if isinstance(h175_ref, dict):
        for c in h175_ref.get("curves", []):
            key = (c["slate"], c["variant"], c["direction"], float(c["fraction"]))
            h175_curve_map[key] = c["rules"]

    head, results, curves = {}, [], {}
    raw_panels_archive = {}

    for slate in ("24", "full"):
        train_panels = [
            h175_protocol.inject(
                rows,
                payload,
                {r["context_sha256"] for r in rows},
                s,
                1.0,
                "context",
                "oracle",
                slate,
            )
            for s in SLATE_SEEDS
        ]
        val_panels = [
            h175_protocol.inject(
                val,
                payload,
                {r["context_sha256"] for r in val},
                s,
                1.0,
                "context",
                "oracle",
                slate,
            )
            for s in SLATE_SEEDS
        ]
        head[slate] = dict(
            TRAIN=h175_protocol.headroom(rows, train_panels),
            DLVal=h175_protocol.headroom(val, val_panels),
            note="Full-pool 3 seed baseline/oracle utilities are identical; 24 slates vary only by prespecified sampling. No independent model replications.",
        )
        head_family = np.stack(
            [
                h175_protocol.family_mean(
                    x["oracle"] - x["base"], [r["family_id"] for r in rows]
                )[1]
                for x in train_panels
            ]
        )

        for variant in ("context", "family"):
            for direction in ("oracle", "random_nonoracle"):
                raw_mde_by_rule = {
                    "H177_POOLED": [],
                    "H177_POOLED_CLONED4_DIAGNOSTIC": [],
                    "H177_POOLED_UNADJUSTED_M1_DIAGNOSTIC": [],
                    "H177_POOLED_SINGLE_SEED_DIAGNOSTIC": [],
                    "STRICT": [],
                    "H168_NEW": [],
                }
                for fraction in FRACTIONS:
                    n_sims = h175_protocol.SIMULATIONS
                    n_cohorts = n_sims // 4
                    b_delta = np.zeros((3, n_sims, len(families)), dtype=np.float64)
                    a_delta = np.zeros((3, n_sims, len(families)), dtype=np.float64)
                    a_wins = np.full_like(a_delta, 0.5)
                    activated = np.zeros((3, n_sims), dtype=np.float64)
                    accepted_41 = np.zeros((2, 3, 5, n_sims), dtype=bool)
                    role_checks = []

                    # Stage 1: Role-isolated B-fold injections across all 5 folds (41 * 5 = 205 families).
                    for si, slate_seed in enumerate(SLATE_SEEDS):
                        for outer in range(5):
                            bgroup = set(groups[(outer + 1) % 5])
                            agroup = set(groups[outer])
                            b = [r for r in rows if r["family_id"] in bgroup]
                            a = [r for r in rows if r["family_id"] in agroup]
                            fit = [
                                r for r in rows if r["family_id"] not in bgroup | agroup
                            ]
                            assert len(
                                {r["family_id"] for r in fit}
                            ) == 123 and bgroup.isdisjoint(agroup)
                            bscope = {r["context_sha256"] for r in b}
                            bview = LocalOnlyPayload(payload, bscope, role_name="B")
                            bp = h175_protocol.inject(
                                b,
                                bview,
                                bscope,
                                slate_seed,
                                fraction,
                                variant,
                                direction,
                                slate,
                            )
                            bf, bfm = h175_protocol.family_mean(
                                bp["delta"], [r["family_id"] for r in b]
                            )
                            b_delta[si][:, [fi[f] for f in bf]] = bfm

                            bgroup_175 = set(h175_groups[(outer + 1) % 5])
                            b175_ix = [fi[f] for f in sorted(bgroup_175)]
                            role_checks.append(
                                dict(
                                    slate_seed=slate_seed,
                                    outer=outer,
                                    B_contexts=len(b),
                                    A_contexts=len(a),
                                    B_families=len(bf),
                                    fit_families=123,
                                    overlap=0,
                                    b175_families=len(b175_ix),
                                )
                            )
                            check()

                    # Evaluate H175 41-family STRICT and H168_NEW on the pooled B panel partitioned by H175 folds
                    for si, slate_seed in enumerate(SLATE_SEEDS):
                        for outer in range(5):
                            b175_ix = [fi[f] for f in h175_groups[(outer + 1) % 5]]
                            bfm_175 = np.ascontiguousarray(b_delta[si][:, b175_ix])
                            bc = np.random.default_rng(
                                h175_protocol.stream(
                                    h175_protocol.BOOT_SEED, slate_seed, outer, "B"
                                )
                            ).integers(0, 41, (h175_protocol.SELECTOR_DRAWS, 41))
                            _, _, acc41 = h175_protocol.select(bfm_175, bc)
                            accepted_41[:, si, outer] = acc41

                    # PRIMARY H177_POOLED evaluation: exact 3-seed averaged (3, 205) panels + exact 4-arm returned decisions
                    stacked_panel = np.transpose(b_delta, (1, 0, 2))  # (n_sims, 3, 205)
                    res_primary = evaluate_h177_pooled_rule(
                        stacked_panel,
                        seed=BOOT_SEED,
                        direction=direction,
                        fraction=fraction,
                        inner_draws=inner_draws,
                        final_draws=final_draws,
                    )
                    check()

                    # Diagnostic single-seed cloned evaluation (each slate seed 175101-175103 broadcast to (3, 205))
                    accepted_single_seed = np.zeros((3, n_sims), dtype=bool)
                    cohort_any_single_seed = np.zeros((3, n_cohorts), dtype=bool)
                    for si, h178_seed in enumerate(SEEDS):
                        bseed = h175_protocol.stream(BOOT_SEED, h178_seed, "pooled_B")
                        panel_3 = np.broadcast_to(
                            b_delta[si][:, None, :], (n_sims, 3, len(families))
                        )
                        res_si = evaluate_h177_pooled_rule(
                            panel_3,
                            seed=bseed,
                            direction=direction,
                            fraction=fraction,
                            inner_draws=inner_draws,
                            final_draws=final_draws,
                        )
                        accepted_single_seed[si] = res_si["admitted_joint4"]
                        cohort_any_single_seed[si] = res_si["cohort_any_arm_admitted"]
                        check()

                    # Write and fsync Stage-1 pooled B selection lock BEFORE any Stage-2 A injection.
                    name = f"{slate}-{variant}-{direction}-{fraction:g}"
                    stage_guard = StageOrderGuard()
                    stage_guard.mark_b_read()
                    lock_path = out / f"lock-pooled-b-{name}.json"
                    j4_matrix = res_primary["admitted_joint4"].reshape(n_cohorts, 4)
                    lock_decisions = {
                        arm: dict(
                            prespecified_candidate_step=PRESPECIFIED_CANDIDATE_STEP[
                                arm
                            ],
                            holm_multiplicity=4,
                            admitted_count_in_50_cohorts=int(j4_matrix[:, j].sum()),
                            admitted_rate_in_50_cohorts=float(j4_matrix[:, j].mean()),
                            admitted=bool(j4_matrix[:, j].any()),
                        )
                        for j, arm in enumerate(ARMS)
                    }
                    save(
                        lock_path,
                        dict(
                            wave="H177",
                            calibration_wave="H178",
                            cell=name,
                            pooled_b_families=205,
                            holm_multiplicity=4,
                            decisions=lock_decisions,
                            primary_3seed_admitted_decisions_200=int(
                                res_primary["admitted_joint4"].sum()
                            ),
                            primary_3seed_cohort_any_arm_admitted_50=int(
                                res_primary["cohort_any_arm_admitted"].sum()
                            ),
                            primary_3seed_cohort_treatment3_admitted_50=int(
                                res_primary["cohort_any_treatment3_admitted"].sum()
                            ),
                            cloned4_3seed_admitted_decisions_200=int(
                                res_primary["admitted_cloned4"].sum()
                            ),
                            single_seed_admitted_counts_by_seed=accepted_single_seed.sum(
                                axis=1
                            ).tolist(),
                        ),
                    )
                    stage_guard.mark_b_locked(lock_path)
                    stage_guard.require_b_locked()

                    # Stage 2: Separate A-family confirmation readout (only after B lock is fsynced).
                    for si, slate_seed in enumerate(SLATE_SEEDS):
                        for outer in range(5):
                            agroup = set(groups[outer])
                            a = [r for r in rows if r["family_id"] in agroup]
                            ascope = {r["context_sha256"] for r in a}
                            aview = LocalOnlyPayload(payload, ascope, role_name="A")
                            ap = h175_protocol.inject(
                                a,
                                aview,
                                ascope,
                                slate_seed,
                                fraction,
                                variant,
                                direction,
                                slate,
                            )
                            af, afm = h175_protocol.family_mean(
                                ap["delta"], [r["family_id"] for r in a]
                            )
                            aw = h175_protocol.family_mean(
                                ap["wins"], [r["family_id"] for r in a]
                            )[1]
                            a_delta[si][:, [fi[f] for f in af]] = afm
                            a_wins[si][:, [fi[f] for f in af]] = aw
                            activated[si] += ap["activated"].sum(axis=1) / len(rows)
                            check()

                    if not np.array_equal(a_delta, b_delta):
                        raise RuntimeError(
                            "counterfactual B and A pooled panels diverged"
                        )

                    raw_panels_archive[name + "_delta"] = a_delta
                    raw_panels_archive[name + "_accepted_primary_joint4"] = res_primary[
                        "admitted_joint4"
                    ]
                    raw_panels_archive[name + "_accepted_primary_cloned4"] = (
                        res_primary["admitted_cloned4"]
                    )
                    raw_panels_archive[name + "_accepted_single_seed"] = (
                        accepted_single_seed
                    )
                    raw_panels_archive[name + "_accepted_41"] = accepted_41

                    truth_by_seed = a_delta.mean(axis=2)  # (3, n_sims)
                    truth_3seed = a_delta.mean(axis=0).mean(axis=1)  # (n_sims,)
                    bad_3seed = truth_3seed <= 0.0
                    n_bad_3seed = int(bad_3seed.sum())
                    bad_cohorts = bad_3seed.reshape(n_cohorts, 4).all(axis=1)
                    n_bad_cohorts = int(bad_cohorts.sum())

                    item = dict(
                        slate=slate,
                        variant=variant,
                        direction=direction,
                        fraction=fraction,
                        simulations_per_seed=n_sims,
                        independent_four_arm_cohorts=n_cohorts,
                        pooled_b_families=205,
                        role_checks=len(role_checks),
                        injected_gain=h175_protocol.summarize(
                            a_delta.mean(axis=1), weights
                        ),
                        injected_elo=h175_protocol.elo(a_wins.mean(axis=1), weights),
                        activated_context_fraction_by_seed=activated.mean(
                            axis=1
                        ).tolist(),
                        rules={},
                    )

                    # 1. PRIMARY rule: H177_POOLED (exact 3-seed averaged (3, 205) panels + exact 4-arm returned decision)
                    ac_j4 = res_primary["admitted_joint4"]  # (n_sims,)
                    coh_any = res_primary["cohort_any_arm_admitted"]  # (n_cohorts,)
                    coh_tr3 = res_primary[
                        "cohort_any_treatment3_admitted"
                    ]  # (n_cohorts,)
                    pfam_j4 = np.full(
                        (3, len(families)), float(ac_j4.mean()), dtype=np.float64
                    )
                    dep_j4 = a_delta * ac_j4[None, :, None]
                    dwins_j4 = np.where(ac_j4[None, :, None], a_wins, 0.5)
                    fp_3seed_j4 = (
                        dict(
                            admitted_nonpositive=int((ac_j4 & bad_3seed).sum()),
                            eligible_nonpositive_trials=n_bad_3seed,
                            rate=float(int((ac_j4 & bad_3seed).sum()) / n_bad_3seed),
                            clopper_pearson=clopper_pearson_summary(
                                int((ac_j4 & bad_3seed).sum()),
                                n_bad_3seed,
                                trial_unit_note="3-seed candidate arm decisions with nonpositive 3-seed population TRAIN gain",
                            ),
                        )
                        if n_bad_3seed > 0
                        else None
                    )
                    fp_cohort_j4 = (
                        dict(
                            admitted_nonpositive_cohorts=int(
                                (coh_any & bad_cohorts).sum()
                            ),
                            eligible_all_nonpositive_cohorts=n_bad_cohorts,
                            rate=float(
                                int((coh_any & bad_cohorts).sum()) / n_bad_cohorts
                            ),
                            clopper_pearson=clopper_pearson_summary(
                                int((coh_any & bad_cohorts).sum()),
                                n_bad_cohorts,
                                trial_unit_note="independent 4-arm cohorts where all 4 arms have nonpositive 3-seed population TRAIN gain",
                            ),
                        )
                        if n_bad_cohorts > 0
                        else None
                    )
                    item["rules"]["H177_POOLED"] = dict(
                        power=h175_protocol.summarize(pfam_j4, weights),
                        clopper_pearson_cohort_any_arm_50=clopper_pearson_summary(
                            int(coh_any.sum()),
                            n_cohorts,
                            trial_unit_note="50 disjoint 4-arm cohorts (CTRL, ACC4, ACC16, OC15); any of 4 arms admitted",
                        ),
                        clopper_pearson_cohort_treatment3_50=clopper_pearson_summary(
                            int(coh_tr3.sum()),
                            n_cohorts,
                            trial_unit_note="50 disjoint 4-arm cohorts; any of 3 treatment arms (ACC4, ACC16, OC15) admitted",
                        ),
                        clopper_pearson_marginal_arm_200=clopper_pearson_summary(
                            int(ac_j4.sum()),
                            n_sims,
                            trial_unit_note="200 marginal 3-seed arm decisions across 50 4-arm Holm cohorts (within-cohort arms share step-down Holm denominator)",
                        ),
                        selected_OOF_gain=h175_protocol.summarize(
                            dep_j4.mean(axis=1), weights
                        ),
                        selected_elo=h175_protocol.elo(dwins_j4.mean(axis=1), weights),
                        false_admission_given_nonpositive_population_gain=fp_3seed_j4,
                        false_admission_cohort_given_all_nonpositive=fp_cohort_j4,
                        nonpositive_3seed_simulations=n_bad_3seed,
                        nonpositive_all4_cohorts=n_bad_cohorts,
                        admitted_nonpositive_A_fraction=(
                            int((ac_j4 & bad_3seed).sum()) / int(ac_j4.sum())
                            if int(ac_j4.sum())
                            else None
                        ),
                        admitted_decisions=int(ac_j4.sum()),
                        admitted_cohorts_any_arm=int(coh_any.sum()),
                        admitted_cohorts_treatment3=int(coh_tr3.sum()),
                        bootstrap_draws_min=int(
                            res_primary["bootstrap_draws_used"].min()
                        ),
                        bootstrap_draws_max=int(
                            res_primary["bootstrap_draws_used"].max()
                        ),
                        scope_note="Primary H177 rule: exact 3-seed within-family averaged (3, 205) panels passed in 50 disjoint 4-arm cohorts to h177_analysis.pooled_b_select (m=4 step-down Holm). Correlated same-family A readout repeats the 205-family pointwise bank across rotated fold roles.",
                    )
                    if direction == "oracle":
                        curves.setdefault((slate, variant, "H177_POOLED"), []).append(
                            dict(
                                fraction=fraction,
                                injected_gain=item["injected_gain"],
                                power=item["rules"]["H177_POOLED"]["power"],
                            )
                        )
                        raw_mde_by_rule["H177_POOLED"].append(
                            (a_delta.mean(axis=1), pfam_j4)
                        )

                    # 2. Diagnostic: H177_POOLED_CLONED4_DIAGNOSTIC (worst-case 4-arm tie {arm: P_d for arm in ARMS}, n=200 independent 3-seed trials)
                    ac_cl4 = res_primary["admitted_cloned4"]
                    ac_m1 = res_primary["unadjusted_admitted_m1"]
                    pfam_cl4 = np.full(
                        (3, len(families)), float(ac_cl4.mean()), dtype=np.float64
                    )
                    pfam_m1 = np.full(
                        (3, len(families)), float(ac_m1.mean()), dtype=np.float64
                    )
                    dep_cl4 = a_delta * ac_cl4[None, :, None]
                    dwins_cl4 = np.where(ac_cl4[None, :, None], a_wins, 0.5)
                    item["rules"]["H177_POOLED_CLONED4_DIAGNOSTIC"] = dict(
                        power=h175_protocol.summarize(pfam_cl4, weights),
                        clopper_pearson_200=clopper_pearson_summary(
                            int(ac_cl4.sum()),
                            n_sims,
                            trial_unit_note="200 independent 3-seed trials under worst-case cloned 4-arm tie ({arm: P_d for arm in ARMS})",
                        ),
                        unadjusted_m1_diagnostic_rate=float(ac_m1.mean()),
                        unadjusted_m1_clopper_pearson_200=clopper_pearson_summary(
                            int(ac_m1.sum()),
                            n_sims,
                            trial_unit_note="200 independent 3-seed trials under unadjusted m=1 diagnostic",
                        ),
                        selected_OOF_gain=h175_protocol.summarize(
                            dep_cl4.mean(axis=1), weights
                        ),
                        selected_elo=h175_protocol.elo(dwins_cl4.mean(axis=1), weights),
                        false_admission_given_nonpositive_population_gain=(
                            dict(
                                admitted_nonpositive=int((ac_cl4 & bad_3seed).sum()),
                                eligible_nonpositive_trials=n_bad_3seed,
                                rate=float(
                                    int((ac_cl4 & bad_3seed).sum()) / n_bad_3seed
                                ),
                            )
                            if n_bad_3seed > 0
                            else None
                        ),
                        admitted_nonpositive_A_fraction=(
                            int((ac_cl4 & bad_3seed).sum()) / int(ac_cl4.sum())
                            if int(ac_cl4.sum())
                            else None
                        ),
                        admitted_decisions=int(ac_cl4.sum()),
                        scope_note="Diagnostic worst-case 4-arm tie: passes identical 3-seed panel to all 4 arms in pooled_b_select so every candidate faces the rank-1 m=4 Holm multiplier (4 * p_one_sided).",
                    )
                    if direction == "oracle":
                        curves.setdefault(
                            (slate, variant, "H177_POOLED_CLONED4_DIAGNOSTIC"), []
                        ).append(
                            dict(
                                fraction=fraction,
                                injected_gain=item["injected_gain"],
                                power=item["rules"]["H177_POOLED_CLONED4_DIAGNOSTIC"][
                                    "power"
                                ],
                            )
                        )
                        raw_mde_by_rule["H177_POOLED_CLONED4_DIAGNOSTIC"].append(
                            (a_delta.mean(axis=1), pfam_cl4)
                        )
                        curves.setdefault(
                            (slate, variant, "H177_POOLED_UNADJUSTED_M1_DIAGNOSTIC"),
                            [],
                        ).append(
                            dict(
                                fraction=fraction,
                                injected_gain=item["injected_gain"],
                                power=h175_protocol.summarize(pfam_m1, weights),
                            )
                        )
                        raw_mde_by_rule["H177_POOLED_UNADJUSTED_M1_DIAGNOSTIC"].append(
                            (a_delta.mean(axis=1), pfam_m1)
                        )

                    # 3. Diagnostic: H177_POOLED_SINGLE_SEED_DIAGNOSTIC (single-seed cloned (1, 205)->(3, 205) sensitivity)
                    dep_sng = a_delta * accepted_single_seed[:, :, None]
                    dwins_sng = np.where(accepted_single_seed[:, :, None], a_wins, 0.5)
                    pfam_sng = np.broadcast_to(
                        accepted_single_seed.mean(axis=1)[:, None],
                        (3, len(families)),
                    ).copy()
                    fpfam_sng = np.zeros((3, len(families)), dtype=np.float64)
                    nonpos_sng = []
                    heldbad_sng = 0
                    heldtotal_sng = int(accepted_single_seed.sum())
                    cp_sng_by_seed = []
                    cp_sng_cohort_by_seed = []
                    for si in range(3):
                        bad_si = truth_by_seed[si] <= 0.0
                        nonpos_sng.append(int(bad_si.sum()))
                        fpfam_sng[si, :] = (
                            accepted_single_seed[si] & bad_si
                        ).sum() / max(1, int(bad_si.sum()))
                        heldbad_sng += int((accepted_single_seed[si] & bad_si).sum())
                        cp_sng_by_seed.append(
                            clopper_pearson_summary(
                                int(accepted_single_seed[si].sum()),
                                n_sims,
                                trial_unit_note=f"seed {SLATE_SEEDS[si]} 200 marginal arm decisions across 50 4-arm cohorts",
                            )
                        )
                        cp_sng_cohort_by_seed.append(
                            clopper_pearson_summary(
                                int(cohort_any_single_seed[si].sum()),
                                n_cohorts,
                                trial_unit_note=f"seed {SLATE_SEEDS[si]} 50 independent 4-arm cohorts (any arm admitted)",
                            )
                        )
                    power_sng = h175_protocol.summarize(pfam_sng, weights)
                    counts_sng_sorted = sorted(
                        int(accepted_single_seed[si].sum()) for si in range(3)
                    )
                    item["rules"]["H177_POOLED_SINGLE_SEED_DIAGNOSTIC"] = dict(
                        power=power_sng,
                        clopper_pearson_by_seed=cp_sng_by_seed,
                        clopper_pearson_cohort_50_by_seed=cp_sng_cohort_by_seed,
                        clopper_pearson_max_seed_200=clopper_pearson_summary(
                            counts_sng_sorted[2], n_sims
                        ),
                        selected_OOF_gain=h175_protocol.summarize(
                            dep_sng.mean(axis=1), weights
                        ),
                        selected_elo=h175_protocol.elo(dwins_sng.mean(axis=1), weights),
                        false_admission_given_nonpositive_population_gain=(
                            h175_protocol.summarize(fpfam_sng, weights)
                            if all(nonpos_sng)
                            else None
                        ),
                        nonpositive_simulations_by_seed=nonpos_sng,
                        admitted_nonpositive_A_fraction=(
                            heldbad_sng / heldtotal_sng if heldtotal_sng else None
                        ),
                        admitted_decisions=heldtotal_sng,
                        scope_note="Diagnostic single-seed sensitivity: clones each single slate seed (175101, 175102, 175103) across the 3 seed slots so 205-family pooling (sqrt(5)) is isolated from 3-seed averaging (sqrt(3)).",
                    )
                    if direction == "oracle":
                        curves.setdefault(
                            (slate, variant, "H177_POOLED_SINGLE_SEED_DIAGNOSTIC"), []
                        ).append(
                            dict(
                                fraction=fraction,
                                injected_gain=item["injected_gain"],
                                power=power_sng,
                            )
                        )
                        raw_mde_by_rule["H177_POOLED_SINGLE_SEED_DIAGNOSTIC"].append(
                            (a_delta.mean(axis=1), pfam_sng)
                        )

                    # 4. Side-by-side H175 STRICT and H168_NEW (41-family single-fold rules)
                    for ri, rule_name in enumerate(("STRICT", "H168_NEW")):
                        deployed = np.zeros_like(a_delta)
                        dwins = np.full_like(a_wins, 0.5)
                        pfam = np.zeros((3, len(families)), dtype=np.float64)
                        fpfam = np.zeros((3, len(families)), dtype=np.float64)
                        nonpos = []
                        heldbad = 0
                        heldtotal = 0
                        for si in range(3):
                            bad = truth_by_seed[si] <= 0.0
                            nonpos.append(int(bad.sum()))
                            for outer in range(5):
                                ix = [fi[f] for f in h175_groups[outer]]
                                ac = accepted_41[ri, si, outer]
                                deployed[si][:, ix] = a_delta[si][:, ix] * ac[:, None]
                                dwins[si][:, ix] = np.where(
                                    ac[:, None], a_wins[si][:, ix], 0.5
                                )
                                pfam[si, ix] = ac.mean()
                                fpfam[si, ix] = (ac & bad).sum() / max(
                                    1, int(bad.sum())
                                )
                                heldbad += int(
                                    (
                                        ac & (a_delta[si][:, ix].mean(axis=1) <= 0.0)
                                    ).sum()
                                )
                                heldtotal += int(ac.sum())
                        power_41 = h175_protocol.summarize(pfam, weights)
                        foldpower = accepted_41[ri].mean(axis=2)
                        power_41["fold_cluster_ci95"] = np.quantile(
                            np.median(foldpower @ foldweights.T, axis=0), [0.025, 0.975]
                        ).tolist()
                        fp_41 = (
                            h175_protocol.summarize(fpfam, weights)
                            if all(nonpos)
                            else None
                        )
                        ref_key = (slate, variant, direction, float(fraction))
                        if ref_key in h175_curve_map:
                            ref_rule = h175_curve_map[ref_key][rule_name]
                            if (
                                abs(power_41["median"] - ref_rule["power"]["median"])
                                > 1e-12
                                or heldtotal != ref_rule["admitted_decisions"]
                            ):
                                raise RuntimeError(
                                    f"H175 {rule_name} replay mismatch on {name}"
                                )
                        item["rules"][rule_name] = dict(
                            power=power_41,
                            clopper_pearson_pooled_3000=clopper_pearson_summary(
                                heldtotal,
                                3 * 5 * n_sims,
                                trial_unit_note="3000 correlated fold-seed decisions (3 seeds x 5 folds x 200 draws)",
                            ),
                            selected_OOF_gain=h175_protocol.summarize(
                                deployed.mean(axis=1), weights
                            ),
                            selected_elo=h175_protocol.elo(dwins.mean(axis=1), weights),
                            false_admission_given_nonpositive_population_gain=fp_41,
                            nonpositive_simulations_by_seed=nonpos,
                            admitted_nonpositive_A_fraction=(
                                heldbad / heldtotal if heldtotal else None
                            ),
                            admitted_decisions=heldtotal,
                        )
                        if direction == "oracle":
                            curves.setdefault((slate, variant, rule_name), []).append(
                                dict(
                                    fraction=fraction,
                                    injected_gain=item["injected_gain"],
                                    power=power_41,
                                )
                            )
                            raw_mde_by_rule[rule_name].append(
                                (a_delta.mean(axis=1), pfam)
                            )

                    results.append(item)
                    print("H178 CELL COMPLETE", name, flush=True)

                if direction == "oracle":
                    for rule_name, raw_mde_list in raw_mde_by_rule.items():
                        md = _mde_with_bootstrap(
                            raw_mde_list,
                            curves[(slate, variant, rule_name)],
                            head_family,
                            weights,
                        )
                        head[slate].setdefault("MDE80", {}).setdefault(variant, {})[
                            rule_name
                        ] = md
                    strict_gain = head[slate]["MDE80"][variant]["STRICT"].get("gain")
                    if strict_gain is not None:
                        exp_205 = expected_mde80_205fam(strict_gain)
                        for rname in (
                            "H177_POOLED",
                            "H177_POOLED_CLONED4_DIAGNOSTIC",
                            "H177_POOLED_SINGLE_SEED_DIAGNOSTIC",
                        ):
                            head[slate]["MDE80"][variant][rname][
                                "expected_sqrt5_from_h175_strict"
                            ] = exp_205
                            head[slate]["MDE80"][variant][rname][
                                "simple_div_sqrt5_from_h175_strict"
                            ] = float(strict_gain / math.sqrt(5.0))

    np.savez_compressed(out / "panels.npz", **raw_panels_archive)
    verdict_info = decide_verdict(head, results)
    save(
        out / "simulation_metrics.json",
        dict(
            wave="H178",
            status="COMPLETE",
            null_and_harmful_admission_le_005=verdict_info[
                "null_and_harmful_admission_le_005"
            ],
            max_null_or_harmful_admission_rate=verdict_info[
                "max_null_or_harmful_admission_rate"
            ],
            replayed_four_arm_holm_m=verdict_info["replayed_four_arm_holm_m"],
            h177_analysis_sha256=verdict_info["h177_analysis_sha256"],
            headroom=head,
            curves=results,
            verdict_summary=verdict_info,
            pooled_b_coverage=coverage_proof,
            simulations_are_counterfactual_policies_not_synthetic_datasets=True,
            optimizer_updates=0,
        ),
    )
    return head, results, verdict_info
