"""Read-only H175 bank replay; no network/model fitting or live selector writes."""

import json
from pathlib import Path
import numpy as np
import science as s


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def prepare(rows, costs, utilities):
    groups = s.groups()
    weights = s.family_weights()
    foldweights = s.family_weights(5, s.BOOT_SEED + 1)
    design = {}
    for slate in ["24", "full"]:
        X = np.stack(
            [s.covariates(rows, costs, utilities, slate, seed) for seed in s.BANK_SEEDS]
        )
        cw = {}
        for si, seed in enumerate(s.SEEDS):
            h = s.strata(X[si])
            population = np.bincount(h, minlength=5) / 205
            for outer in range(5):
                fit, b, a = s.roles(groups, outer)
                count = s.counts(41, s.stream(s.BOOT_SEED, seed, outer, "selector"))
                ps, missing = s.post_weights(count, h[b], population)
                cw[si, outer] = (count / 41, ps, missing)
        design[slate] = (X, cw)
    return groups, weights, foldweights, design


def case(D, wins, slate, design, group, weights, foldweights):
    if (
        D.shape != (3, 200, 205)
        or wins.shape != D.shape
        or not np.isfinite(D).all()
        or not np.isfinite(wins).all()
        or np.any((wins < 0) | (wins > 1))
    ):
        raise ValueError("bank dimensions/finiteness")
    X, cw = design[slate]
    adjusted = np.empty((4, 3, 200, 205))
    theta = np.empty((2, 3, 5, 200, 4))
    post_fallback = []
    accepted = np.zeros((4, 3, 5, 200), bool)
    gain = np.zeros((4, 3, 5, 200))
    ci = np.zeros((4, 3, 5, 200, 2))
    bootvar = np.zeros((4, 3, 5, 200))
    error = np.zeros_like(adjusted)
    truth = D.mean(axis=2)
    for si in range(3):
        adjusted[:, si], theta[:, si], fall = s.adjusted_panel(X[si], D[si], group)
        post_fallback.append(fall.tolist())
        for outer in range(5):
            fit, b, a = s.roles(group, outer)
            for mi in range(4):
                w = cw[si, outer][1] if mi == 3 else cw[si, outer][0]
                values = D[si][:, b] if mi == 3 else adjusted[mi, si][:, b]
                if mi == 3:
                    point = adjusted[mi, si][:, b].mean(axis=1)
                    boot = values @ w.T
                    interval = np.quantile(boot, [0.025, 0.975], axis=1).T
                    accept = (point > s.MARGIN + s.EPS) & (interval[:, 0] > 0)
                    variance = boot.var(axis=1)
                else:
                    point, interval, accept, variance = s.selector(values, w)
                gain[mi, si, outer] = point
                ci[mi, si, outer] = interval
                accepted[mi, si, outer] = accept
                bootvar[mi, si, outer] = variance
                error[mi, si][:, b] = (point - truth[si])[:, None]
    summaries = {}
    strict_first = error[0].mean(axis=1)
    strict_second = (error[0] ** 2).mean(axis=1)
    for mi, method in enumerate(s.METHODS):
        power = np.empty((3, 205))
        deployed = np.zeros_like(D)
        winprob = np.full_like(D, 0.5)
        cp = np.zeros((3, 5))
        conditional = []
        negative_counts = []
        for si in range(3):
            bad = truth[si] <= 0
            negative_counts.append(int(bad.sum()))
            conditional_row = []
            for outer in range(5):
                fit, b, a = s.roles(group, outer)
                ac = accepted[mi, si, outer]
                power[si, b] = ac.mean()
                cp[si, outer] = s.cp_upper(int(ac.sum()), 200)
                # Actual deployed utility is the original policy, never its adjusted estimator.
                deployed[si][:, b] = D[si][:, b] * ac[:, None]
                conditional_row.append(
                    float((ac & bad).sum() / bad.sum()) if bad.any() else None
                )
                winprob[si][:, b] = np.where(ac[:, None], wins[si][:, b], 0.5)
            conditional.append(conditional_row)
        power_summary = s.summarize(power, weights)
        power_summary["fold_block_ci95"] = np.quantile(
            np.median(accepted[mi].mean(axis=2) @ foldweights.T, axis=0), [0.025, 0.975]
        ).tolist()
        bias_family = (adjusted[mi] - D).mean(axis=1)
        bias = s.summarize(bias_family, weights)
        bias["max_abs_per_seed"] = float(np.max(np.abs(bias_family.mean(axis=1))))
        ratio = s.variance_ratio(
            error[mi].mean(axis=1),
            (error[mi] ** 2).mean(axis=1),
            strict_first,
            strict_second,
            weights,
        )
        summaries[method] = dict(
            power=power_summary,
            CP95_upper_by_seed_fold=cp.tolist(),
            CP95_upper_max=float(cp.max()),
            CP_trials_per_fold=200,
            conditional_nonpositive_admission_by_seed_fold=conditional,
            nonpositive_population_draws_by_seed=negative_counts,
            bias_vs_STRICT=bias,
            empirical_estimator_error_variance_ratio=ratio,
            mean_conditional_bootstrap_variance_ratio=float(
                bootvar[mi].mean() / bootvar[0].mean()
            )
            if bootvar[0].mean() > 1e-18
            else 1.0,
            selected_true_gain=s.summarize(deployed.mean(axis=1), weights),
            selected_true_utility=s.summarize(
                deployed.mean(axis=1) + X[:, :, 0], weights
            ),
            selected_policy_Elo=s.two_policy_elo(winprob.mean(axis=1), weights),
            point_estimator_gain=s.summarize(adjusted[mi].mean(axis=1), weights),
        )
    return dict(
        methods=summaries,
        injected_true_gain=s.summarize(D.mean(axis=1), weights),
        poststrat_point_fallbacks=post_fallback,
        poststrat_bootstrap_missing_stratum_counts={
            str(k): v[2] for k, v in cw.items()
        },
    ), dict(
        delta=D,
        wins=wins,
        adjusted=adjusted,
        theta=theta,
        accepted=accepted,
        B_gain=gain,
        B_ci=ci,
        bootstrap_variance=bootvar,
        error=error,
    )


def target_bank(rows, costs, utilities, slate, variant, target):
    """New seed181 streams, same cached real candidate outcomes and seed175 slates.

    Set Bernoulli f=target/full oracle headroom: expected finite205-family gain
    equals target. Realized policy draws need not have exactly that gain.
    """
    result = []
    allwins = []
    probabilities = []
    heads = []
    for analysis_seed, bank_seed in zip(s.SEEDS, s.BANK_SEEDS, strict=True):
        base = []
        head = []
        keys = []
        for row in rows:
            cid = row["context_sha256"]
            ix = s.slate_indices(cid, len(costs[cid]), slate, bank_seed)
            u = utilities[cid][ix]
            c = costs[cid][ix]
            b = float(u[np.argmin(c)])
            base.append(b)
            head.append(float(u.max()) - b)
            keys.append(cid if variant == "context" else row["family_id"])
        head = np.asarray(head)
        global_head = float(s.family_mean(head, rows).mean())
        f = target / global_head if global_head > 0 else np.inf
        if not 0 <= f <= 1:
            raise ValueError("requested target not attainable in this bank")
        ctx = np.stack(
            [
                (
                    np.random.default_rng(
                        s.stream(analysis_seed, "target_power", variant, key)
                    ).random(200)
                    < f
                )
                * h
                for key, h in zip(keys, head, strict=True)
            ],
            axis=1,
        )
        result.append(s.family_mean(ctx, rows))
        allwins.append(
            s.family_mean(
                np.where(ctx > 1e-4, 1.0, np.where(ctx < -1e-4, 0.0, 0.5)), rows
            )
        )
        probabilities.append(f)
        heads.append(global_head)
    return np.stack(result), np.stack(allwins), probabilities, heads


def mde(curve):
    result = None
    for row in curve:
        if row["power"]["median"] >= 0.8:
            result = dict(
                status="GRID_DETECTED",
                fraction=row["fraction"],
                gain=row["injected_true_gain"]["median"],
            )
            break
    if result is None:
        result = dict(
            status="RIGHT_CENSORED",
            fraction=None,
            gain=None,
            gain_lower_tested_bound=curve[-1]["injected_true_gain"]["median"],
        )
    crossing = np.full(len(curve[0]["power_boot"]), np.nan)
    for row in curve:
        hit = np.isnan(crossing) & (row["power_boot"] >= 0.8)
        crossing[hit] = row["gain_boot"][hit]
    finite = np.isfinite(crossing)
    rate = float((~finite).mean())
    # Right-censored crossings are +infinity, never silently dropped.
    ordered = np.sort(np.where(finite, crossing, np.inf))
    bounds = []
    for q in [0.025, 0.975]:
        x = ordered[max(0, int(np.ceil(q * len(ordered))) - 1)]
        bounds.append(float(x) if np.isfinite(x) else None)
    result.update(
        gain_ci95=bounds,
        bootstrap_right_censored_fraction=rate,
        CI_definition="10k shared family resamples, median seed axis0, recompute first power>=.8 grid crossing; null upper endpoint denotes right-censoring beyond largest tested grid",
    )
    return result


def run(rows, costs, utilities, bank, out, check):
    group, weights, foldweights, design = prepare(rows, costs, utilities)
    results = []
    curves = {}
    gate = {
        m: dict(
            max_calibration_UCB=0.0,
            max_abs_bias=0.0,
            worst_calibration_cell=None,
            worst_bias_cell=None,
        )
        for m in s.METHODS
    }
    np.savez_compressed(
        out / "covariates.npz",
        **{slate: values[0] for slate, values in design.items()},
        families=np.array(sorted({r["family_id"] for r in rows})),
    )
    for slate in ["24", "full"]:
        for variant in ["context", "family"]:
            for direction in ["oracle", "random_nonoracle"]:
                for fraction in s.FRACTIONS:
                    label = f"{slate}-{variant}-{direction}-{fraction:g}"
                    path = bank / (label + ".npz")
                    with np.load(path, allow_pickle=False) as z:
                        D = z["delta"].copy()
                        wins = z["wins"].copy()
                    D.setflags(write=False)
                    summary, raw = case(
                        D, wins, slate, design, group, weights, foldweights
                    )
                    summary.update(
                        label=label,
                        slate=slate,
                        variant=variant,
                        direction=direction,
                        fraction=fraction,
                        source="H175 frozen200-draw bank",
                        bank_seeds=list(s.BANK_SEEDS),
                        analysis_bootstrap_seeds=list(s.SEEDS),
                    )
                    np.savez_compressed(out / (label + ".npz"), **raw)
                    results.append(summary)
                    for method, data in summary["methods"].items():
                        state = gate[method]
                        if (fraction == 0 or direction == "random_nonoracle") and data[
                            "CP95_upper_max"
                        ] > state["max_calibration_UCB"]:
                            state.update(
                                max_calibration_UCB=data["CP95_upper_max"],
                                worst_calibration_cell=label,
                            )
                        if (
                            data["bias_vs_STRICT"]["max_abs_per_seed"]
                            > state["max_abs_bias"]
                        ):
                            state.update(
                                max_abs_bias=data["bias_vs_STRICT"]["max_abs_per_seed"],
                                worst_bias_cell=label,
                            )
                        if direction == "oracle":
                            pf = np.empty((3, 205))
                            for si in range(3):
                                for outer in range(5):
                                    b = s.roles(group, outer)[1]
                                    pf[si, b] = raw["accepted"][
                                        s.METHODS.index(method), si, outer
                                    ].mean()
                            curves.setdefault((slate, variant, method), []).append(
                                dict(
                                    fraction=fraction,
                                    power=data["power"],
                                    injected_true_gain=summary["injected_true_gain"],
                                    power_boot=s.seed_bootstrap(pf, weights),
                                    gain_boot=s.seed_bootstrap(D.mean(axis=1), weights),
                                )
                            )
                    check()
                    print("BANK CASE COMPLETE", label, flush=True)
    target_results = []
    for slate in ["24", "full"]:
        for variant in ["context", "family"]:
            for target in s.TARGETS:
                label = f"target-{slate}-{variant}-{target:g}"
                D, wins, probability, head = target_bank(
                    rows, costs, utilities, slate, variant, target
                )
                summary, raw = case(D, wins, slate, design, group, weights, foldweights)
                summary.update(
                    label=label,
                    slate=slate,
                    variant=variant,
                    expected_true_gain=target,
                    bernoulli_fraction_by_seed=probability,
                    oracle_headroom_by_seed=head,
                    simulation_seeds=list(s.SEEDS),
                    bank_slate_seeds=list(s.BANK_SEEDS),
                    source="new independent181 oracle activation streams on the same H175 cached real cost/outcome bank",
                )
                np.savez_compressed(out / (label + ".npz"), **raw)
                target_results.append(summary)
                for method, data in summary["methods"].items():
                    if (
                        data["bias_vs_STRICT"]["max_abs_per_seed"]
                        > gate[method]["max_abs_bias"]
                    ):
                        gate[method].update(
                            max_abs_bias=data["bias_vs_STRICT"]["max_abs_per_seed"],
                            worst_bias_cell=label,
                        )
                check()
                print("TARGET CASE COMPLETE", label, flush=True)
    mdes = {
        slate: {
            variant: {
                method: mde(curves[slate, variant, method]) for method in s.METHODS
            }
            for variant in ["context", "family"]
        }
        for slate in ["24", "full"]
    }
    for method in s.METHODS:
        reductions = []
        for slate in ["24", "full"]:
            baseline = mdes[slate]["family"]["STRICT"]
            candidate = mdes[slate]["family"][method]
            if (
                baseline["status"] == "GRID_DETECTED"
                and candidate["status"] == "GRID_DETECTED"
            ):
                reductions.append(1 - candidate["gain"] / baseline["gain"])
            elif (
                baseline["status"] == "RIGHT_CENSORED"
                and candidate["status"] == "GRID_DETECTED"
            ):
                reductions.append(
                    1 - candidate["gain"] / baseline["gain_lower_tested_bound"]
                )
            else:
                reductions.append(None)
        gate[method].update(
            family_MDE_reduction_24_full=reductions,
            calibration_pass=gate[method]["max_calibration_UCB"] <= 0.05,
            bias_pass=gate[method]["max_abs_bias"] < 0.001 - s.EPS,
        )
        gate[method]["qualifies"] = bool(
            method in ["CUPED1", "CUPED_K"]
            and s.decision(
                reductions,
                gate[method]["max_calibration_UCB"],
                gate[method]["max_abs_bias"],
                method,
            )
        )
    return dict(
        wave="H181",
        status="COMPLETE",
        verdict="ADOPT_CANDIDATE_ESTIMATOR"
        if any(gate[m]["qualifies"] for m in ["CUPED1", "CUPED_K"])
        else "NO_GAIN",
        primary="H175-style41-family B mean estimator of finite205-family paired gain; FIT123 excludesB andA; one candidate,m1,10k paired bootstrap. Not exact H177 pooled-m4 requalification.",
        bank_seeds=s.BANK_SEEDS,
        analysis_seeds=s.SEEDS,
        fold_seed=s.FOLD_SEED,
        boot_seed=s.BOOT_SEED,
        bank_cases=results,
        target_power=target_results,
        MDE80=mdes,
        gate=gate,
        optimizer_updates=0,
        backward_calls=0,
        methods=s.METHODS,
        context_count=1219,
        family_count=205,
        selection_bootstrap_draws=10000,
        CP_confidence=0.95,
        CP_scope="one-sided exact CP on200 independent policy draws per seed/fold; overlapping folds/cases/seeds never pooled into a larger n",
        fixed_moment_scope="baseline X mean/stratum masses are known for the fixed205-family bank; this is not unknown new-family population-moment uncertainty",
        CI_scope="10k shared-family resamples across3 source MCseed panels, seed reductionaxis0; conditional on fitted nuisance coefficients, fixedfolds and simulated policies. Fold-block sensitivity included.",
        serving_promotion=False,
    )
