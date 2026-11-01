"""Role-isolated counterfactual simulations and finite-cohort power summaries."""

import json
from pathlib import Path
import numpy as np
import protocol as p


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def inner_B_records(obj):
    """Explicit saved selector schemas only; never descend into outcome summaries."""
    records = []

    def add(path, value, kind="cell_best"):
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(value)
        ):
            records.append(
                dict(json_path=path, gain=float(value), cohort="inner-B", kind=kind)
            )

    def entries(value):
        return (
            value.items()
            if isinstance(value, dict)
            else enumerate(value)
            if isinstance(value, list)
            else []
        )

    # H162 publication: full cell-level B best gains, separate from posthoc OOF/Val.
    for i, cell in entries(obj.get("cell_selection_summary", [])):
        if isinstance(cell, dict):
            for arm, gain in entries(cell.get("unconstrained_selected_gains", {})):
                add(
                    f"/cell_selection_summary/{i}/unconstrained_selected_gains/{arm}",
                    gain,
                )
    if records:
        return records
    # H166 publication and H168 raw publication preserve cell decision dictionaries.
    for container in ["cells", "cell_selections"]:
        for i, cell in entries(obj.get(container, [])):
            if not isinstance(cell, dict):
                continue
            for arm, decision in entries(cell.get("decisions", {})):
                if not isinstance(decision, dict):
                    continue
                prefix = f"/{container}/{i}/decisions/{arm}"
                if "unconstrained_selected_gain" in decision:
                    add(
                        prefix + "/unconstrained_selected_gain",
                        decision["unconstrained_selected_gain"],
                    )
                elif (
                    isinstance(decision.get("family_means"), list)
                    and decision["family_means"]
                ):
                    means = decision["family_means"]
                    add(
                        prefix + "/family_means:max-minus-baseline",
                        max(means) - means[0],
                    )
        if records:
            return records
    # H165/H167/H169/H170 raw best-only selector records. No selected-OOF contrast.
    for i, cell in entries(obj.get("selections", [])):
        if not isinstance(cell, dict):
            continue
        for arm, decision in cell.items():
            if not isinstance(decision, dict):
                continue
            prefix = f"/selections/{i}/{arm}"
            if "unconstrained_selected_gain" in decision:
                add(
                    prefix + "/unconstrained_selected_gain",
                    decision["unconstrained_selected_gain"],
                )
            elif "best_step" in decision and isinstance(
                decision.get("gain"), (float, int)
            ):
                add(prefix + "/gain", decision["gain"])
            elif (
                isinstance(decision.get("family_means"), list)
                and decision["family_means"]
            ):
                means = decision["family_means"]
                add(prefix + "/family_means:max-minus-baseline", max(means) - means[0])
    if records:
        return records
    # H163 fixed ensemble comparison on B, no fitted checkpoint step.
    for i, cell in entries(obj.get("decisions", [])):
        if not isinstance(cell, dict):
            continue
        for arm, decision in entries(cell.get("arms", {})):
            if isinstance(decision, dict) and isinstance(
                decision.get("selector"), dict
            ):
                add(
                    f"/decisions/{i}/arms/{arm}/selector/delta",
                    decision["selector"].get("delta"),
                )
    if records:
        return records
    # H161 retains actual arm-wise maxima, not the individual B gains.
    for arm, diagnostic in entries(obj.get("selector_fold_diagnostics", {})):
        if isinstance(diagnostic, dict):
            add(
                f"/selector_fold_diagnostics/{arm}/max_best_step_gain",
                diagnostic.get("max_best_step_gain"),
                kind="arm_max_summary",
            )
    return records


def history(paths, root):
    reports = []
    for row in paths:
        if row.get("path") is None:
            reports.append(dict(wave=row["wave"], status="UNAVAILABLE_AT_PREREG"))
            continue
        obj = json.loads(
            (Path(row["path"]) if row["absolute"] else root / row["path"]).read_text()
        )
        records = inner_B_records(obj)
        gains = [r["gain"] for r in records]
        source_status = str(obj.get("status", obj.get("run_status", "NOT_RECORDED")))
        incomplete = any(
            tag in source_status.upper()
            for tag in [
                "INCOMPLETE",
                "NOT_RUN",
                "PREREGISTERED",
                "RUNNING",
                "IN_PROGRESS",
            ]
        )
        leak = row["wave"] == "H169" or obj.get("calibration_leak") is True
        reports.append(
            dict(
                wave=row["wave"],
                path=row["path"],
                sha256=row["sha256"],
                source_status=source_status,
                explicit_incomplete=incomplete,
                calibration_leak=leak,
                status="EXTRACTED" if gains else "NO_REPLAYABLE_INNER_B_GAINS",
                verdict_threshold_eligible=bool(gains and not incomplete and not leak),
                count=len(records),
                max_unconstrained_inner_B_gain=max(gains) if gains else None,
                median_extracted_gain=float(np.median(gains)) if gains else None,
                records=records,
                interpretation="Explicit inner-B schemas only; arm-max summaries distinguished from cell records. OOF/Val never substituted. Incomplete/leaky studies diagnostic only; unknown completeness labeled, not fabricated. Historical reuse is not independent replication.",
            )
        )
    return reports


def run(rows, val, payload, out, check):
    groups = p.folds(rows)
    families = sorted({r["family_id"] for r in rows})
    fi = {f: i for i, f in enumerate(families)}
    weights = p.report_weights(205)
    foldweights = p.report_weights(5, p.BOOT_SEED + 1)
    head, results, curves = {}, [], {}
    for slate in ["24", "full"]:
        # Descriptive TRAIN headroom is separate from role-local selector simulations.
        train_panels = []
        for seed in p.SEEDS:
            parts = [
                p.inject(
                    rows,
                    payload,
                    {r["context_sha256"] for r in rows},
                    seed,
                    1.0,
                    "context",
                    "oracle",
                    slate,
                )
            ]
            train_panels.append(parts[0])
        val_panels = [
            p.inject(
                val,
                payload,
                {r["context_sha256"] for r in val},
                seed,
                1.0,
                "context",
                "oracle",
                slate,
            )
            for seed in p.SEEDS
        ]
        head[slate] = dict(
            TRAIN=p.headroom(rows, train_panels),
            DLVal=p.headroom(val, val_panels),
            note="Full-pool 3 seed baseline/oracle utilities are identical; 24 slates vary only by prespecified sampling. No independent model replications.",
        )
        head_family = np.stack(
            [
                p.family_mean(x["oracle"] - x["base"], [r["family_id"] for r in rows])[
                    1
                ]
                for x in train_panels
            ]
        )
        for variant in ["context", "family"]:
            for direction in ["oracle", "random_nonoracle"]:
                cases = []
                raw_mde = []
                for fraction in p.FRACTIONS:
                    delta = np.zeros((3, p.SIMULATIONS, 205))
                    wins = np.full_like(delta, 0.5)
                    accepted = np.zeros((2, 3, 5, p.SIMULATIONS), bool)
                    bgain = np.zeros((3, 5, p.SIMULATIONS))
                    bci = np.zeros((3, 5, p.SIMULATIONS, 2))
                    activated = np.zeros((3, p.SIMULATIONS))
                    role_checks = []
                    for si, seed in enumerate(p.SEEDS):
                        for outer in range(5):
                            bgroup, agroup = (
                                set(groups[(outer + 1) % 5]),
                                set(groups[outer]),
                            )
                            b = [r for r in rows if r["family_id"] in bgroup]
                            a = [r for r in rows if r["family_id"] in agroup]
                            fit = [
                                r for r in rows if r["family_id"] not in bgroup | agroup
                            ]
                            assert len(
                                {r["family_id"] for r in fit}
                            ) == 123 and bgroup.isdisjoint(agroup)
                            # Create the B view from B IDs only; no reference to A outcomes.
                            bscope = {r["context_sha256"] for r in b}
                            bview = {c: payload[c] for c in bscope}
                            bp = p.inject(
                                b,
                                bview,
                                bscope,
                                seed,
                                fraction,
                                variant,
                                direction,
                                slate,
                            )
                            bf, bfm = p.family_mean(
                                bp["delta"], [r["family_id"] for r in b]
                            )
                            bc = np.random.default_rng(
                                p.stream(p.BOOT_SEED, seed, outer, "B")
                            ).integers(0, 41, (p.SELECTOR_DRAWS, 41))
                            g, ci, acc = p.select(bfm, bc)
                            bgain[si, outer] = g
                            bci[si, outer] = ci
                            accepted[:, si, outer] = acc
                            # A payload injection starts only after B decision bytes are durable.
                            lock = (
                                out
                                / f"lock-{slate}-{variant}-{direction}-{fraction:g}-{seed}-{outer}.npz"
                            )
                            with lock.open("xb") as file:
                                np.savez_compressed(
                                    file,
                                    B_ids=np.array(sorted(bscope)),
                                    gain=g,
                                    ci95=ci,
                                    accepted=acc,
                                )
                                file.flush()
                                import os

                                os.fsync(file.fileno())
                            ascope = {r["context_sha256"] for r in a}
                            aview = {c: payload[c] for c in ascope}
                            ap = p.inject(
                                a,
                                aview,
                                ascope,
                                seed,
                                fraction,
                                variant,
                                direction,
                                slate,
                            )
                            af, afm = p.family_mean(
                                ap["delta"], [r["family_id"] for r in a]
                            )
                            aw = p.family_mean(ap["wins"], [r["family_id"] for r in a])[
                                1
                            ]
                            delta[si][:, [fi[f] for f in af]] = afm
                            wins[si][:, [fi[f] for f in af]] = aw
                            activated[si] += ap["activated"].sum(axis=1) / len(rows)
                            role_checks.append(
                                dict(
                                    seed=seed,
                                    outer=outer,
                                    B_contexts=len(b),
                                    A_contexts=len(a),
                                    B_families=len(bf),
                                    A_families=len(af),
                                    fit_families=123,
                                    overlap=0,
                                )
                            )
                            check()
                    # The same counterfactual policy is reproducible independently in B and A.
                    for si in range(3):
                        for outer in range(5):
                            ix = [fi[f] for f in groups[(outer + 1) % 5]]
                            assert np.array_equal(
                                np.ascontiguousarray(delta[si][:, ix]).mean(axis=1),
                                bgain[si, outer],
                            )
                    name = f"{slate}-{variant}-{direction}-{fraction:g}"
                    np.savez_compressed(
                        out / (name + ".npz"),
                        delta=delta,
                        wins=wins,
                        accepted=accepted,
                        B_gain=bgain,
                        B_ci=bci,
                        activated_fraction=activated,
                    )
                    item = dict(
                        slate=slate,
                        variant=variant,
                        direction=direction,
                        fraction=fraction,
                        simulations_per_seed=200,
                        selector_decisions_per_seed=1000,
                        role_checks=len(role_checks),
                        injected_gain=p.summarize(delta.mean(axis=1), weights),
                        injected_elo=p.elo(wins.mean(axis=1), weights),
                        activated_context_fraction_by_seed=activated.mean(
                            axis=1
                        ).tolist(),
                        rules={},
                    )
                    truth = delta.mean(
                        axis=2
                    )  # per seed/draw equal-family finite TRAIN truth
                    for ri, rule in enumerate(["STRICT", "H168_NEW"]):
                        deployed = np.zeros_like(delta)
                        dwins = np.full_like(wins, 0.5)
                        pfam = np.zeros((3, 205))
                        fpfam = np.zeros((3, 205))
                        nonpos = []
                        heldbad = 0
                        heldtotal = 0
                        for si in range(3):
                            bad = truth[si] <= 0
                            nonpos.append(int(bad.sum()))
                            for outer in range(5):
                                ix = [fi[f] for f in groups[outer]]
                                ac = accepted[ri, si, outer]
                                deployed[si][:, ix] = delta[si][:, ix] * ac[:, None]
                                dwins[si][:, ix] = np.where(
                                    ac[:, None], wins[si][:, ix], 0.5
                                )
                                pfam[si, ix] = ac.mean()
                                fpfam[si, ix] = (ac & bad).sum() / max(1, bad.sum())
                                heldbad += int(
                                    (ac & (delta[si][:, ix].mean(axis=1) <= 0)).sum()
                                )
                                heldtotal += int(ac.sum())
                        power = p.summarize(pfam, weights)
                        foldpower = accepted[ri].mean(axis=2)
                        power["fold_cluster_ci95"] = np.quantile(
                            np.median(foldpower @ foldweights.T, axis=0), [0.025, 0.975]
                        ).tolist()
                        power["CI_scope"] = (
                            "10k family CI conditions on fixed folds/decisions; within-fold correlations retained in additional 5-fold block CI; no new-fold generalization claim."
                        )
                        fp = p.summarize(fpfam, weights) if all(nonpos) else None
                        item["rules"][rule] = dict(
                            power=power,
                            selected_OOF_gain=p.summarize(
                                deployed.mean(axis=1), weights
                            ),
                            selected_elo=p.elo(dwins.mean(axis=1), weights),
                            false_admission_given_nonpositive_population_gain=fp,
                            nonpositive_simulations_by_seed=nonpos,
                            admitted_nonpositive_A_fraction=heldbad / heldtotal
                            if heldtotal
                            else None,
                            admitted_decisions=heldtotal,
                        )
                        if direction == "oracle":
                            curves.setdefault((slate, variant, rule), []).append(
                                dict(
                                    fraction=fraction,
                                    injected_gain=item["injected_gain"],
                                    power=power,
                                )
                            )
                    if direction == "oracle":
                        raw_mde.append((delta.mean(axis=1), accepted.mean(axis=3)))
                    results.append(item)
                    cases.append(item)
                    print("SIMULATION COMPLETE", name, flush=True)
                if direction == "oracle":
                    for ri, rule in enumerate(["STRICT", "H168_NEW"]):
                        boot_g = []
                        boot_p = []
                        for famgain, acc in raw_mde:
                            boot_g.append(np.median(famgain @ weights.T, axis=0))
                            pf = np.empty((3, 205))
                            for outer in range(5):
                                pf[:, [fi[f] for f in groups[outer]]] = acc[
                                    ri, :, outer, None
                                ]
                            boot_p.append(np.median(pf @ weights.T, axis=0))
                        boot_g = np.array(boot_g)
                        boot_p = np.array(boot_p)
                        reached = boot_p >= 0.8
                        hit = reached.any(axis=0)
                        indices = reached.argmax(axis=0)
                        sampled = np.where(
                            hit, boot_g[indices, np.arange(p.REPORT_DRAWS)], np.inf
                        )
                        denom = np.median(head_family @ weights.T, axis=0)
                        percent = np.divide(
                            100 * sampled,
                            denom,
                            out=np.full_like(sampled, np.inf),
                            where=denom > 0,
                        )

                        def interval(x):
                            ordered = np.sort(x)
                            out = []
                            for q in [0.025, 0.975]:
                                v = ordered[int(round(q * (len(ordered) - 1)))]
                                out.append(float(v) if np.isfinite(v) else None)
                            return out

                        md = p.mde(
                            curves[(slate, variant, rule)],
                            head[slate]["TRAIN"]["equal_family_headroom"]["median"],
                        )
                        md.update(
                            gain_ci95=interval(sampled),
                            headroom_percent_ci95=interval(percent),
                            right_censored_bootstrap_fraction=float((~hit).mean()),
                            CI_scope="Conditional family bootstrap of grid crossing; null upper endpoint means right censored, not zero.",
                        )
                        head[slate].setdefault("MDE80", {}).setdefault(variant, {})[
                            rule
                        ] = md
    save(
        out / "simulation_metrics.json",
        dict(
            headroom=head,
            curves=results,
            simulations_are_counterfactual_policies_not_synthetic_datasets=True,
            optimizer_updates=0,
        ),
    )
    return head, results
