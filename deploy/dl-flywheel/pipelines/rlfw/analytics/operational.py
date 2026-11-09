"""Expert-review 11.8–11.10 observations; no learner or evaluator execution."""

import json
import math
import statistics
from collections import defaultdict

from .scratch import number

FEEDBACK_STAGES = (
    "proposed",
    "compute_started",
    "locally_verified",
    "remotely_qualified",
    "adopted",
    "first_trained",
    "first_evaluated",
)
UNMEASURED = {
    "qualified_context_program_outcomes_per_hour": "outcomes/hour",
    "qualified_non_catalog_programs_per_hour": "programs/hour",
    "feedback_to_first_update_seconds": "seconds",
    "contexts_receiving_actor_targets": "contexts",
    "nonzero_hwm_advantage_support_fraction": "fraction",
    "actor_grad_norm:control": "L2 norm",
    "actor_grad_norm:L_PI-archive": "L2 norm",
    "actor_grad_norm:RLOO": "L2 norm",
    "duplicate_draw_rate": "fraction of draw positions",
    "expected_distinct_D_B": "programs",
    "archive_exposure_by_parent": "exposures",
    "actor_probability_movement_on_verified_improvements": "probability",
    "entropy": "nats",
    "heldout_gain_per_declared_resource_budget": "gain/budget",
    "parent_l6_paired_mean_delta": "loss",
}


def row(
    metric,
    value,
    *,
    date,
    commit,
    source,
    unit,
    scope="flywheel",
    n=None,
    note="",
    details=None,
    kind="operational",
    context_set="unmeasured",
):
    return {
        "date": date,
        "kind": kind,
        "metric": metric,
        "scope": scope,
        "status": "NON-QUALIFYING" if value is not None else "unmeasured",
        "value": number(value),
        "n": n,
        "unit": unit,
        "note": note,
        "sources": source,
        "source_commit": commit,
        "context_set_sha256": context_set,
        "details": json.dumps(details or {}, sort_keys=True, allow_nan=False),
    }


def unmeasured_rows(*, date, commit, source):
    """Availability rows do not turn absent qualified/exposure records into zeros."""
    metrics = UNMEASURED | {
        "feedback_timestamp:" + stage: "UTC seconds" for stage in FEEDBACK_STAGES
    }
    return [
        row(
            metric,
            None,
            date=date,
            commit=commit,
            source=source,
            unit=unit,
            note="No complete, identity-bound provider observation. Scratch evaluation and training counts cannot fill this production metric.",
        )
        for metric, unit in metrics.items()
    ]


def throughput_rows(export, receipt, *, date, commit, source):
    """Check G-S3GP's two denominators against the copied finished receipt."""
    if (
        export.get("schema") != "jope.g-s3gp-throughput-denominator/v1"
        or export.get("label") != "NON-QUALIFYING"
        or receipt.get("label") != "NON-QUALIFYING"
        or receipt.get("state") != "finished"
        or receipt.get("exit_code") != 0
        or export.get("sealed_outer_test_scored") is not False
        or receipt.get("sealed_outer_test_scored") is not False
        or export.get("source_commit") != receipt.get("source_commit")
    ):
        raise ValueError("finished NON-QUALIFYING provider throughput receipt required")
    count = receipt["programs_scored"]
    if type(count) is not int or count < 0 or export["programs_scored"] != count:
        raise ValueError("program count differs from physical receipt")
    active, allocated = export["active_cpu"], export["allocated"]
    cpu, wall = number(receipt["cpu_seconds"]), number(receipt["wall_seconds"])
    cores = number(allocated["reserved_cores"])
    if (
        any(v is None for v in (cpu, wall, cores))
        or min(cpu, wall, cores) <= 0
        or not active.get("source")
    ):
        raise ValueError("positive measured time and declared allocation required")
    expected = [
        (active["cpu_seconds"], cpu),
        (active["active_core_hours"], cpu / 3600),
        (active["programs_per_active_core_hour"], count * 3600 / cpu),
        (allocated["cpu_quota_percent"], cores * 100),
        (allocated["wall_seconds"], wall),
        (allocated["allocated_core_hours"], cores * wall / 3600),
        (allocated["programs_per_allocated_core_hour"], count * 3600 / (cores * wall)),
        (receipt["programs_per_core_hour"], count * 3600 / cpu),
    ]
    if any(
        not math.isclose(number(a), b, rel_tol=1e-9, abs_tol=1e-9) for a, b in expected
    ):
        raise ValueError("provider throughput denominator arithmetic mismatch")
    values = {
        "programs_scored": (count, "program evaluations"),
        "active_cpu_core_hours": (cpu / 3600, "active CPU core-hours"),
        "allocated_core_hours": (cores * wall / 3600, "allocated core-hours"),
        "wall_seconds": (wall, "seconds"),
        "programs_per_active_cpu_hour": (
            count * 3600 / cpu,
            "programs/active CPU-hour",
        ),
        "programs_per_allocated_core_hour": (
            count * 3600 / (cores * wall),
            "programs/allocated core-hour",
        ),
        "programs_per_wall_hour": (count * 3600 / wall, "programs/wall-hour"),
        "compiles": (receipt.get("cache_compiles"), "compiles"),
        "cached_disclosures": (receipt.get("cache_hits"), "cached disclosures"),
    }
    return [
        row(
            metric,
            value,
            date=date,
            commit=commit,
            source=source,
            unit=unit,
            scope=f"G-S3GP/round{export['round']}",
            n=count,
            note="Parity-unqualified scratch. Active CPU is producer RUSAGE_SELF+CHILDREN; allocation is provider-declared constant cores over measured wall. Neither rate is qualified outcome throughput. Compiles and cached disclosures are separate counters.",
            details={
                "provider_commit": receipt["source_commit"],
                "reserved_cores": cores,
                "active_cpu_source": active["source"],
            },
        )
        for metric, (value, unit) in values.items()
    ]


def draw_statistics(draws, probabilities=None, *, B=None, complete_policy=False):
    """Keep duplicate draw positions; D_B needs the complete actual finite policy."""
    duplicate = None
    if draws is not None:
        if not isinstance(draws, list):
            raise ValueError("draw positions must be an explicit array")
        if any(not isinstance(x, str) or not x for x in draws):
            raise ValueError("explicit program identity per draw position required")
        if draws:
            duplicate = 1 - len(set(draws)) / len(draws)
    result = {
        "duplicate_draw_rate": duplicate,
        "expected_distinct_D_B": None,
        "entropy": None,
    }
    if not complete_policy:
        return result
    if type(B) is not int or B < 1 or not probabilities:
        raise ValueError("complete finite policy and positive integer B required")
    ps = [number(p, nonnegative=True) for p in probabilities]
    if any(p is None or p > 1 for p in ps) or not math.isclose(
        sum(ps), 1, abs_tol=1e-9
    ):
        raise ValueError("probabilities must sum to one over complete support")
    result["expected_distinct_D_B"] = sum(
        1 if p == 1 else -math.expm1(B * math.log1p(-p)) for p in ps
    )
    result["entropy"] = -sum(p * math.log(p) for p in ps if p)
    return result


def feedback_ages(events, *, observed_until):
    """Explicit seven-stage clocks; adoption/exposure is not an optimizer update."""
    if number(observed_until) is None:
        raise ValueError("finite observation clock required")
    grouped, identities = defaultdict(dict), {}
    for event in events:
        stage, at, key = event["stage"], number(event["at_unix"]), event["join_key"]
        if stage not in FEEDBACK_STAGES or not key or at is None:
            raise ValueError("known feedback stage and exact join key required")
        identity = (event["context_id"], event["parent_id"])
        if not isinstance(key, str) or any(
            not isinstance(x, str) or not x for x in identity
        ):
            raise ValueError(
                "explicit feedback join/context/parent identities required"
            )
        if key in identities and identities[key] != identity:
            raise ValueError("feedback context/parent identity mismatch")
        identities[key] = identity
        if at <= observed_until:
            grouped[key][stage] = min(grouped[key].get(stage, at), at)
    result = []
    for key, clocks in sorted(grouped.items()):
        ordered = [clocks[s] for s in FEEDBACK_STAGES if s in clocks]
        if ordered != sorted(ordered):
            raise ValueError("feedback stage clocks out of order")
        proposed, trained = clocks.get("proposed"), clocks.get("first_trained")
        result.append(
            {
                "join_key": key,
                "context_id": identities[key][0],
                "parent_id": identities[key][1],
                "timestamps": {s: clocks.get(s) for s in FEEDBACK_STAGES},
                "feedback_to_first_update_seconds": trained - proposed
                if trained is not None and proposed is not None
                else None,
                "right_censored": trained is None,
                "age_seconds": observed_until - proposed
                if proposed is not None
                else None,
            }
        )
    return result


def actor_rows(observations, *, date, commit, source, scope):
    """One receipt per actual update/objective; never infer targets from epochs."""
    rows, seen = [], set()
    for observation in observations:
        objective, update = observation["objective"], observation["update"]
        if not objective or type(update) is not int or update < 1:
            raise ValueError("objective and actual positive optimizer update required")
        if (objective, update) in seen:
            raise ValueError("duplicate objective/update observation")
        seen.add((objective, update))
        targets = observation.get("actor_target_context_ids")
        if targets is not None and not isinstance(targets, list):
            raise ValueError("actor target context IDs must be an explicit array")
        if targets is not None and any(
            not isinstance(c, str) or not c for c in targets
        ):
            raise ValueError("explicit actor target context identities required")
        support = observation.get("advantage_support", {})
        fraction = None
        if (
            support.get("hwm_identity_verified") is True
            and support.get("complete") is True
        ):
            total, nonzero = support["total"], support["nonzero"]
            if (
                type(total) is not int
                or type(nonzero) is not int
                or not 0 <= nonzero <= total
            ):
                raise ValueError("complete HWM advantage support counters required")
            fraction = nonzero / total if total else None
        values = draw_statistics(
            observation.get("draws"),
            observation.get("probabilities"),
            B=observation.get("B"),
            complete_policy=observation.get("complete_policy") is True,
        )
        values.update(
            {
                "actor_grad_norm:" + objective: number(
                    observation.get("actor_grad_norm"), nonnegative=True
                ),
                "contexts_receiving_actor_targets": len(set(targets))
                if targets is not None
                else None,
                "nonzero_hwm_advantage_support_fraction": fraction,
            }
        )
        details = {
            "objective": objective,
            "update": update,
            "B": observation.get("B"),
            "target_context_coverage": "reported update only; not whole-day coverage",
        }
        for metric, value in values.items():
            rows.append(
                row(
                    metric,
                    value,
                    date=date,
                    commit=commit,
                    source=source,
                    scope=scope + "/" + objective + "/update/" + str(update),
                    n=1,
                    unit="L2 norm"
                    if metric.startswith("actor_grad_norm:")
                    else UNMEASURED[metric],
                    note="Explicit NON-QUALIFYING update diagnostics. HWM fraction needs a verified comparison identity; D_B needs complete actual policy mass. Epoch, archive inventory and adoption do not establish actor exposure.",
                    details=details,
                )
            )
        exposures = observation.get("archive_exposure_by_parent")
        if exposures is not None:
            counter_kind = observation.get("exposure_counter_kind")
            if counter_kind not in ("step", "cumulative"):
                raise ValueError("archive exposure counter kind required")
            for parent, count in sorted(exposures.items()):
                if not parent or type(count) is not int or count < 0:
                    raise ValueError(
                        "explicit parent and nonnegative exposure count required"
                    )
                rows.append(
                    row(
                        "archive_exposure_by_parent",
                        count,
                        date=date,
                        commit=commit,
                        source=source,
                        scope=scope + "/parent/" + parent,
                        n=1,
                        unit="exposures",
                        note="Reported "
                        + counter_kind
                        + " counter; snapshots are not summed.",
                        details=details
                        | {"parent_id": parent, "counter_kind": counter_kind},
                    )
                )
    return rows


def native_actor_audit_rows(audit, *, date, commit, source):
    """Read R10's probes without relabelling them as actual update exposures."""
    if audit.get("state") != "PASS_INITIAL_NON_QUALIFYING" or not audit.get(
        "diagnostic_source_sha256"
    ):
        raise ValueError("native NON-QUALIFYING initial actor audit required")
    rows = []
    for event in audit["actor_diagnostics"]:
        if (
            event.get("kind") != "actor_diagnostics"
            or type(event.get("updates")) is not int
            or event["updates"] < 0
            or type(event.get("epoch")) is not int
            or event["epoch"] < 0
        ):
            raise ValueError("explicit actor probe update/epoch required")
        n = event["fixture_contexts"]
        if type(n) is not int or n < 1:
            raise ValueError("reported diagnostic fixture contexts required")
        scope = f"CX-OP/R10-archive/{event['arm']}/probe/epoch{event['epoch']}/updates{event['updates']}"
        details = {
            k: event[k]
            for k in (
                "arm",
                "epoch",
                "updates",
                "fixture_sha256",
                "parameter_sha256",
                "policy",
            )
        }
        details["diagnostic_source_sha256"] = audit["diagnostic_source_sha256"]
        for term, norms in event["gradient_norms_by_term"].items():
            for scale in ("raw", "applied"):
                rows.append(
                    row(
                        "actor_probe_grad_norm:" + term + ":" + scale,
                        number(norms[scale]["actor"], nonnegative=True),
                        date=date,
                        commit=commit,
                        source=source,
                        scope=scope,
                        n=n,
                        unit="L2 norm",
                        details=details | {"coefficient": norms["coefficient"]},
                        note=f"Producer gradient probe on {n} fixed contexts; applied means coefficient-scaled term gradient. It is not the gradient consumed by an optimizer update or whole-day target exposure.",
                    )
                )
        for panel in event["probability_movement"]:
            B = panel["B"]
            if type(B) is not int or B < 1:
                raise ValueError("explicit positive draw budget required")
            for policy in ("full_trie", "pi_M"):
                value = number(panel["D_B_" + policy], nonnegative=True)
                maximum = (
                    min(B, panel["fully_measured_actions"]) if policy == "pi_M" else B
                )
                if value is None or value < 1 - 1e-9 or value > maximum + 1e-9:
                    raise ValueError(
                        "reported expected distinct count outside draw budget"
                    )
                rows.append(
                    row(
                        "expected_distinct_D_B:" + policy,
                        value,
                        date=date,
                        commit=commit,
                        source=source,
                        scope=scope + "/" + panel["context_id"],
                        n=1,
                        unit="programs",
                        details=details
                        | {
                            "B": B,
                            "context_id": panel["context_id"],
                            "probabilities_recomputed_by_CX_MET": False,
                        },
                        note=f"Provider-reported finite-policy expectation at B={B} on its fixed probe context. Full trie and measured-leaf conditional pi_M are different policies. This is not observed distinct draws or a production average.",
                    )
                )
    support = audit["hwm_effective_support"]
    total, comparable, unknown, positive = [
        support[k]
        for k in (
            "train_contexts",
            "comparable_hwm_contexts",
            "null_hwm_contexts",
            "fixed_clipped_utility_positive_contexts",
        )
    ]
    if (
        any(type(x) is not int for x in (total, comparable, unknown, positive))
        or min(total, comparable, unknown, positive) < 0
        or total <= 0
        or comparable + unknown != total
        or not 0 <= positive <= comparable
        or not math.isclose(
            number(support["effective_fraction_all_train"]),
            positive / total,
            rel_tol=1e-9,
        )
    ):
        raise ValueError("native scratch HWM support counters do not reconcile")
    rows.append(
        row(
            "nonzero_hwm_advantage_support_fraction_observed_lower_bound",
            positive / total,
            date=date,
            commit=commit,
            source=source,
            scope="CX-OP/R10-archive/scratch-HWM",
            n=total,
            unit="fraction",
            details=support,
            note="Known-positive lower bound for producer scratch clipped-utility HWM support, not a qualified reward/evaluator closure. NULL HWM stays unknown; denominator is all training contexts.",
        )
    )
    rows.append(
        row(
            "nonzero_hwm_advantage_support_fraction_comparable",
            positive / comparable if comparable else None,
            date=date,
            commit=commit,
            source=source,
            scope="CX-OP/R10-archive/scratch-HWM",
            n=comparable,
            unit="fraction",
            details=support,
            note="Producer scratch support among explicitly comparable HWM contexts only. Missing incumbents are excluded and reported; no qualified reward/evaluator closure.",
        )
    )
    return rows


def build(manifest_path, output):
    """Copy provider files first, append measured/NULL rows, render Parquet only."""
    import datetime as dt
    import subprocess
    from pathlib import Path

    from .extract import Copies, digest
    from .report import read_daily, render, write_daily

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    copies = Copies(output / "inputs")
    manifest_copy = copies.copy(manifest_path)
    manifest = json.loads(manifest_copy.read_bytes())
    date = manifest["date"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    def copied(ref):
        return copies.copy(
            ref["path"], expected_sha=ref.get("sha256"), expected_bytes=ref.get("bytes")
        )

    def source(ref):
        r = copies.records[str(Path(ref["path"]).absolute())]
        return r["source"] + ":1 sha256=" + r["sha256"]

    base = read_daily([copied(ref) for ref in manifest.get("base_daily", [])])
    rows = unmeasured_rows(
        date=date,
        commit=commit,
        source=str(manifest_path) + ":1 sha256=" + digest(manifest_copy),
    )
    for refs in manifest.get("throughput", []):
        export, receipt = [
            json.loads(copied(refs[k]).read_bytes()) for k in ("export", "receipt")
        ]
        rows.extend(
            throughput_rows(
                export,
                receipt,
                date=date,
                commit=commit,
                source=source(refs["export"]) + " ; " + source(refs["receipt"]),
            )
        )
    ref = manifest.get("parents")
    if ref:
        document = json.loads(copied(ref).read_bytes())
        if document.get("schema") != "jope.cx-met-parent-map/v1":
            raise ValueError("explicit normalized physical parent map required")
        rows.extend(
            parent_cluster_rows(
                base, document["rows"], date=date, commit=commit, source=source(ref)
            )
        )
    ref = manifest.get("feedback")
    if ref:
        document = json.loads(copied(ref).read_bytes())
        if document.get("schema") != "jope.rlfw-feedback-age/v1":
            raise ValueError("explicit seven-stage feedback export required")
        for observation in feedback_ages(
            document["events"], observed_until=document["observed_until"]
        ):
            details = observation
            scope = "feedback/" + observation["join_key"]
            for metric, value, unit in [
                (
                    "feedback_to_first_update_seconds",
                    observation["feedback_to_first_update_seconds"],
                    "seconds",
                )
            ]:
                rows.append(
                    row(
                        metric,
                        value,
                        date=date,
                        commit=commit,
                        source=source(ref),
                        scope=scope,
                        n=1,
                        unit=unit,
                        details=details,
                        note="Exact receipt join; no adoption-to-training alias. Missing first update is right-censored.",
                    )
                )
            for stage, at in observation["timestamps"].items():
                rows.append(
                    row(
                        "feedback_timestamp:" + stage,
                        at,
                        date=date,
                        commit=commit,
                        source=source(ref),
                        scope=scope,
                        n=1,
                        unit="UTC seconds",
                        details=details,
                        note="Explicit producer clock only; absent stage remains NULL.",
                    )
                )
    for ref in manifest.get("actor", []):
        document = json.loads(copied(ref).read_bytes())
        if (
            document.get("schema") != "jope.rlfw-actor-diagnostics/v1"
            or document.get("non_qualifying") is not True
            or document.get("program_execution") is not False
            or document.get("promotion_input") is not False
        ):
            raise ValueError(
                "explicit NON-QUALIFYING actor diagnostics export required"
            )
        rows.extend(
            actor_rows(
                document["observations"],
                date=date,
                commit=commit,
                source=source(ref),
                scope=document["lane"] + "/" + document["run"],
            )
        )
    for ref in manifest.get("native_actor_audits", []):
        document = json.loads(copied(ref).read_bytes())
        rows.extend(
            native_actor_audit_rows(
                document, date=date, commit=commit, source=source(ref)
            )
        )
    files = write_daily(rows, output / "daily")
    actual = read_daily(files)
    render(base + actual, output / "dashboard.html")
    receipt = {
        "schema": "jope.cx-met-operational-ingest/v1",
        "source_commit": commit,
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "non_qualifying": True,
        "program_execution": False,
        "promotion_input": False,
        "rows": len(actual),
        "dashboard_rows": len(base) + len(actual),
        "inputs": list(copies.records.values()),
        "outputs": [
            {"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size}
            for p in [*files, output / "dashboard.html"]
        ],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cgroup = next(
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    scope = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    quota, period = (scope / "cpu.max").read_text().split()
    memory = (scope / "memory.max").read_text().strip()
    if (
        quota == "max"
        or int(quota) / int(period) > 2
        or memory == "max"
        or int(memory) > 4 * 1024**3
    ):
        raise ValueError("operational ingest requires CPU<=200% and memory<=4GiB")
    receipt = build(args.manifest, args.output)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ("source_commit", "observed_at", "rows", "dashboard_rows")
            }
        )
    )


def parent_cluster_rows(pairs, parents, *, date, commit, source):
    """Mean cuts within parent first; epochs/exports remain separate cohorts."""
    mapping = {}
    for item in parents:
        context, parent = item["context_id"], item["parent_id"]
        if not context or not parent or context in mapping:
            raise ValueError("unique explicit context-to-parent map required")
        mapping[context] = parent
    grouped = defaultdict(list)
    for pair in pairs:
        if pair["kind"] != "scratch_context_pair":
            continue
        d = json.loads(pair["details"])
        key = (
            pair["scope"],
            pair["context_set_sha256"],
            d["epoch"],
            d["updates"],
            d["export_sha256"],
            d["control_arm"],
        )
        grouped[key].append((pair, d))
    rows = []
    for key, records in sorted(grouped.items()):
        scope, roster_sha, epoch, updates, export_sha, control = key
        paired_source = (
            source
            + " ; "
            + " ; ".join(sorted({p["sources"] for p, _ in records if "sources" in p}))
        )
        unique = {}
        for pair, details in records:
            context = details["context_id"]
            if context in unique:
                prior, prior_details = unique[context]
                if (
                    pair["metric"] != prior["metric"]
                    or pair["value"] != prior["value"]
                    or details != prior_details
                ):
                    raise ValueError(
                        "conflicting paired context in one physical export/epoch"
                    )
            else:
                unique[context] = (pair, details)
        duplicate_copies = len(records) - len(unique)
        records = list(unique.values())
        by_context = {d["context_id"]: p["value"] for p, d in records}
        expected = {d["fixed_contexts"] for _, d in records}
        complete = (
            len(expected) == 1
            and len(records) == next(iter(expected))
            and all(d["complete_roster"] for _, d in records)
            and all(c in mapping and v is not None for c, v in by_context.items())
        )
        details = {
            "epoch": epoch,
            "updates": updates,
            "export_sha256": export_sha,
            "control_arm": control,
            "contexts": len(records),
            "duplicate_copies_not_counted": duplicate_copies,
            "missing_parent_ids": sum(c not in mapping for c in by_context),
            "complete_roster": complete,
            "independent_unit": "parent",
            "family_generalization": "unmeasured",
        }
        by_parent = defaultdict(list)
        if complete:
            for context, value in by_context.items():
                by_parent[mapping[context]].append(number(value))
        means = [statistics.mean(v) for v in by_parent.values()]
        n = len(means) if complete else None
        details["parents"] = n
        note = "Equal-weight parent means after averaging paired cuts within each parent. Exploratory parent inputs only; repeated epochs are separate, no new-family or qualification verdict. Missing parent joins leave the full-panel aggregate NULL."
        for parent, values in sorted(by_parent.items()):
            rows.append(
                row(
                    "parent_l6_paired_mean_delta",
                    statistics.mean(values),
                    date=date,
                    commit=commit,
                    source=paired_source,
                    scope=scope + "/parent/" + parent,
                    n=len(values),
                    unit="loss",
                    note=note,
                    kind="parent_cluster_input",
                    context_set=roster_sha,
                    details=details | {"parent_id": parent},
                )
            )
        variance = statistics.variance(means) if complete and len(means) > 1 else None
        for metric, value, unit in [
            (
                "parent_l6_paired_mean_delta",
                statistics.mean(means) if means else None,
                "loss",
            ),
            ("parent_l6_paired_variance", variance, "loss squared"),
            (
                "parent_l6_paired_mean_SE",
                math.sqrt(variance / n) if variance is not None else None,
                "loss",
            ),
            ("paired_parent_count", n, "parents"),
        ]:
            rows.append(
                row(
                    metric,
                    value,
                    date=date,
                    commit=commit,
                    source=paired_source,
                    scope=scope,
                    n=n,
                    unit=unit,
                    note=note,
                    kind="parent_cluster_summary",
                    context_set=roster_sha,
                    details=details,
                )
            )
    return rows


if __name__ == "__main__":
    main()
