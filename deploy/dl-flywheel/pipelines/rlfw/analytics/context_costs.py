"""Measured per-context L6 paired costs and exploratory normal power sizing."""

import hashlib
import json
import math
import re
import statistics

from .scratch import (
    EXPORT_SCHEMA,
    R22_SCHEMAS,
    control_arm,
    number,
    receipt_rows,
    r22_points,
)

SCHEMA = "jope.rlfw-scratch-context-costs/v1"
IDENTITY = {
    "anchor_parameter_sha256",
    "trie_sha256",
    "normalizer_sha256",
    "validation_split_sha256",
    "evaluation_protocol_sha256",
}


def roster_hash(contexts):
    return hashlib.sha256(
        json.dumps(sorted(contexts), separators=(",", ":")).encode()
    ).hexdigest()


def comparison_power(n, sigma, delta=0.005, alpha=0.05):
    if n < 1 or sigma <= 0:
        raise ValueError("positive n and observed sigma required")
    normal = statistics.NormalDist()
    z = normal.inv_cdf(1 - alpha / 2)
    signal = abs(delta) * math.sqrt(n) / sigma
    return normal.cdf(-z - signal) + 1 - normal.cdf(z - signal)


def contexts_needed(sigma, delta=0.005, alpha=0.05, target=0.8):
    if not 0 < sigma or not 0 < abs(delta) or not 0 < alpha < 1 or not 0 < target < 1:
        raise ValueError("finite positive power inputs required")
    lo, hi = 1, 1
    while comparison_power(hi, sigma, delta, alpha) < target:
        hi *= 2
        if hi > 100_000_000:
            return None
    while lo < hi:
        middle = (lo + hi) // 2
        if comparison_power(middle, sigma, delta, alpha) >= target:
            hi = middle
        else:
            lo = middle + 1
    return lo


def context_rows(
    export,
    parent,
    *,
    source,
    receipt_sha,
    consumer_commit,
    as_of_date,
    native_index_sha=None,
):
    if (
        export.get("schema") != SCHEMA
        or export.get("non_qualifying") is not True
        or export.get("benchmark_valid") is not False
        or export.get("promotion_input") is not False
    ):
        raise ValueError("explicit non-qualifying context export required")
    identity = export.get("comparison_identity", {})
    if not IDENTITY <= identity.keys() or (
        native_index_sha is None and any(not identity[k] for k in IDENTITY)
    ):
        raise ValueError("full context cost comparison identity required")
    anchor = (
        (parent or {}).get("inputs", {}).get("checkpoint", {}).get("parameter_sha256")
    )
    if parent is not None and (
        not anchor or anchor != identity["anchor_parameter_sha256"]
    ):
        raise ValueError("context export differs from physical parent run anchor")
    roster = export["context_ids"]
    if (
        not roster
        or any(not isinstance(c, str) or not c for c in roster)
        or len(set(roster)) != len(roster)
        or roster_hash(roster) != export["context_set_sha256"]
    ):
        raise ValueError("unique pinned fixed context roster required")
    observations = {}
    for block in export["observations"]:
        arm, epoch, updates = block["arm"], block["epoch"], block["updates"]
        if (
            type(epoch) is not int
            or epoch < 0
            or type(updates) is not int
            or updates < 0
        ):
            raise ValueError("reported epoch/update budget required")
        key = (arm, epoch)
        if key in observations:
            raise ValueError("duplicate arm/epoch context block")
        costs = {}
        for item in block["contexts"]:
            context = item["context_id"]
            if context not in roster or context in costs:
                raise ValueError("context outside fixed roster or duplicate")
            if "items" in item and (
                type(item["items"]) is not int or item["items"] < 1
            ):
                raise ValueError("positive reported evaluation item count required")
            number(item["mean_cost"])
            costs[context] = item
        observations[key] = (updates, costs)
    comparator = control_arm({a for a, _ in observations}, export.get("control_arm"))
    rows = []
    date = as_of_date
    for (arm, epoch), (updates, treatment) in sorted(observations.items()):
        if arm == comparator:
            continue
        control_updates, control = observations.get((comparator, epoch), (None, {}))
        matched = []
        for context in sorted(set(control) & set(treatment)):
            c, t = control[context], treatment[context]
            if (
                control_updates != updates
                or c.get("item_set_sha256") != t.get("item_set_sha256")
                or c.get("items") != t.get("items")
                or c.get("support_denominator_sha256")
                != t.get("support_denominator_sha256")
                or c["mean_cost"] is None
                or t["mean_cost"] is None
            ):
                continue
            matched.append((context, c["mean_cost"], t["mean_cost"]))
        complete = len(matched) == len(roster)
        details = {
            "lane": export["lane"],
            "run": export["run"],
            "arm": arm,
            "control_arm": comparator,
            "epoch": epoch,
            "updates": updates,
            "comparison_identity": identity,
            "parent_run_receipt_sha256": export.get("run_receipt_sha256"),
            "parent_receipt_verified": parent is not None,
            "export_sha256": receipt_sha,
            "fixed_contexts": len(roster),
            "matched_contexts": len(matched),
            "provider_source_commit": export.get("source_commit"),
            "complete_roster": complete,
            "native_index_sha256": native_index_sha,
        }
        note = "Context-macro L6 cost deltas; exploratory normal approximation assumes independent contexts, no family gate. Target delta=.005, alpha=.05, power=.80. Absent per-context item-set/count identities remain unmeasured; shared declared evaluation protocol is required."
        if parent is None:
            note += " Final run receipt/anchor reconciliation unmeasured; provider-declared identities only."
        if native_index_sha:
            note += " Native single-index/single-JSONL cohort; absent provider input/protocol hashes remain NULL."

        def emit(kind, metric, value, unit, extra=None):
            rows.append(
                {
                    "date": date,
                    "kind": kind,
                    "metric": metric,
                    "scope": f"{export['lane']}/{export['run']}/{arm}",
                    "status": "NON-QUALIFYING"
                    if value is not None
                    else "unmeasured / NON-QUALIFYING",
                    "value": number(value),
                    "n": 1 if kind == "scratch_context_pair" else len(matched),
                    "unit": unit,
                    "note": note
                    if complete
                    else "Incomplete/mismatched fixed roster; full-panel variance/power unmeasured. "
                    + note,
                    "sources": source,
                    "source_commit": consumer_commit,
                    "context_set_sha256": export["context_set_sha256"],
                    "details": json.dumps(
                        details | (extra or {}), sort_keys=True, allow_nan=False
                    ),
                }
            )

        for context, c, t in matched:
            emit(
                "scratch_context_pair",
                "l6_context_control_minus_arm",
                c - t,
                "loss",
                {"context_id": context, "control_cost": c, "arm_cost": t},
            )
        delta = [c - t for _, c, t in matched]
        enough = complete and len(delta) >= 2
        mean = statistics.mean(delta) if complete and delta else None
        variance = statistics.variance(delta) if enough else None
        sigma = math.sqrt(variance) if variance is not None else None
        se = sigma / math.sqrt(len(delta)) if sigma is not None else None
        emit("scratch_context_summary", "l6_paired_mean_delta", mean, "loss")
        emit(
            "scratch_context_summary",
            "l6_between_context_paired_delta_variance",
            variance,
            "loss squared",
        )
        emit(
            "scratch_context_summary",
            "l6_control_between_context_cost_variance",
            statistics.variance([c for _, c, _ in matched]) if enough else None,
            "loss squared",
        )
        emit(
            "scratch_context_summary",
            "l6_arm_between_context_cost_variance",
            statistics.variance([t for _, _, t in matched]) if enough else None,
            "loss squared",
        )
        emit("scratch_context_summary", "l6_paired_mean_SE", se, "loss")
        for metric, sign in (
            ("l6_paired_mean_CI95_lower", -1),
            ("l6_paired_mean_CI95_upper", 1),
        ):
            emit(
                "scratch_context_summary",
                metric,
                mean + sign * 1.959963984540054 * se if se is not None else None,
                "loss",
            )
        emit(
            "scratch_context_summary",
            "l6_contexts_needed_delta_0_005_power_0_80",
            contexts_needed(sigma) if sigma is not None and sigma > 0 else None,
            "paired contexts",
        )
        emit(
            "scratch_context_summary",
            "l6_normal_power_at_observed_contexts",
            comparison_power(len(delta), sigma)
            if sigma is not None and sigma > 0
            else None,
            "fraction",
        )
    return rows


def native_context_rows(
    index, events, rosters, *, source, index_sha, consumer_commit, as_of_date
):
    """CL's paired_eval.py native index/JSONL, without importing evaluator code."""
    if index.get("schema") != "jope.r22-lpi-scratch-paired-index/v1":
        raise ValueError("native paired index schema required")
    if "scratch" not in index.get("label", "").lower() or not index.get("eval_path"):
        raise ValueError("native scratch evaluation definition required")
    for name, ref in index["sets"].items():
        ids = rosters[name]
        sha = hashlib.sha256(
            json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if len(ids) != ref["contexts"] or sha != ref["sha256"]:
            raise ValueError("native context roster hash/count mismatch")
    anchor = next(
        (
            c.get("parameter_sha256")
            for c in index["checkpoints"]
            if c.get("checkpoint") == "start"
        ),
        None,
    )
    result = []
    coefficients = {
        k: float(v)
        for k, v in re.findall(
            r"\b(ce|bt|list|q|feas|rt) ([0-9.]+)", index["eval_path"]
        )
    }
    for set_name in ("a", "b_prefix_1000", "b"):
        if set_name not in rosters:
            continue
        observations = []
        weighted = []
        seen = set()
        for checkpoint in index["checkpoints"]:
            if (
                set_name not in checkpoint["sets_evaluated"]
                or checkpoint["arm"] == "all"
            ):
                continue
            key = (checkpoint["arm"], checkpoint["epoch"])
            if key in seen:
                raise ValueError("duplicate native arm/epoch/set")
            seen.add(key)
            selected = [
                r
                for r in events
                if r.get("set_list") == set_name
                and all(
                    r.get(k) == checkpoint.get(k)
                    for k in (
                        "checkpoint",
                        "arm",
                        "epoch",
                        "updates",
                        "parameter_sha256",
                        "state_sha256",
                    )
                )
            ]
            if len(selected) != len(rosters[set_name]) or any(
                "l6_loss" not in r for r in selected
            ):
                continue
            observations.append(
                {
                    "arm": checkpoint["arm"],
                    "epoch": checkpoint["epoch"],
                    "updates": checkpoint["updates"],
                    "contexts": [
                        {"context_id": r["context_id"], "mean_cost": r["l6_loss"]}
                        for r in selected
                    ],
                }
            )
            if set(coefficients) == {"ce", "bt", "list", "q", "feas", "rt"} and all(
                "numerators" in r and "denominators" in r for r in selected
            ):
                totals = {
                    k: sum(
                        number(r["denominators"][k], nonnegative=True) for r in selected
                    )
                    for k in coefficients
                }
                contributions = []
                for row in selected:
                    cost = len(selected) * sum(
                        coefficients[k] * number(row["numerators"][k]) / totals[k]
                        if totals[k]
                        else 0
                        for k in coefficients
                    )
                    contributions.append(
                        {
                            "context_id": row["context_id"],
                            "mean_cost": cost,
                            "support_denominator_sha256": hashlib.sha256(
                                json.dumps(row["denominators"], sort_keys=True).encode()
                            ).hexdigest(),
                        }
                    )
                macro = (
                    checkpoint.get("aggregate", {})
                    .get(set_name, {})
                    .get("l6_macro_loss")
                )
                if (
                    macro is None
                    or abs(
                        statistics.mean(p["mean_cost"] for p in contributions)
                        - number(macro)
                    )
                    > 1e-6
                ):
                    raise ValueError(
                        "native weighted contributions do not reconcile recorded L6 macro"
                    )
                weighted.append(
                    {
                        "arm": checkpoint["arm"],
                        "epoch": checkpoint["epoch"],
                        "updates": checkpoint["updates"],
                        "contexts": contributions,
                    }
                )
        export = {
            "schema": SCHEMA,
            "non_qualifying": True,
            "benchmark_valid": False,
            "promotion_input": False,
            "lane": "CL",
            "run": index["sweep_run"].rsplit("/", 1)[-1] + "/set-" + set_name,
            "run_receipt_sha256": None,
            "context_ids": rosters[set_name],
            "context_set_sha256": roster_hash(rosters[set_name]),
            "comparison_identity": {
                "anchor_parameter_sha256": anchor,
                "trie_sha256": None,
                "normalizer_sha256": None,
                "validation_split_sha256": index["sets"][set_name]["sha256"],
                "evaluation_protocol_sha256": None,
            },
            "observations": observations,
        }
        raw_rows = context_rows(
            export,
            None,
            source=source,
            receipt_sha=index_sha,
            consumer_commit=consumer_commit,
            as_of_date=as_of_date,
            native_index_sha=index_sha,
        )
        result.extend(
            r
            | {
                "metric": r["metric"].replace("l6_", "l6_raw_", 1),
                "scope": r["scope"] + "/raw-context-loss",
                "note": "Raw per-context loss mean differs from pooled supported-term L6 macro. "
                + r["note"],
            }
            for r in raw_rows
        )
        export["observations"] = weighted
        result.extend(
            r
            | {
                "note": "Supported-term contributions N*sum(coef*num_i/total_den); mean reconciles saved L6 macro. "
                + r["note"]
            }
            for r in context_rows(
                export,
                None,
                source=source,
                receipt_sha=index_sha,
                consumer_commit=consumer_commit,
                as_of_date=as_of_date,
                native_index_sha=index_sha,
            )
        )
    return result


def indexed_context_rows(
    index,
    checkpoints,
    parent,
    events,
    rosters,
    *,
    source,
    index_sha,
    parent_sha,
    consumer_commit,
    as_of_date,
):
    """CL v5/v6 in-process paired rows, bound to checkpoint index and evaluation receipt."""
    if (
        index.get("schema") != "jope.r22-lpi-scratch-paired-index/v1"
        or checkpoints.get("schema") != "jope.r22-lpi-scratch-checkpoint-index/v1"
        or parent.get("schema") not in R22_SCHEMAS
    ):
        raise ValueError("native indexed paired schemas required")
    if (
        parent.get("non_qualifying") is not True
        or parent.get("benchmark_valid") is not False
        or parent.get("promotion_input") is not False
    ):
        raise ValueError("native indexed scratch boundary required")
    if index["run"] != checkpoints["run"]:
        raise ValueError("paired/checkpoint index run mismatch")
    coefficients = parent["inputs"]["l6_coefficients"]
    if set(coefficients) != {"ce", "bt", "list", "q", "feas", "rt"}:
        raise ValueError("complete declared supported-term coefficients required")
    anchor = parent["inputs"]["checkpoint"]["parameter_sha256"]
    evaluations = {(p["arm"], p["epoch"]): p for p in r22_points(parent)}
    output = []
    for set_name, ref in index["sets"].items():
        ids = rosters[set_name]
        # The in-process producer hashes the ordered roster, including its b sample.
        ordered_sha = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest()
        if len(ids) != ref["contexts"] or ordered_sha != ref["sha256"]:
            raise ValueError("indexed fixed roster hash/count mismatch")
        observations = []
        for checkpoint in checkpoints["checkpoints"]:
            key = (checkpoint["arm"], checkpoint["epoch"])
            evaluation = evaluations.get(key)
            if evaluation is None:
                continue
            if any(
                evaluation.get(k) != checkpoint.get(k)
                for k in ("parameter_sha256", "updates")
            ):
                raise ValueError("checkpoint/evaluation parameter or update mismatch")
            selected = [
                r
                for r in events
                if r.get("set_list") == set_name
                and r.get("checkpoint") == checkpoint["tag"]
                and all(
                    r.get(k) == checkpoint.get(k)
                    for k in ("arm", "epoch", "updates", "parameter_sha256")
                )
            ]
            if len(selected) != len(ids):
                continue
            totals = {
                k: sum(number(r["denominators"][k], nonnegative=True) for r in selected)
                for k in coefficients
            }
            costs = [
                {
                    "context_id": r["context_id"],
                    "mean_cost": len(ids)
                    * sum(
                        number(coefficients[k]) * number(r["numerators"][k]) / totals[k]
                        if totals[k]
                        else 0
                        for k in coefficients
                    ),
                    "support_denominator_sha256": hashlib.sha256(
                        json.dumps(r["denominators"], sort_keys=True).encode()
                    ).hexdigest(),
                }
                for r in selected
            ]
            macro = evaluation.get(
                {"a": "a_l6", "b": "b_l6", "heldout": "heldout_l6"}.get(set_name, ""),
                {},
            ).get("loss")
            if (
                macro is None
                or abs(statistics.mean(c["mean_cost"] for c in costs) - number(macro))
                > 1e-6
            ):
                raise ValueError(
                    "indexed weighted costs do not reconcile recorded L6 macro"
                )
            observations.append(
                {
                    "arm": checkpoint["arm"],
                    "epoch": checkpoint["epoch"],
                    "updates": checkpoint["updates"],
                    "contexts": costs,
                }
            )
        # Both explicitly named baselines remain separate panels; never pool their rows.
        for comparator in ("control", "r23"):
            if comparator not in {b["arm"] for b in observations}:
                continue
            export = {
                "schema": SCHEMA,
                "non_qualifying": True,
                "benchmark_valid": False,
                "promotion_input": False,
                "lane": "CL",
                "run": index["run"].rsplit("/", 1)[-1]
                + "/set-"
                + set_name
                + "/vs-"
                + comparator,
                "control_arm": comparator,
                "run_receipt_sha256": parent_sha,
                "context_ids": ids,
                "context_set_sha256": roster_hash(ids),
                "comparison_identity": {
                    "anchor_parameter_sha256": anchor,
                    "trie_sha256": None,
                    "normalizer_sha256": None,
                    "validation_split_sha256": ref["sha256"],
                    "evaluation_protocol_sha256": None,
                },
                "observations": observations,
            }
            result = context_rows(
                export,
                parent,
                source=source,
                receipt_sha=index_sha,
                consumer_commit=consumer_commit,
                as_of_date=as_of_date,
                native_index_sha=index_sha,
            )
            output.extend(
                r
                | {
                    "note": "Indexed supported-term contributions, saved L6 macro reconciled; cohort: "
                    + ref.get("definition", set_name)
                    + ". "
                    + r["note"]
                }
                for r in result
            )
    return output


def scope_rows(document, *, consumer_commit, as_of_date, source):
    """Scope occupancy is an alternative to process wall costs; never sum them."""
    properties = document["properties"]
    start = document.get("active_enter_unix")
    end = document.get("inactive_enter_unix")
    observed = document["observed_remote_unix"]
    active = properties.get("ActiveState") == "active"
    seconds = (
        (observed if active else end) - start if start and (active or end) else None
    )
    if seconds is not None and seconds < 0:
        raise ValueError("scope clock precedes start")
    cpu = properties.get("CPUUsageNSec", "[not set]")
    cpu = int(cpu) / 1e9 if cpu.isdigit() else None
    run = document["run"]
    rows = receipt_rows(
        {
            "schema": EXPORT_SCHEMA,
            "non_qualifying": True,
            "benchmark_valid": False,
            "promotion_input": False,
            "points": [],
            "costs": {},
        },
        run=run,
        lane=document["lane"],
        source=source,
        receipt_sha=document["sha256"],
        consumer_commit=consumer_commit,
        as_of_date=as_of_date,
    )
    template = rows[0]
    note = "Scope UTC start/end or snapshot time, including launch/init; alternative occupancy measure, never added to process GPU wall-hours. GPU assignment is owner-declared, not utilization. RuntimeMax is a cap, not elapsed."
    return [
        template
        | {
            "kind": "scratch_scope_cost",
            "metric": metric,
            "value": value,
            "status": ("PARTIAL / NON-QUALIFYING" if active else "NON-QUALIFYING")
            if value is not None
            else "unmeasured / NON-QUALIFYING",
            "unit": unit,
            "note": note,
            "details": json.dumps(document, sort_keys=True),
        }
        for metric, value, unit in [
            (
                "scope_wall_hours",
                seconds / 3600 if seconds is not None else None,
                "scope wall-hours",
            ),
            (
                "scope_gpu_assigned_wall_hours",
                seconds / 3600
                if seconds is not None and document.get("gpu_count") == 1
                else None,
                "GPU assigned scope wall-hours",
            ),
            (
                "scope_cumulative_cpu_core_hours",
                cpu / 3600 if cpu is not None else None,
                "scope core-hours",
            ),
        ]
    ]
