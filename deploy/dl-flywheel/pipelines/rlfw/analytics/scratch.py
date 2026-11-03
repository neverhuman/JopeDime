"""Read-only scratch receipt analytics. Scratch observations never qualify HWM gain."""

import datetime as dt
import html
import json
import math
from collections import defaultdict


R22_SCHEMAS = {
    "jope.r22-lpi-scratch-train/v1",
    "jope.r22-lpi-scratch-train/v2",
    "jope.r22-lpi-scratch-train/v3",
    "jope.r22-lpi-scratch-train/v4",
    "jope.r22-lpi-scratch-train/v5",
    "jope.r22-lpi-scratch-train/v6",
}
EXPORT_SCHEMA = "jope.rlfw-scratch-measurements/v1"


def control_arm(arms, declared=None):
    """Select the named native comparator; never choose another treatment arm."""
    if declared is not None:
        if not isinstance(declared, str) or declared not in arms:
            raise ValueError("declared control absent from observations")
        return declared
    candidates = set(arms) & {"control", "r23_control"}
    if len(candidates) > 1:
        raise ValueError("ambiguous control; explicit control_arm required")
    return next(iter(candidates), "control")


def number(value, *, nonnegative=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric observation required")
    if not math.isfinite(value) or nonnegative and value < 0:
        raise ValueError("finite observation required")
    return float(value)


def r22_points(receipt):
    """Keep reported evaluation epochs, including disqualified arms and epoch zero."""
    points = list(receipt.get("initial_eval", []))
    for epoch in receipt.get("epochs", []):
        for point in epoch.get("eval", []):
            if point["epoch"] != epoch["epoch"]:
                raise ValueError("receipt epoch identity mismatch")
            points.append(point)
    return points


def verify_r22_metrics(receipt, events):
    """Join the selected evaluation fields to the copied JSONL, never re-evaluate."""
    expected = {(p["arm"], p["epoch"]): p for p in r22_points(receipt)}
    if len(expected) != len(r22_points(receipt)):
        raise ValueError("duplicate evaluation identity")
    observed = {}
    steps = defaultdict(int)
    for event in events:
        if event.get("kind") == "step":
            steps[event["arm"]] += 1
        if event.get("kind") == "eval":
            key = (event["arm"], event["epoch"])
            if key in observed:
                raise ValueError("duplicate JSONL evaluation")
            observed[key] = event
    if set(expected) != set(observed):
        raise ValueError("receipt/JSONL evaluation inventory mismatch")
    for key, p in expected.items():
        for field in (
            "lpi_validation_eval_only",
            "l6_validation_panel",
            "l6_full_validation",
            "probe",
            "updates",
            "kl_disqualify",
            "kl_alarm",
            "a_l6",
            "b_l6",
            "heldout_l6",
            "heldout_lpi",
        ):
            if p.get(field) != observed[key].get(field):
                raise ValueError("receipt/JSONL evaluation value mismatch")
    if "updates_per_arm" not in receipt and steps:
        return "Receipt final update inventory absent; JSONL steps preserved as raw evidence, final reported updates unmeasured."
    reported = {k: v for k, v in receipt.get("updates_per_arm", {}).items() if v != 0}
    if dict(steps) != reported:
        raise ValueError("receipt/JSONL update inventory mismatch")
    return None


def normalize(receipt, receipt_sha, closure=None):
    """Adapt the two retained R22 schemas and the explicit multi-lane export schema."""
    schema = receipt.get("schema")
    if schema == "jope.r22-lpi-scratch-paired-eval/v1":
        if (
            receipt.get("non_qualifying") is not True
            or "scratch" not in receipt.get("label", "").lower()
        ):
            raise ValueError("native paired evaluation scratch boundary required")
        start, end = (
            number(receipt.get("started_unix")),
            number(receipt.get("finished_unix")),
        )
        elapsed = end - start if start is not None and end is not None else None
        return {
            "schema": EXPORT_SCHEMA,
            "non_qualifying": True,
            "benchmark_valid": False,
            "promotion_input": False,
            "started_unix": start,
            "points": [],
            "costs": {
                "gpu_process_wall_seconds": elapsed
                if str(receipt.get("device", "")).startswith("cuda")
                else None,
                "gpu_count": 1
                if str(receipt.get("device", "")).startswith("cuda")
                else None,
            },
            "run_error": receipt.get("error"),
            "stop_reason": receipt.get("state", "unmeasured"),
            "updates_per_arm": {},
        }
    if schema in {"jope.rlfw-g-abl-real-pairs/v1", "jope.rlfw-g-abl-r23-pairs/v1"}:
        if (
            receipt.get("non_qualifying") is not True
            or receipt.get("promoted") is not False
            or receipt.get("benchmark_valid") is not False
            or receipt.get("live_state") is not False
        ):
            raise ValueError("real-pair scratch boundary flags required")
        points = []
        for arm, data in receipt.get("arms", {}).items():
            for epoch, loss in enumerate(data.get("epoch_loss", [])):
                points.append(
                    {
                        "arm": arm,
                        "epoch": epoch,
                        "metric": "real_pair_training_loss",
                        "value": loss,
                        "unit": "loss",
                        "n": data.get("pairs"),
                        "note": "Reported training loss; provider epochs start at zero. Validation loss and paired q uncertainty unmeasured.",
                    }
                )
            guard = data.get("guard", {})
            for check in guard.get("history", []):
                for field in ("kl_mean", "kl_max"):
                    points.append(
                        {
                            "arm": arm,
                            "epoch": check["epoch"],
                            "metric": "r23_exact_trie_" + field + "_to_v1",
                            "value": check.get(field),
                            "unit": "nats",
                            "n": guard.get("probe_contexts"),
                            "updates": check.get("at_step"),
                            "kl_disqualify": guard.get("tripped") is True,
                            "note": "Provider exact-trie guard; epoch0 is pre-update. Sticky HOLD/adoption status retained; measured-support KL separate.",
                        }
                    )
            kl = (
                receipt.get("eval", {})
                .get(arm, {})
                .get("policy_kl_to_anchor_on_measured_support")
            )
            points.append(
                {
                    "arm": arm,
                    "epoch": None,
                    "metric": "measured_support_kl_to_anchor",
                    "value": None if kl == "unmeasured" else kl,
                    "unit": "nats",
                    "n": None,
                    "note": "Final KL on labelled in-catalog support, not full-trie/R22 64-probe KL; validation epoch unmeasured.",
                }
            )
        return {
            "schema": EXPORT_SCHEMA,
            "non_qualifying": True,
            "benchmark_valid": False,
            "promotion_input": False,
            "source_commit": receipt.get("source_commit"),
            "anchor_sha256": receipt.get(
                "v1_anchor_parameter_sha256", receipt.get("ckpt_copy_parameter_sha256")
            ),
            "points": points,
            "costs": {},
            "updates_per_arm": {
                a: d["per_arm_steps"]
                for a, d in receipt.get("arms", {}).items()
                if "per_arm_steps" in d
            },
            "stop_reason": "real-pair scratch; checked measurement pairs="
            + str(receipt.get("arena_checked_pairs_reconfirmed")),
        }
    if schema == "jope.rlfw-g-abl-c11/v1":
        synthetic = receipt.get("gpu", {}).get("synthetic", {})
        if (
            receipt.get("promoted") is not False
            or synthetic.get("synthetic_fixture_not_a_flywheel_result") is not True
        ):
            raise ValueError("legacy C11 synthetic-only receipt boundary required")
        points = [
            {
                "arm": "synthetic_" + arm,
                "epoch": None,
                "metric": "synthetic_toy_train_loss_last",
                "value": synthetic.get(arm + "_loss_last"),
                "unit": "loss",
                "n": None,
                "note": "Synthetic toy training loss only; no real pairs, validation loss, benchmark or promotion input.",
            }
            for arm in ("ipo", "dpo")
        ]
        return {
            "schema": EXPORT_SCHEMA,
            "non_qualifying": True,
            "benchmark_valid": False,
            "promotion_input": False,
            "source_commit": receipt.get("source_commit"),
            "points": points,
            "costs": {},
            "stop_reason": "legacy synthetic-only; real train pairs="
            + str(receipt.get("real_train_pairs")),
            "updates_per_arm": {
                "synthetic_" + arm: synthetic[arm + "_steps"]
                for arm in ("ipo", "dpo")
                if arm + "_steps" in synthetic
            },
        }
    if schema in {"jope.cx-kv-checkpoint-eval/v1"} | {
        f"jope.cx-op-r{i}-kv-eval/v1" for i in range(5, 10)
    }:
        if (
            receipt.get("non_qualifying") is not True
            or receipt.get("promotion_input") is not False
            or receipt.get("program_execution") is not False
            or receipt.get("optimizer_updates") != 0
        ):
            raise ValueError("scratch inference boundary flags required")
        m2 = receipt.get("m2", {})
        n = m2.get("multi_label_contexts")
        arm_name = (
            receipt["checkpoint_name"]
            if schema == "jope.cx-kv-checkpoint-eval/v1"
            else receipt["arm"]
        )
        points = [
            {
                "arm": arm_name,
                "epoch": None,
                "metric": metric,
                "value": value,
                "unit": unit,
                "n": count,
            }
            for metric, value, unit, count in [
                (
                    "development_panel_kl_to_anchor",
                    receipt.get("panel_kl_mean_nats"),
                    "nats",
                    receipt.get(
                        "context_panel_count", len(receipt.get("context_panel", []))
                    ),
                ),
                (
                    "development_panel_entropy",
                    receipt.get("entropy_mean_nats"),
                    "nats",
                    receipt.get(
                        "context_panel_count", len(receipt.get("context_panel", []))
                    ),
                ),
                (
                    "M2_selected_support_spearman",
                    m2.get("context_macro_spearman"),
                    "rho",
                    n,
                ),
                (
                    "M2_measured_top8_recall",
                    m2.get("topk_measured_support", {})
                    .get("8", {})
                    .get("macro_recall"),
                    "fraction",
                    n,
                ),
                (
                    "M2_measured_top32_recall",
                    m2.get("topk_measured_support", {})
                    .get("32", {})
                    .get("macro_recall"),
                    "fraction",
                    n,
                ),
            ]
        ]
        return {
            "schema": EXPORT_SCHEMA,
            "non_qualifying": True,
            "benchmark_valid": False,
            "promotion_input": False,
            "source_commit": receipt.get("source_commit"),
            "started_unix": receipt.get("started_unix"),
            "points": points,
            "anchor_sha256": receipt.get("v1_anchor_parameter_sha256"),
            "input_manifest_sha256": receipt.get("code_manifest_sha256"),
            "costs": {
                "gpu_process_wall_seconds": receipt.get("wall_seconds"),
                "gpu_count": 1,
            },
            "updates_per_arm": {},
            "stop_reason": "checkpoint inference only; training epoch unmeasured",
        }
    if schema not in R22_SCHEMAS | {EXPORT_SCHEMA}:
        raise ValueError("unsupported scratch receipt schema")
    if (
        receipt.get("non_qualifying") is not True
        or receipt.get("benchmark_valid") is not False
        or receipt.get("promotion_input") is not False
    ):
        raise ValueError("explicit NON-QUALIFYING/benchmark/promotion flags required")
    if closure:
        if closure.get("run_receipt", {}).get("sha256") != receipt_sha:
            raise ValueError("resource closure belongs to another receipt")
    if schema == EXPORT_SCHEMA:
        return receipt
    inputs = receipt.get("inputs", {})
    config = receipt.get("config", {})
    config = config.get("body", config)
    points = []
    for p in r22_points(receipt):
        fields = (
            ("l6_heldout", p.get("heldout_l6", {}), "loss", "loss"),
            ("lpi_heldout", p.get("heldout_lpi", {}), "mean", "loss"),
            ("lpi_validation", p.get("lpi_validation_eval_only", {}), "mean", "loss"),
            ("l6_validation_panel", p.get("l6_validation_panel", {}), "loss", "loss"),
            ("l6_full_validation", p.get("l6_full_validation", {}), "loss", "loss"),
            ("kl_to_anchor", p.get("probe", {}), "kl_mean", "nats"),
            ("kl_max_context", p.get("probe", {}), "kl_max", "nats"),
        )
        for metric, values, field, unit in fields:
            if field in values:
                points.append(
                    {
                        "arm": p["arm"],
                        "epoch": p["epoch"],
                        "metric": metric,
                        "value": values[field],
                        "unit": unit,
                        "n": values.get("contexts"),
                        "updates": p.get("updates"),
                        "kl_disqualify": p.get("kl_disqualify", False),
                        "kl_alarm": p.get("kl_alarm", False),
                    }
                )
    meter = (closure or {}).get("scope", {}).get("scope_meter", {})
    cpu = meter.get("cpu_usage_usec")
    device = str(receipt.get("device", receipt.get("config", {}).get("device", "")))
    return {
        "schema": EXPORT_SCHEMA,
        "non_qualifying": True,
        "benchmark_valid": False,
        "promotion_input": False,
        "started_unix": receipt.get("started_unix"),
        "source_commit": receipt.get("source_commit"),
        "harness_sha256": receipt.get(
            "harness_sha256", inputs.get("harness_sha256", {})
        ),
        "anchor_sha256": inputs.get("checkpoint", {}).get("parameter_sha256"),
        "input_manifest_sha256": (closure or {})
        .get("data", {})
        .get("input_manifest_sha256"),
        "stop_reason": receipt.get("stop_reason", receipt.get("state", "unmeasured")),
        "run_error": receipt.get("error"),
        "kl_disqualify_threshold": config.get("kl_disqualify"),
        "points": points,
        "costs": {
            "gpu_process_wall_seconds": receipt.get("wall_seconds")
            if "NVIDIA" in device
            else None,
            "gpu_count": 1 if "NVIDIA" in device else None,
            "learner_cgroup_cpu_seconds": float(cpu) / 1e6 if cpu is not None else None,
        },
        "updates_per_arm": receipt.get("updates_per_arm", {}),
    }


def receipt_rows(
    receipt,
    *,
    run,
    lane,
    source,
    receipt_sha,
    consumer_commit,
    as_of_date,
    closure=None,
    verified_pairings=None,
):
    r = normalize(receipt, receipt_sha, closure)
    started = number(r.get("started_unix"))
    date = (
        dt.datetime.fromtimestamp(started, dt.timezone.utc).date().isoformat()
        if started is not None
        else as_of_date
    )
    identity = {
        k: r.get(k)
        for k in (
            "source_commit",
            "harness_sha256",
            "anchor_sha256",
            "input_manifest_sha256",
            "stop_reason",
            "run_error",
            "kl_disqualify_threshold",
        )
    }
    identity.update(
        run=run,
        lane=lane,
        receipt_sha256=receipt_sha,
        date_basis="run_started_unix" if started is not None else "snapshot_as_of_only",
    )
    rows = []

    def emit(
        kind,
        metric,
        value,
        unit,
        *,
        arm="all arms shared",
        n=None,
        note="",
        details=None,
        status="NON-QUALIFYING",
    ):
        value = number(value)
        rows.append(
            {
                "date": date,
                "kind": kind,
                "metric": metric,
                "scope": f"{lane}/{run}/{arm}",
                "status": status
                if value is not None
                else "unmeasured / NON-QUALIFYING",
                "value": value,
                "n": n,
                "unit": unit,
                "note": note,
                "sources": source,
                "source_commit": consumer_commit,
                "context_set_sha256": r.get("input_manifest_sha256") or "unmeasured",
                "details": json.dumps(
                    identity | (details or {}), sort_keys=True, allow_nan=False
                ),
            }
        )

    seen, lookup = set(), {}
    for p in r.get("points", []):
        epoch = p["epoch"]
        if epoch is not None and (type(epoch) is not int or epoch < 0):
            raise ValueError("nonnegative integer epoch required")
        key = (p["arm"], epoch, p["metric"])
        if key in seen:
            raise ValueError("duplicate arm/epoch/metric")
        seen.add(key)
        number(p.get("value"))
        if p.get("n") is not None and (type(p["n"]) is not int or p["n"] < 0):
            raise ValueError("nonnegative sample count required")
        lookup[key] = p
        disqualified = p.get("kl_disqualify") is True
        emit(
            "scratch_epoch" if epoch is not None else "scratch_evaluation",
            p["metric"],
            p.get("value"),
            p["unit"],
            arm=p["arm"],
            n=p.get("n"),
            details=p,
            status="KL-DISQUALIFIED / NON-QUALIFYING"
            if disqualified
            else "NON-QUALIFYING",
            note=p.get("note")
            or (
                "Reported validation/probe observation; no benchmark or promotion claim."
                if epoch is not None
                else "Checkpoint inference; training epoch unmeasured. Development panel KL differs from R22 64 probes; M2 support is selected."
            ),
        )
    comparator = control_arm(
        {p["arm"] for p in r.get("points", [])}, r.get("control_arm")
    )
    for (arm, epoch, metric), p in lookup.items():
        if arm == comparator or p["unit"] != "loss":
            continue
        control = lookup.get((comparator, epoch, metric))
        delta = None
        if (
            control
            and control["unit"] == p["unit"]
            and control.get("n") == p.get("n")
            and type(control.get("updates")) is int
            and control.get("updates") == p.get("updates")
            and control.get("value") is not None
            and p.get("value") is not None
        ):
            delta = control["value"] - p["value"]
        emit(
            "scratch_delta",
            metric + "_delta_vs_control",
            delta,
            "loss",
            arm=arm,
            n=p.get("n"),
            details=p
            | {
                "control_arm": comparator,
                "delta_direction": "control minus arm; positive lowers loss",
            },
            note="Same-run/same-epoch aggregate loss difference. Fold-q SE cannot resolve loss units; paired loss uncertainty unmeasured.",
        )
    costs = r.get("costs", {})
    gpu_seconds = number(costs.get("gpu_process_wall_seconds"), nonnegative=True)
    gpu_count = costs.get("gpu_count")
    if gpu_count is not None and (type(gpu_count) is not int or gpu_count < 1):
        raise ValueError("positive integer GPU count required")
    emit(
        "scratch_cost",
        "gpu_process_wall_hours",
        gpu_seconds * gpu_count / 3600
        if gpu_seconds is not None and gpu_count is not None
        else None,
        "GPU wall-hours",
        note="Shared run total once, not per-arm; reported GPU process wall occupancy, not utilization or inclusive launcher time.",
    )
    cpu = number(costs.get("learner_cgroup_cpu_seconds"), nonnegative=True)
    emit(
        "scratch_cost",
        "learner_cpu_core_hours",
        cpu / 3600 if cpu is not None else None,
        "learner cgroup core-hours",
        note="Copied run-bound cgroup CPU meter; not evaluator CPU.",
    )
    emit(
        "scratch_cost",
        "G_per_evaluator_core_hour",
        None,
        "improvements/core-hour",
        note="No qualifying HWM improvement/evaluator-cgroup join; optimizer updates and loss deltas are not G.",
    )
    for arm, count in r.get("updates_per_arm", {}).items():
        if type(count) is not int or count < 0:
            raise ValueError("nonnegative integer optimizer updates required")
        emit(
            "scratch_cost",
            "optimizer_updates_reported",
            count,
            "updates",
            arm=arm,
            note="Receipt-reported updates; analytics does not replay optimizer/checkpoint tensors.",
        )
    for p in r.get("q_comparisons", []):
        delta, se = number(p.get("delta_q")), number(p.get("se_pair"), nonnegative=True)
        bound = p.get("paired_evidence_identity")
        required = {
            "context",
            "dataset_hash",
            "cut_id",
            "evaluator",
            "vocabulary",
            "protocol",
            "challenger_sha",
            "incumbent_sha",
        }
        proof = (verified_pairings or {}).get(p.get("paired_evidence_sha256"))
        matched = (
            proof is not None
            and proof.get("identity") == bound
            and proof.get("se_pair") == se
            and proof.get("delta_q") == delta
            and isinstance(bound, dict)
            and required <= bound.keys()
            and all(bound[k] for k in required)
            and p.get("unit") == "log_rmse_ratio_q"
            and p.get("paired_evidence_sha256")
            and se is not None
            and se > 0
        )
        status = "unmeasured / NON-QUALIFYING"
        ratio = None
        if matched and delta is not None:
            ratio = abs(delta) / se
            status = (
                "EXCEEDS PAIRED 1.96 SE" if ratio > 1.96 else "WITHIN PAIRED NOISE"
            ) + " / NON-QUALIFYING"
        emit(
            "scratch_resolution",
            "paired_q_abs_delta_over_SE",
            ratio,
            "SE units",
            arm=p.get("arm", "unmeasured"),
            details=p,
            status=status,
            note="Per-comparison, matching-identity paired evidence only. Retained median SE is a reference, not transferable uncertainty or a family-level gate.",
        )
    return rows


def render_scratch(rows):
    """Inline SVG curves and an escaped table, built from Parquet readback only."""
    selected = [r for r in rows if r["kind"].startswith("scratch_")]
    if not selected:
        return ""
    esc = html.escape
    body = "<h2>NON-QUALIFYING scratch training / evaluation</h2><p>Reported observations only. Loss/KL units differ from paired evaluator-q uncertainty. Shared GPU/CPU costs are charged once per run; missing joins stay unmeasured.</p>"
    curves = defaultdict(lambda: defaultdict(list))
    for r in selected:
        if r["kind"] == "scratch_epoch" and r["value"] is not None:
            d = json.loads(r["details"])
            curves[(d["lane"], d["run"], r["metric"], r["unit"])][d["arm"]].append(
                (d["epoch"], r["value"])
            )
    palette = ["#8ed7ca", "#f0b46a", "#b4a7ff", "#f58aa2", "#8bc9ff"]
    for (lane, run, metric, unit), arms in sorted(curves.items()):
        points = [p for values in arms.values() for p in values]
        xmax = max(1, max(x for x, _ in points))
        low, high = min(y for _, y in points), max(y for _, y in points)
        span = high - low or max(1, abs(high)) * 0.05
        title = f"{lane} / {run} · {metric} ({unit})"
        body += (
            "<h3>"
            + esc(title)
            + '</h3><svg role="img" viewBox="0 0 720 250" style="max-width:720px;width:100%;background:#182735"><title>'
            + esc(title)
            + "</title>"
        )
        body += '<path d="M55 15 V205 H705" fill="none" stroke="#aac3da"/>'
        body += f'<text x="5" y="25" fill="#e2edf7">{high:.5g}</text><text x="5" y="205" fill="#e2edf7">{low:.5g}</text><text x="55" y="225" fill="#e2edf7">epoch 0</text><text x="640" y="225" fill="#e2edf7">{xmax}</text>'
        for i, (arm, values) in enumerate(sorted(arms.items())):
            xy = [
                (55 + 640 * x / xmax, 200 - 180 * (y - low) / span)
                for x, y in sorted(values)
            ]
            color = palette[i % len(palette)]
            body += f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in xy)}" fill="none" stroke="{color}" stroke-width="2"/>'
            for x, y in xy:
                body += f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>'
            body += f'<text x="{55 + i * 145}" y="245" fill="{color}">{esc(arm)}</text>'
        body += "</svg>"
    body += "<table><thead><tr><th>Scope</th><th>Epoch</th><th>Metric</th><th>Value</th><th>Unit</th><th>Status / limits</th><th>Source</th></tr></thead><tbody>"
    for r in selected:
        if r["kind"] == "scratch_context_pair":
            continue  # Full pairs remain in Parquet; the dashboard shows their cohort summaries.
        d = json.loads(r["details"])
        value = "unmeasured" if r["value"] is None else f"{r['value']:.8g}"
        cells = (
            r["scope"],
            d.get("epoch", "—"),
            r["metric"],
            value,
            r["unit"],
            r["status"] + "; " + r["note"],
            r["sources"],
        )
        body += (
            "<tr>" + "".join("<td>" + esc(str(x)) + "</td>" for x in cells) + "</tr>"
        )
    return body + "</tbody></table>"
