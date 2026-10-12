"""Consume owner-exported receipt joins for rolling 1 h / 24 h measurements."""

import datetime as dt
import json
import statistics
from collections import defaultdict

from .extract import ledger_events
from .metrics import STAGES, feedback_latencies, macro, north_star, quantiles


def measured_windows(copies, manifest, *, source_commit):
    """No clock inference from file mtime or SQLite commit order.

    Manifest pins JSON contexts/timeline/feedback and the research_hwm DB by
    {path,sha256,bytes}. timeline joins exact (hwm_key,version) publications to
    retained UTC receipts; first_seen requires the exporter's lifetime census.
    CPU receipts bind the same window and per-parent evaluator cgroup seconds.
    Unknown fixed-context coverage stays unresolved.
    """
    if manifest["schema"] != "jope.rlfw-measurement-input/v1":
        raise ValueError("measurement manifest schema")

    def load(ref):
        return copies.json(
            ref["path"], expected_sha=ref["sha256"], expected_bytes=ref["bytes"]
        )

    contexts = load(manifest["contexts"])
    timeline = load(manifest["timeline"])
    ledger_ref = manifest["ledger"]
    copies.copy(
        ledger_ref["path"],
        expected_sha=ledger_ref["sha256"],
        expected_bytes=ledger_ref["bytes"],
    )
    ledger = {
        (r["hwm_key"], r["version"]): r
        for r in ledger_events(copies, ledger_ref["path"])
    }
    events, seen = [], set()
    for t in timeline:
        key = (t["hwm_key"], t["version"])
        if key in seen or key not in ledger:
            raise ValueError("unique existing ledger version timestamp join required")
        seen.add(key)
        row = ledger[key]
        publication = row["publication"]
        if (
            publication.get("hwm_key") != row["hwm_key"]
            or publication.get("source") != "measured"
        ):
            raise ValueError("ledger publication comparison/source mismatch")
        events.append(
            {
                **t,
                "program_sha256": publication["program_sha256"],
                "q_grid": publication["q_grid"],
                "verified": row["event"] != "refused"
                and publication["outcome"] == "qualified_complete",
                "ledger_event": row["event"],
            }
        )
    feedback = load(manifest["feedback"]) if manifest.get("feedback") else []
    rows = []
    for w in manifest["windows"]:
        if w["end"] - w["start"] not in (3600, 86400):
            raise ValueError("registered rolling 1 h or 24 h window required")
        cpu = load(w["cpu"]) if w.get("cpu") else None
        if cpu is not None and (
            cpu["schema"] != "jope.evaluator-cgroup-window/v1"
            or cpu["start"] != w["start"]
            or cpu["end"] != w["end"]
        ):
            raise ValueError("CPU receipt must bind exact evaluator window")
        # An undated qualified publication may fall inside or outside this
        # window. Neither a zero rate nor a complete gain can be established.
        fixed_keys = {c["hwm_key"] for c in contexts}
        complete_times = all(
            key in seen
            for key, r in ledger.items()
            if r["hwm_key"] in fixed_keys
            and r["event"] != "refused"
            and r["publication"].get("source") == "measured"
            and r["publication"].get("outcome") == "qualified_complete"
        )
        metrics = north_star(
            events,
            contexts,
            start=w["start"],
            end=w["end"],
            cpu_seconds=cpu["parents"] if cpu else None,
            cpu_source="evaluator_cgroup" if cpu and complete_times else None,
        )
        if not complete_times:
            for name in ("U_useful", "G_per_core_hour", "delta_H"):
                metrics[name] = None
        context_parents = {c["context_id"]: c["parent"] for c in contexts}
        latency = [
            r
            for r in feedback_latencies(feedback, observed_until=w["end"])
            if r["proposal_at"] is not None
            and w["start"] <= r["proposal_at"] < w["end"]
            and r["context_id"] in context_parents
            and r["parent"] == context_parents[r["context_id"]]
        ]
        times = [r["T_feedback"] for r in latency if r["T_feedback"] is not None]
        # Completed-only latency is selective; the north star needs complete
        # exposure joins and equal weight per fixed context, then per parent.
        by_context = defaultdict(list)
        for r in latency:
            if (
                r["context_id"] in context_parents
                and r["parent"] == context_parents[r["context_id"]]
            ):
                by_context[r["context_id"]].append(r["T_feedback"])
        metrics["T_feedback"] = macro(
            [
                (
                    c["parent"],
                    statistics.mean(by_context[c["context_id"]])
                    if by_context[c["context_id"]]
                    and all(x is not None for x in by_context[c["context_id"]])
                    else None,
                )
                for c in contexts
            ],
            sorted(set(context_parents.values())),
        )
        metrics["T_feedback:p50"] = quantiles(times)["p50"]
        metrics["feedback_right_censored"] = (
            sum(r["censored"] for r in latency) if latency else None
        )
        names = [
            "U_useful",
            "G_per_core_hour",
            "delta_H",
            "T_feedback",
            "T_feedback:p50",
            "feedback_right_censored",
        ]
        for a, b in zip(STAGES, STAGES[1:]):
            stage = a + "→" + b
            for q, value in quantiles(
                [
                    r["stage_seconds"][stage]
                    for r in latency
                    if r["stage_seconds"][stage] is not None
                ]
            ).items():
                if q != "n":
                    name = "feedback:" + stage + ":" + q
                    metrics[name] = value
                    names.append(name)
        for metric in names:
            value = metrics[metric]
            rows.append(
                {
                    "date": dt.datetime.fromtimestamp(w["end"], dt.timezone.utc)
                    .date()
                    .isoformat(),
                    "kind": "summary",
                    "metric": metric,
                    "scope": w["name"],
                    "status": "measured" if value is not None else "unmeasured",
                    "value": value,
                    "n": len(contexts),
                    "unit": "see METRICS.md",
                    "note": "Exact-version receipts; proposal cohort in half-open window. U/G/ΔH/T parent macro requires fixed closure; latency quantiles are completed-only with censor count.",
                    "sources": json.dumps(
                        [
                            ref["path"] + ":1"
                            for ref in (
                                manifest["contexts"],
                                manifest["timeline"],
                                ledger_ref,
                            )
                        ]
                    ),
                    "source_commit": source_commit,
                    "context_set_sha256": manifest["contexts"]["sha256"],
                    "details": json.dumps(
                        {"window": w, "latencies": latency}, sort_keys=True
                    ),
                }
            )
    return rows
