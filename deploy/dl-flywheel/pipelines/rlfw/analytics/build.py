"""Build D9 from retained BENCH artifacts. Run in a capped CPU-only scope."""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from continuous.consumer.dltrain.labels import epsilon as label_epsilon, robust_scale
from .extract import Copies, benchmark_slots, comparison_identity, digest, qualified
from .metrics import STAGES, macro, paired_noise, quantiles
from .report import read_daily, render, s3_census, s3_daily, write_daily
from .windows import measured_windows


def scope_evidence():
    path = next(
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    root = Path("/sys/fs/cgroup") / path.lstrip("/")
    quota, period = (root / "cpu.max").read_text().split()
    memory = (root / "memory.max").read_text().strip()
    if (
        quota == "max"
        or int(quota) / int(period) > 2
        or memory == "max"
        or int(memory) > 4 * 1024**3
    ):
        raise ValueError("scratch requires CPUQuota<=200% and MemoryMax<=4G")
    return {
        "path": path,
        "cpu.max": quota + " " + period,
        "memory.max": memory,
        "cpu.stat": (root / "cpu.stat").read_text(),
        "memory.peak": (root / "memory.peak").read_text().strip(),
    }


def build(
    bench,
    retained,
    copies_root,
    output,
    *,
    s3_export=None,
    s3_census_manifest=None,
    window_manifest=None,
):
    started = time.time()
    scope_start = scope_evidence()
    bench, retained, output = (
        Path(bench).resolve(),
        Path(retained).resolve(),
        Path(output).resolve(),
    )
    if (
        not retained.is_relative_to(bench / "runs")
        or output.is_relative_to(bench)
        or Path(copies_root).resolve().is_relative_to(bench)
    ):
        raise ValueError(
            "retained inputs under BENCH/runs; output outside BENCH required"
        )
    copies = Copies(copies_root)
    labels_pin = copies.copy(bench / "code_pin/consumer/dltrain/labels.py")
    labels_source = Path(__file__).parents[2] / "continuous/consumer/dltrain/labels.py"
    if digest(labels_pin) != digest(labels_source):
        raise ValueError(
            "epsilon algorithm differs from benchmark's pinned labels source"
        )
    roster_path = bench / "cache/D5-kept-core53.json"
    roster = copies.json(roster_path)["contexts"]
    context_sha = digest(roster_path)
    frozen = copies.json(bench / "cache/D5-hwm-cache.json")
    frozen = {
        (e["context_id"], e["program_sha256"]): e
        for e in frozen.values()
        if qualified(e)
    }
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
    ).strip()
    analysis_path = retained / "analysis.json"
    analysis = copies.json(analysis_path)
    date = (
        dt.datetime.fromtimestamp(float(analysis["at"]), dt.timezone.utc)
        .date()
        .isoformat()
    )
    output.mkdir(parents=True, exist_ok=True)
    observations = output / "observations.jsonl"
    if observations.exists():
        raise ValueError("use a new output directory for each immutable run")
    rows, noise, phase, candidates = [], [], defaultdict(list), {}
    observed_at = float(analysis["at"])

    def add(
        metric,
        value,
        *,
        n=0,
        scope="fixed-D5-core53",
        unit="",
        status=None,
        note="",
        sources=None,
        kind="summary",
        details=None,
    ):
        row = {
            "date": date,
            "kind": kind,
            "metric": metric,
            "scope": scope,
            "status": status or ("unmeasured" if value is None else "measured"),
            "value": value,
            "n": n,
            "unit": unit,
            "note": note,
            "sources": json.dumps(sources or [str(analysis_path) + ":1"]),
            "source_commit": source_commit,
            "context_set_sha256": context_sha,
            "details": json.dumps(details or {}, sort_keys=True, allow_nan=False),
        }
        rows.append(row)

    # No dated comparable ledger or lifetime novelty/exposure proof was exported
    # with these arena snapshots. Their absence must not become a zero-rate claim.
    add(
        "U_useful",
        None,
        unit="new programs/wall-s/parent",
        note="No lifetime first-seen program inventory + frozen rolling window closure.",
    )
    add(
        "G_per_core_hour",
        None,
        unit="improvements/core-h/parent",
        note="No dated research_hwm events joined to evaluator cgroup CPU seconds. Process CPU is not substituted.",
    )
    add(
        "delta_H",
        None,
        unit="micro-q/1e6",
        note="Fixed roster includes censored/unresolved contexts; positive observed gains are only lower bounds.",
    )
    add(
        "T_feedback",
        None,
        unit="seconds",
        note="No proposal-commit to first physically committed learner-exposure join in retained D5 artifacts.",
    )
    for a, b in zip(STAGES, STAGES[1:]):
        add(
            "feedback:" + a + "→" + b,
            None,
            unit="seconds",
            note="No identity-bound receipts for both endpoints.",
        )

    with observations.open("x") as stream:
        for path in sorted((retained / "inputs").glob("*-outbox.sqlite")):
            slots, closed = benchmark_slots(copies, path)
            game = path.name.removesuffix("-outbox.sqlite")
            censored = sum(
                s["payload"].get("censor_reason") is not None
                or s["payload"]["outcome"] in ("infra_unknown", "parity_custody")
                for s in slots
            )
            add(
                "censoring_rate",
                censored / len(slots) if slots else None,
                n=len(slots),
                scope=game,
                unit="fraction of logical slots",
                details={"censored": censored},
                sources=[str(path) + ":slots(rowid)"],
                note="Logical slots include replay/cache; unknown/parity custody kept nonnumeric. Slot rate is distinct from lineage exclusions.",
            )
            for s in slots:
                p, e = s["payload"], s["evidence"]
                # Explicit extracted rows provide file:line evidence for SQLite data.
                stream.write(
                    json.dumps(
                        {
                            "db": str(path),
                            "rowid": s["rowid"],
                            "payload": p,
                            "evidence": e,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                if p["arm"] == "hwm" or not qualified(e):
                    continue
                identity = comparison_identity(e)
                if identity is None:
                    continue
                key = (identity, e["program_sha256"])
                if key in candidates:
                    if candidates[key]["q_grid"] != e["q_grid"]:
                        raise ValueError(
                            "same comparable program has conflicting measured q"
                        )
                    continue
                candidates[key] = e
                if not e.get("cache_hit"):
                    for stage, seconds in e.get("phase_seconds", {}).items():
                        if seconds < 0:
                            raise ValueError("negative phase duration")
                        phase[stage].append(seconds)
            if game in analysis["games"]:
                g = analysis["games"][game]
                throughput = g["achieved_throughput"]
                taxonomy = g["physical_response_taxonomy"]
                unknown = g["physical_native_requests"] - g["physical_native_responses"]
                physical_censored = unknown + sum(
                    taxonomy.get(k, 0) for k in ("parity_custody", "infra_unknown")
                )
                add(
                    "physical_censoring_rate",
                    physical_censored / g["physical_native_requests"],
                    scope=game,
                    n=g["physical_native_requests"],
                    unit="fraction of requests",
                    note="Parity/infra custody plus missing responses; excluded work remains in the denominator.",
                    details={
                        "taxonomy": taxonomy,
                        "missing_responses": unknown,
                        "censored": physical_censored,
                    },
                )
                add(
                    "producer_responses_per_second",
                    throughput["physical_native_responses_per_hour"] / 3600,
                    scope=game,
                    n=g["physical_native_responses"],
                    unit="responses/wall-s",
                    note="Retained physical audit includes failed/excluded work; imported work is excluded. Producer state="
                    + g["completion"]["state"],
                    details={
                        "requests": g["physical_native_requests"],
                        "elapsed_seconds": throughput["elapsed_producer_seconds"],
                        "source_commit": g["completion"]["source_commit"],
                    },
                )

    parents = {c["lineage_id"] for c in roster}
    lower_bounds, observed_useful = [], defaultdict(set)
    for c in roster:
        old = frozen.get((c["context_id"], c["hwm_program_sha256"]))
        if old is None:
            lower_bounds.append((c["lineage_id"], None))
            continue
        best = old["q_grid"]
        for (identity, program), new in candidates.items():
            if identity != comparison_identity(old):
                continue
            best = max(best, new["q_grid"])
            if new["q_grid"] > old["q_grid"]:
                observed_useful[c["lineage_id"]].add(program)
            old_eval = copies.artifact(old["artifacts"]["evaluation"], bench)
            new_eval = copies.artifact(new["artifacts"]["evaluation"], bench)
            targets = [y for fold in old_eval["inner_cv"] for y in fold["targets"]]
            epsilon = label_epsilon(targets, robust_scale(targets))
            seed = int(
                hashlib.sha256((c["context_id"] + program).encode()).hexdigest()[:16],
                16,
            )
            measure = paired_noise(
                old_eval["inner_cv"], new_eval["inner_cv"], epsilon=epsilon, seed=seed
            )
            noise.append(measure)
            add(
                "paired_noise",
                measure["SE_pair"],
                scope=c["context_id"] + ":" + program,
                n=2000,
                kind="pair",
                unit="log RMSE ratio SE",
                sources=[
                    old["artifacts"]["evaluation"]["path"] + ":1",
                    new["artifacts"]["evaluation"]["path"] + ":1",
                ],
                details=measure,
                note="Ordered inner-fold targets checked; paired resampling within each fold. Exploratory two-fold uncertainty, no teacher promotion.",
            )
        lower_bounds.append((c["lineage_id"], (best - old["q_grid"]) / 1e6))
    add(
        "delta_H_observed_lower_bound",
        macro(lower_bounds, parents),
        n=len(roster),
        unit="q",
        status="lower_bound",
        sources=[str(roster_path) + ":1", str(observations) + ":1"],
        note="Frozen core53, parent macro. Unobserved candidate outcomes cannot lower the observed maximum; missing comparator would make bound null.",
    )
    add(
        "observed_distinct_useful_programs",
        sum(len(v) for v in observed_useful.values()),
        n=len(candidates),
        unit="program-parent pairs",
        note="Observed verified programs above frozen comparator; lifetime novelty and per-second U_useful remain unmeasured.",
        sources=[str(observations) + ":1"],
    )
    for metric in ("SE_pair", "abs_d0_minus_d1", "bootstrap_se", "tau_c"):
        q = quantiles([n[metric] for n in noise])
        for percentile in ("p50", "p95", "p99"):
            add(
                metric + ":" + percentile,
                q[percentile],
                n=q["n"],
                unit="log RMSE ratio",
                note="Observed comparable pairs; selective retained cohort, no full-corpus claim.",
                sources=[str(output / "daily" / (date + ".parquet")) + ":pair rows"],
            )
    for stage, seconds in sorted(phase.items()):
        for percentile, value in quantiles(seconds).items():
            if percentile != "n":
                add(
                    "phase:" + percentile,
                    value,
                    n=len(seconds),
                    scope=stage,
                    unit="seconds",
                    note="Noncached unique comparable results; nested phases are reported separately, never summed.",
                    sources=[str(observations) + ":1"],
                )
        add(
            "phase_service_rate",
            len(seconds) / sum(seconds) if sum(seconds) else None,
            n=len(seconds),
            scope=stage,
            unit="samples/service-s",
            note="Serial service-time denominator; not concurrent wall throughput.",
            sources=[str(observations) + ":1"],
        )
    for path in sorted((bench / "runs/d5b").glob("analysis-*/analysis.json")):
        a = copies.json(path)
        verdict = (a.get("paired") or {}).get("verdict")
        if verdict:
            add(
                "arena_gate",
                None,
                status="recorded:" + verdict,
                kind="gate",
                scope=path.parent.name,
                note="Retained directional Lane A verdict; not a G2/G3 promotion. numeric_campaign_REPORT_allowed="
                + str(a.get("numeric_campaign_REPORT_allowed")),
                sources=[str(path) + ":1"],
                details={"at": a.get("at"), "source_commit": a.get("source_commit")},
            )
    if s3_export:
        exported = s3_daily(copies.copy(s3_export))
        for s in exported:
            for metric, value, unit in (
                ("S3_bytes", s["bytes"], "bytes"),
                ("S3_USD", s["usd"], "USD/day"),
            ):
                add(
                    metric,
                    value,
                    scope=s["source"],
                    unit=unit,
                    status=s["cost_kind"] if value is not None else "unmeasured",
                    sources=[str(s3_export) + ":Parquet row"],
                )
                rows[-1]["date"] = s["date"]
    else:
        add(
            "S3_bytes",
            None,
            unit="bytes",
            note="Await G-S3GP local ledger export; no independent S3 lister.",
        )
        add(
            "S3_USD",
            None,
            unit="USD/day",
            note="Await G-S3GP cost export; no invented billing or price assumptions.",
        )
    if s3_census_manifest:
        m = copies.json(s3_census_manifest)
        if m["schema"] != "jope.rlfw-s3-census-input/v1":
            raise ValueError("census manifest schema")

        def pin(ref):
            return copies.copy(
                ref["path"], expected_sha=ref["sha256"], expected_bytes=ref["bytes"]
            )

        census_path = pin(m["parquet"])
        provider = json.loads(pin(m["receipt"]).read_text())
        if digest(census_path) != provider["parquet_export"]["parquet_sha256"]:
            raise ValueError("provider Parquet identity mismatch")
        ledger = [
            json.loads(line) for line in pin(m["ledger"]).read_text().splitlines()
        ]
        event = next(
            e for e in ledger if e["sha256"] == provider["ledger_event_sha256"]
        )
        census_rows, census = s3_census(census_path, provider)
        sources = [
            m["parquet"]["path"] + ":Parquet rows 1-7",
            m["receipt"]["path"] + ":1",
        ]
        for metric, value, unit, status in (
            ("S3_census_bytes", census["bytes"], "bytes observed", "measured"),
            (
                "S3_census_objects",
                census["object_count"],
                "objects observed",
                "measured",
            ),
            (
                "S3_census_monthly_USD",
                census.get("standard_usd_per_month"),
                "USD/month",
                "provider-estimate",
            ),
        ):
            add(
                metric,
                value,
                n=len(census_rows),
                scope=m["scope"],
                unit=unit,
                status=status,
                note="Provider census of one objects prefix; not production bytes added/day or billed costs. Truncated="
                + str(census["truncated"]),
                sources=sources,
                details={
                    "provider_commit": m["provider_commit"],
                    "roles": census_rows,
                    "provider_ledger_day": event["day"],
                },
            )
            rows[-1]["date"] = event["day"]
    if window_manifest:
        rows.extend(
            measured_windows(
                copies, copies.json(window_manifest), source_commit=source_commit
            )
        )
    files = write_daily(rows, output / "daily")
    readback = read_daily(files)
    if readback != sorted(rows, key=lambda r: r["date"]):
        raise ValueError("Parquet roundtrip mismatch")
    render(readback, output / "dashboard.html")
    receipt = {
        "schema": "jope.rlfw-analytics/v1",
        "source_commit": source_commit,
        "source_sha256": {
            str(p.relative_to(Path(__file__).parent)): digest(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "command": sys.argv,
        "exit_status": 0,
        "started_unix": started,
        "finished_unix": time.time(),
        "observed_until_unix": observed_at,
        "context_set_sha256": context_sha,
        "rows": len(rows),
        "paired_measurements": len(noise),
        "epsilon_rule": "pinned dltrain.labels.epsilon from both saved inner score folds (fitting rows only)",
        "epsilon_source_sha256": digest(labels_source),
        "bootstrap_draws": 2000,
        "copies": list(copies.records.values()),
        "parquet": [{"path": str(f), "sha256": digest(f)} for f in files],
        "compute_limit": {"CPUQuota": "200%", "MemoryMax": "4G", "GPU": False},
        "scope_start": scope_start,
        "scope_finish": scope_evidence(),
        "scope": "retained Lane A observations; no live-loop qualification",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bench", "retained", "copies", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--s3-export", type=Path)
    p.add_argument(
        "--s3-census-manifest",
        type=Path,
        help="Pinned G-S3GP role census/receipt/ledger exports",
    )
    p.add_argument(
        "--window-manifest",
        type=Path,
        help="Pinned owner-exported ledger/UTC/cgroup/exposure joins",
    )
    a = p.parse_args()
    receipt = build(
        a.bench,
        a.retained,
        a.copies,
        a.output,
        s3_export=a.s3_export,
        s3_census_manifest=a.s3_census_manifest,
        window_manifest=a.window_manifest,
    )
    print(
        json.dumps(
            {k: receipt[k] for k in ("rows", "paired_measurements", "source_commit")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
