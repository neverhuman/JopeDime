"""Daily Parquet and a portable static D9 dashboard; no network resources."""

import html
import json
import math
from collections import defaultdict
from pathlib import Path


def write_daily(rows, output):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("date", pa.string()),
            ("kind", pa.string()),
            ("metric", pa.string()),
            ("scope", pa.string()),
            ("status", pa.string()),
            ("value", pa.float64()),
            ("n", pa.int64()),
            ("unit", pa.string()),
            ("note", pa.string()),
            ("sources", pa.string()),
            ("source_commit", pa.string()),
            ("context_set_sha256", pa.string()),
            ("details", pa.string()),
        ]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["date"]].append(row)
    files = []
    for date, records in sorted(grouped.items()):
        if len(date) != 10 or any(c not in "0123456789-" for c in date):
            raise ValueError("UTC YYYY-MM-DD partition required")
        path = output / (date + ".parquet")
        if path.exists():
            raise ValueError("daily partition already exists; use a new evidence run")
        pq.write_table(
            pa.Table.from_pylist(records, schema=schema), path, compression="zstd"
        )
        files.append(path)
    return files


def read_daily(files):
    import pyarrow.parquet as pq

    return [r for f in files for r in pq.read_table(f).to_pylist()]


def render(rows, output):
    """Render only values read back from Parquet, with explicit measurement status."""
    esc = html.escape
    summary = [r for r in rows if r["kind"] == "summary"]
    gates = [r for r in rows if r["kind"] == "gate"]

    def table(records):
        cells = []
        for r in records:
            value = "unmeasured" if r["value"] is None else f"{r['value']:.8g}"
            cells.append(
                "<tr>"
                + "".join(
                    "<td>" + esc(str(x)) + "</td>"
                    for x in (
                        r["date"],
                        r["metric"],
                        r["scope"],
                        value,
                        r["unit"],
                        r["status"],
                        r["n"],
                        r["note"],
                        r["sources"],
                    )
                )
                + "</tr>"
            )
        return (
            "<table><thead><tr>"
            + "".join(
                "<th>" + x + "</th>"
                for x in (
                    "UTC date",
                    "Metric",
                    "Scope",
                    "Value",
                    "Unit",
                    "Status",
                    "N",
                    "Evidence limit",
                    "Sources",
                )
            )
            + "</tr></thead><tbody>"
            + "".join(cells)
            + "</tbody></table>"
        )

    body = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>RL flywheel D9 measurements</title>'
    body += "<style>body{font:15px system-ui;background:#101923;color:#e2edf7;margin:2rem}h1,h2{color:#8ed7ca}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:.6rem;border:1px solid #395163;text-align:left;overflow-wrap:anywhere}th{background:#203448}td:last-child{max-width:25rem}p{max-width:80rem}</style>"
    body += "<h1>RL flywheel · D9</h1><p>Read-only retained observations. Missing values remain unmeasured. Cohort lower bounds and phase service rates do not qualify a live improving learner. No publication, deployment or promotion is performed.</p>"
    body += (
        "<h2>North stars, latency, paired noise, throughput and storage</h2>"
        + table(summary)
    )
    body += "<h2>Recorded gate history</h2>" + table(gates)
    operational = [r for r in rows if r["kind"] == "operational"]
    parents = {}
    for record in rows:
        if record["kind"] != "parent_cluster_summary":
            continue
        details = json.loads(record["details"])
        key = (
            record["scope"],
            record["context_set_sha256"],
            record["metric"],
            details["export_sha256"],
            details["control_arm"],
        )
        if (
            key not in parents
            or details["epoch"] > json.loads(parents[key]["details"])["epoch"]
        ):
            parents[key] = record
    if operational:
        body += (
            "<h2>Useful feedback, actor diagnostics and throughput denominators</h2>"
            + table(operational)
        )
    if parents:
        compact = []
        for key, record in parents.items():
            if record["metric"] != "parent_l6_paired_mean_delta":
                continue
            details = json.loads(record["details"])
            values = {}
            for metric in ("parent_l6_paired_mean_SE", "parent_l6_paired_variance"):
                value = parents[key[:2] + (metric,) + key[3:]]["value"]
                values[metric] = "unmeasured" if value is None else f"{value:.8g}"
            compact.append(
                record
                | {
                    "note": f"Epoch {details['epoch']}; updates {details['updates']}; contexts {details['contexts']}; parent mean SE {values['parent_l6_paired_mean_SE']}; parent variance {values['parent_l6_paired_variance']}."
                }
            )
        body += (
            "<h2>Paired inputs by parent lineage</h2><p>Latest matched cohort per export. Cuts are averaged within parent before equal-weight parent summaries. Epochs, update counts and rosters remain separate. These development inputs do not establish new-family generalization. All parent inputs and summaries remain in Parquet.</p>"
            + table(compact)
        )
    from .scratch import render_scratch

    body += render_scratch(rows)
    commits = sorted({r["source_commit"] for r in rows})
    body += (
        "<p>Source commit"
        + ("s" if len(commits) > 1 else "")
        + ": "
        + esc(", ".join(commits) if commits else "unmeasured")
        + "</p></html>"
    )
    Path(output).write_text(body + "\n")


def s3_daily(path):
    """Consume G-S3GP's local Parquet export only; no lister or pricing fallback.

    Daily export contract: date, bytes, usd, cost_kind, source. USD is an estimate
    when cost_kind=estimate; billing observations must be labelled billed.
    NULL dollars stay unmeasured even if bytes exist. Each row is a disjoint
    prefix/date subtotal, not an overlapping bucket+prefix roll-up.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    required = {"date", "bytes", "usd", "cost_kind", "source"}
    if not required <= set(table.column_names):
        raise ValueError("G-S3GP daily export schema missing columns")
    result = []
    for row in table.to_pylist():
        if (
            row["cost_kind"] not in ("estimate", "billed", "unmeasured")
            or row["bytes"] is not None
            and row["bytes"] < 0
            or row["usd"] is not None
            and (not math.isfinite(row["usd"]) or row["usd"] < 0)
        ):
            raise ValueError("invalid storage export")
        result.append(row)
    return result


def s3_census(path, receipt):
    """Consume the provider's actual role census and its reported monthly estimate.

    These bytes are a stock in one objects prefix, not daily additions. No
    production request-rate inference or CX-MET price calculation is permitted.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if not {"role", "count", "bytes", "storage_class"} <= set(table.column_names):
        raise ValueError("G-S3GP census schema missing columns")
    if receipt["schema"] != "jope.s3gp-dry-run/v1":
        raise ValueError("provider census receipt schema")
    rows, seen = table.to_pylist(), set()
    for r in rows:
        key = (r["role"], r["storage_class"])
        if key in seen or any(
            type(r[k]) is not int or r[k] < 0 for k in ("count", "bytes")
        ):
            raise ValueError("disjoint finite census subtotals required")
        seen.add(key)
    census = receipt["objects_prefix_census"]
    if (
        sum(r["bytes"] for r in rows) != census["bytes"]
        or sum(r["count"] for r in rows) != census["object_count"]
        or len(rows) != receipt["parquet_export"]["rows"]
    ):
        raise ValueError("census totals differ from provider receipt")
    return rows, census
