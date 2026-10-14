"""Emit baseline Parquet, HTML and Markdown from pinned local owner exports."""

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess

from .baseline import DOMAINS, baseline_rows, interval
from .extract import Copies, digest
from .report import read_daily, render, write_daily


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(
        Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid
    )


def copied_ref(copies, ref):
    if (
        not Path(ref["path"]).is_absolute()
        or len(ref["sha256"]) != 64
        or type(ref["bytes"]) is not int
    ):
        raise ValueError("absolute path, SHA256 and byte length required")
    return copies.copy(
        ref["path"], expected_sha=ref["sha256"], expected_bytes=ref["bytes"]
    )


def markdown(rows, output):
    """Like HTML, all displayed numbers come from the actual Parquet readback."""
    lines = [
        "# GB-08 baseline observations",
        "",
        "NON-QUALIFYING readback. L6 is a loss; lower is better. No new evaluation or qualification.",
        "",
        "| Metric | Scope | Value | Unit | Status | Evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in rows:
        value = (
            "unmeasured" if record["value"] is None else format(record["value"], ".12g")
        )
        cells = [
            record["metric"],
            record["scope"],
            value,
            record["unit"],
            record["status"],
            record["sources"],
        ]
        lines.append(
            "| "
            + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "Analysis commit: " + rows[0]["source_commit"],
            "",
            "Every row retains its exact window, producer commit (nullable), coverage, clock/denominator details and missing-proof reason in Parquet. Completed latency quantiles describe completed observations only. Stock bytes are not daily additions; estimates are not bills.",
        ]
    )
    Path(output).write_text("\n".join(lines) + "\n")


def build(manifest, output, *, commit, source_dirty=False, scope=None):
    """No S3/SQLite adapter: only immutable copied JSON and provider Parquet."""
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("use a new immutable output directory")
    output.mkdir(parents=True)
    copies = Copies(output / "inputs")
    manifest_copy = copies.copy(manifest)
    spec = read_json(manifest_copy)
    if spec.get("schema") != "jope.rlfw-baseline-manifest/v1":
        raise ValueError("baseline manifest schema")
    date = dt.date.fromisoformat(spec["date"])
    start, end = interval(spec["window"])
    midnight = dt.datetime.combine(date, dt.time(), dt.timezone.utc).timestamp()
    if spec.get("window_kind") == "utc_day":
        if start != midnight or end != midnight + 86400:
            raise ValueError("UTC-day window must be midnight to midnight")
    elif spec.get("window_kind") != "whole_run":
        raise ValueError("explicit utc_day or whole_run window required")
    exports = {}
    sources = dict.fromkeys(DOMAINS, f"{manifest}:1 sha256:{digest(manifest_copy)}")
    sources["manifest"] = f"{manifest}:1 sha256:{digest(manifest_copy)}"
    if set(spec.get("exports", {})) - set(DOMAINS):
        raise ValueError("unknown export domain")
    for domain, ref in spec.get("exports", {}).items():
        path = copied_ref(copies, ref)
        export = read_json(path)
        if "ledger" in export:
            if domain != "cost" or "records" in export.get("data", {}):
                raise ValueError("ledger is exclusive to owner cost exports")
            import pyarrow.parquet as pq

            ledger_path = copied_ref(copies, export["ledger"])
            export.setdefault("data", {})["records"] = pq.read_table(
                ledger_path
            ).to_pylist()
        exports[domain] = export
        sources[domain] = f"{ref['path']}:1 sha256:{ref['sha256']}"
    rows = baseline_rows(
        exports,
        window=spec["window"],
        date=date.isoformat(),
        commit=commit,
        sources=sources,
    )
    for record in rows:
        details = json.loads(record["details"])
        details.update(
            {
                "window_kind": spec["window_kind"],
                "manifest_sha256": digest(manifest_copy),
                "synthetic": spec.get("synthetic") is True,
            }
        )
        record["details"] = json.dumps(details, sort_keys=True, allow_nan=False)
    files = write_daily(rows, output / "daily")
    readback = read_daily(files)
    if readback != rows or any(
        r["value"] is not None and not math.isfinite(r["value"]) for r in readback
    ):
        raise ValueError("Parquet readback differs")
    render(readback, output / "dashboard.html")
    markdown(readback, output / "METRICS.md")
    inputs = output / "INPUTS.json"
    inputs.write_text(json.dumps(list(copies.records.values()), indent=2) + "\n")
    artifacts = [*files, output / "dashboard.html", output / "METRICS.md", inputs]
    receipt = {
        "schema": "jope.rlfw-baseline-build/v1",
        "source_commit": commit,
        "source_dirty": source_dirty,
        "rows": len(readback),
        "null_rows": sum(r["value"] is None for r in readback),
        "files": [
            {
                "path": str(p.relative_to(output)),
                "sha256": digest(p),
                "bytes": p.stat().st_size,
            }
            for p in artifacts
        ],
        "inputs": list(copies.records.values()),
        "window": spec["window"],
        "window_kind": spec["window_kind"],
        "scope": scope,
        "non_qualifying": True,
        "synthetic": spec.get("synthetic") is True,
        "S3_calls": 0,
        "training_execution": False,
        "live_DB_handles": 0,
        "passed": True,
        "source_review_or_hosted_CI": False,
    }
    # Last artifact is the completion marker; a failed build is never complete.
    with (output / "COMPLETED.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reuse the existing CPU-only scratch admission contract, including 4 GiB.
    from .build import scope_evidence

    scope = scope_evidence()
    work = Path(__file__).parent
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=work, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=work, text=True
        ).strip()
    )
    result = build(
        args.manifest, args.output, commit=commit, source_dirty=dirty, scope=scope
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "source_commit",
                    "rows",
                    "null_rows",
                    "passed",
                    "non_qualifying",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
