#!/usr/bin/env python3
"""Aggregate genome-boost final reports into per-backend summaries.

Reads every ``reports/<hash>/<backend>/final-report.v1.json`` under the
campaign root and writes CSV + JSON + a brief Markdown summary into
``runs/genome-boost-20260725-v1/report/`` (win-rate, mean/median delta and
delta_pct per backend). Everything summarized here is permanently
``test_adaptive: true`` / ``benchmark_valid: false``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CAMPAIGN_ID = "genome-boost-20260725-v1"
DEFAULT_CAMPAIGN_ROOT = Path("/home/ubuntu/remote_super/quant/campaigns") / CAMPAIGN_ID
DEFAULT_OUTPUT = Path("/home/ubuntu/jope-dime/runs") / CAMPAIGN_ID / "report"
SUMMARY_SCHEMA = "jope-dime.genome-boost-report.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        temporary = Path(tmp.name)
    os.replace(temporary, path)


def collect(campaign_root: Path) -> list[dict]:
    reports = []
    reports_root = campaign_root / "reports"
    if not reports_root.is_dir():
        raise SystemExit(f"no reports directory under {campaign_root}; run finalize first")
    for path in sorted(reports_root.glob("*/*/final-report.v1.json")):
        report = json.loads(path.read_text())
        if report.get("schema") != "jope-dime.genome-boost-final-report.v1":
            raise SystemExit(f"{path}: unexpected report schema")
        if report.get("benchmark_valid") is not False or report.get("test_adaptive") is not True:
            raise SystemExit(f"{path}: mislabeled report")
        reports.append(report)
    if not reports:
        raise SystemExit(f"no final reports found under {reports_root}")
    return reports


def backend_summary(rows: list[dict]) -> dict:
    deltas = [row["delta"] for row in rows]
    delta_pcts = [row["delta_pct"] for row in rows if row.get("delta_pct") is not None]
    wins = sum(1 for row in rows if row.get("boosted_wins"))
    leader_rows = [row for row in rows if "leader_rmse" in row]
    return {
        "cells": len(rows),
        "wins": wins,
        "win_rate": wins / len(rows),
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "mean_delta_pct": statistics.fmean(delta_pcts) if delta_pcts else None,
        "median_delta_pct": statistics.median(delta_pcts) if delta_pcts else None,
        "boosted_beats_leader": sum(1 for row in leader_rows if row.get("boosted_beats_leader")),
        "baseline_beats_leader": sum(1 for row in leader_rows if row.get("baseline_beats_leader")),
        "leader_comparisons": len(leader_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    reports = collect(arguments.campaign_root)

    by_backend: dict[str, list[dict]] = {}
    for report in reports:
        by_backend.setdefault(report["backend"], []).append(report)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "test_adaptive": True,
        "benchmark_valid": False,
        "created_at": utc_now(),
        "total_cells": len(reports),
        "backends": {backend: backend_summary(rows) for backend, rows in sorted(by_backend.items())},
        "overall": backend_summary(reports),
    }
    atomic_write_bytes(
        arguments.output / "summary.v1.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )

    csv_path = arguments.output / "report.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset_hash",
        "backend",
        "k_genomes",
        "baseline_rmse",
        "boosted_rmse",
        "delta",
        "delta_pct",
        "boosted_wins",
        "leader_rmse",
        "boosted_beats_leader",
        "benchmark_valid",
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", dir=csv_path.parent, prefix=f".{csv_path.name}.", delete=False, newline=""
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for report in sorted(reports, key=lambda row: (row["dataset_hash"], row["backend"])):
            writer.writerow({field: report.get(field, "") for field in fields})
        temporary = Path(tmp.name)
    os.replace(temporary, csv_path)

    lines = [
        f"# {CAMPAIGN_ID} report",
        "",
        "**test_adaptive: true — benchmark_valid: false.** Genome features were",
        "selected against each dataset's own test RMSE and train-side features",
        "are in-sample; nothing here is a held-out generalization claim.",
        "",
        f"Cells scored: {len(reports)} (dataset x backend, baseline vs boosted).",
        "",
        "| backend | cells | win rate | mean delta | median delta | mean delta % | median delta % |",
        "|---|---|---|---|---|---|---|",
    ]
    for backend, rows in sorted(by_backend.items()):
        stats = summary["backends"][backend]
        mean_pct = "-" if stats["mean_delta_pct"] is None else f"{stats['mean_delta_pct']:.3f}"
        median_pct = "-" if stats["median_delta_pct"] is None else f"{stats['median_delta_pct']:.3f}"
        lines.append(
            f"| {backend} | {stats['cells']} | {stats['win_rate']:.3f} | "
            f"{stats['mean_delta']:.6g} | {stats['median_delta']:.6g} | {mean_pct} | {median_pct} |"
        )
    overall = summary["overall"]
    lines += [
        "",
        f"Overall win rate: {overall['win_rate']:.3f} over {overall['cells']} cells; "
        f"boosted beats the test-adaptive leader in {overall['boosted_beats_leader']} of "
        f"{overall['leader_comparisons']} comparisons (baseline: {overall['baseline_beats_leader']}).",
        "",
        "delta = baseline_rmse - boosted_rmse (positive = boost helped).",
    ]
    atomic_write_bytes(arguments.output / "REPORT.md", ("\n".join(lines) + "\n").encode())
    print(f"report written to {arguments.output}")
    for backend, rows in sorted(by_backend.items()):
        stats = summary["backends"][backend]
        print(
            f"{backend}: {stats['cells']} cells, win rate {stats['win_rate']:.3f}, "
            f"median delta {stats['median_delta']:.6g}"
        )


if __name__ == "__main__":
    main()
