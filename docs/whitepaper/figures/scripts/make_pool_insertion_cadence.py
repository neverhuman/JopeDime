#!/usr/bin/env python3
"""Plot consecutive pool-insertion timestamp gaps and optionally verify the source window."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp_gap(previous: str, current: str) -> Decimal:
    start, end = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in [previous, current]]
    assert start.tzinfo is not None and end.tzinfo is not None
    delta = end - start
    microseconds = (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds
    return Decimal(microseconds) / Decimal(1000000)


def main() -> None:
    figures = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=figures / "data/pool_insertion_cadence.csv")
    parser.add_argument("--output", type=Path, default=figures / "fig_pool_insertion_cadence.pdf")
    parser.add_argument("--provenance", type=Path, default=figures / "data/pool_insertion_cadence.provenance.json")
    parser.add_argument("--verify-source-window", type=Path)
    args = parser.parse_args()
    source_path = args.data.with_suffix(".source.json")
    source = json.loads(source_path.read_text())
    data = args.data.read_bytes()
    assert sha(data) == source["csv_sha256"]
    rows = list(csv.DictReader(data.decode().splitlines()))
    assert len(rows) == source["intervals"] == 100
    assert [int(row["interval_index"]) for row in rows] == list(range(1, 101))
    for i, row in enumerate(rows):
        gap = timestamp_gap(row["previous_timestamp_utc"], row["timestamp_utc"])
        assert Decimal(row["gap_seconds"]) == gap and gap >= 0
        assert int(row["source_line"]) == int(row["previous_source_line"]) + 1
        if i:
            assert row["previous_source_line"] == rows[i - 1]["source_line"]
            assert row["previous_timestamp_utc"] == rows[i - 1]["timestamp_utc"]
            assert row["previous_record_sha256"] == rows[i - 1]["source_record_sha256"]
    if args.verify_source_window:
        raw = args.verify_source_window.read_bytes()
        assert sha(raw) == source["window_sha256"] and len(raw) == source["window_bytes"]
        lines = raw.splitlines(keepends=True)
        assert len(lines) == source["source_records"] == 101
        records = [json.loads(line) for line in lines]
        assert all(r["kind"] == "promotion" and r["cv_gate_passed"] is True for r in records)
        assert [records[0]["round"], records[-1]["round"]] == source["round_window"]
        for i, row in enumerate(rows):
            assert int(row["previous_source_line"]) == source["window_first_source_line"] + i
            assert int(row["source_line"]) == source["window_first_source_line"] + i + 1
            assert row["previous_record_sha256"] == sha(lines[i])
            assert row["source_record_sha256"] == sha(lines[i + 1])
            for prefix, record in [("previous_", records[i]), ("", records[i + 1])]:
                assert int(row[prefix + "round"]) == record["round"]
                assert row[prefix + "timestamp_utc"] == record["timestamp"]
    gaps = [Decimal(row["gap_seconds"]) for row in rows]
    assert sum(g == 0 for g in gaps) == source["zero_gaps"]
    assert [format(min(gaps), "f"), format(max(gaps), "f")] == source["gap_range_seconds"]
    font = Path(font_manager.findfont(font_manager.FontProperties(family="Liberation Serif"), fallback_to_default=False))
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.23, top=0.94)
    ax.scatter([int(row["interval_index"]) for row in rows], [float(g) for g in gaps],
               s=8, color="#17649a", linewidths=0)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=-0.08)
    ax.set_xlabel("Consecutive insertion interval")
    ax.set_ylabel("Timestamp gap (s)")
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(FixedLocator([0, 1, 10, 100, 1000]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(axis="y", color="0.88", linewidth=0.4)
    ax.set_axisbelow(True)
    captured = datetime.fromisoformat(source["captured_at"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Title": "Logged pool-insertion cadence", "Author": "",
                "Subject": "Engineering event timestamp differences", "CreationDate": captured, "ModDate": captured})
    plt.close(fig)
    provenance = {
        "schema": "whitepaper.pool-insertion-cadence-figure/v1", "input_csv_sha256": sha(data),
        "source_receipt_sha256": sha(source_path.read_bytes()), "generator_sha256": sha(Path(__file__).read_bytes()),
        "figure_sha256": sha(args.output.read_bytes()), "source_records": 101, "plotted_intervals": 100,
        "round_window": source["round_window"], "units": "Seconds between adjacent logged UTC timestamps; horizontal coordinate is interval ordinal.",
        "transformations": "Exact timestamp differences, no rounding, resampling, duration cutoff, smoothing or outcome selection. Scatter only; symmetric-log axis linear within1second, retaining zero if observed.",
        "source_trace": "Every point maps to two consecutive source lines and their newline-inclusive record SHA256 hashes; --verify-source-window checks all source records and derived differences.",
        "scope": "Training-pool program/context insertion timestamps, distinct from adapter admission, proposal runtime, independent learning gain or a speed comparison.",
        "matplotlib_version": matplotlib.__version__, "font_family": "Liberation Serif", "font_file_sha256": sha(font.read_bytes()),
        "figure_inches": [3.25, 2.05], "minimum_text_size_pt": 8.5, "fixed_canvas_no_tight_crop": True,
        "reproduce": "python3 figures/scripts/make_pool_insertion_cadence.py",
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"intervals": 100, "records": 101, "round_window": source["round_window"],
                      "figure_sha256": provenance["figure_sha256"], "source_window_verified": bool(args.verify_source_window)}))


if __name__ == "__main__":
    main()
