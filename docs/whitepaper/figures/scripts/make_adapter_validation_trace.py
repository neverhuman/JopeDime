#!/usr/bin/env python3
"""Render logged adapter decisions; optionally verify every CSV row against a journal."""
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
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    figures = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=figures / "data/adapter_validation_trace.csv")
    parser.add_argument("--output", type=Path, default=figures / "fig_adapter_validation_trace.pdf")
    parser.add_argument("--provenance", type=Path, default=figures / "data/adapter_validation_trace.provenance.json")
    parser.add_argument("--verify-journal", type=Path)
    args = parser.parse_args()
    source_path = args.data.with_suffix(".source.json")
    source = json.loads(source_path.read_text())
    data = args.data.read_bytes()
    assert sha(data) == source["csv_sha256"]
    rows = list(csv.DictReader(data.decode().splitlines()))
    assert len(rows) == source["rows"] == 200
    rounds = [int(row["round"]) for row in rows]
    assert rounds == list(range(source["round_window"][0], source["round_window"][1] + 1))
    columns = ["candidate_validation_utility", "prior_retained_validation_utility", "retained_validation_utility"]
    for i, row in enumerate(rows):
        candidate, prior, retained = [Decimal(row[key]) for key in columns]
        assert all(value.is_finite() for value in [candidate, prior, retained])
        assert row["accepted"] in {"0", "1"}
        assert retained == (candidate if row["accepted"] == "1" else prior)
        if i:
            assert prior == Decimal(rows[i - 1]["retained_validation_utility"])
    assert sum(int(row["accepted"]) for row in rows) == source["admissions"]
    if args.verify_journal:
        with args.verify_journal.open("rb") as stream:
            prefix = stream.read(source["complete_prefix_bytes"])
        assert sha(prefix) == source["complete_prefix_sha256"]
        lines = prefix.splitlines(keepends=True)
        assert len(lines) == source["complete_source_records"]
        assert [int(row["source_line"]) for row in rows] == list(range(len(lines) - 199, len(lines) + 1))
        mapping = dict(zip(columns, ["candidate_utility", "previous_best_utility", "best_utility"]))
        for row in rows:
            line = lines[int(row["source_line"]) - 1]
            assert sha(line) == row["source_record_sha256"]
            raw = json.loads(line, parse_float=Decimal)
            assert raw["round"] == int(row["round"])
            assert int(raw["accepted"]) == int(row["accepted"])
            for projected, original in mapping.items():
                assert row[projected] == str(raw[original])
    font = Path(font_manager.findfont(font_manager.FontProperties(family="Liberation Serif"), fallback_to_default=False))
    plt.rcParams.update({
        "font.family": "Liberation Serif", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.23, top=0.82)
    ax.plot(rounds, [float(row[columns[0]]) for row in rows], color="0.55", linewidth=0.55,
            marker=".", markersize=1.7, label="Candidate")
    ax.step(rounds, [float(row[columns[2]]) for row in rows], where="post", color="#17649a",
            linewidth=1.2, label="Retained")
    ax.set_xlabel("Adapter decision round")
    ax.set_ylabel("Logged validation utility")
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.margins(x=0.015, y=0.08)
    ax.grid(axis="y", color="0.88", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=2, frameon=False,
              handlelength=1.5, columnspacing=1.2, borderaxespad=0)
    captured = datetime.fromisoformat(source["captured_at"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Title": "Logged adapter validation utility",
                "Author": "", "Subject": "Adaptive engineering telemetry",
                "CreationDate": captured, "ModDate": captured})
    plt.close(fig)
    provenance = {
        "schema": "whitepaper.adapter-validation-trace-figure/v1",
        "input_csv_sha256": sha(data), "source_receipt_sha256": sha(source_path.read_bytes()),
        "generator_sha256": sha(Path(__file__).read_bytes()), "figure_sha256": sha(args.output.read_bytes()),
        "rows": len(rows), "round_window": source["round_window"], "admissions": source["admissions"],
        "candidate_values": "CSV candidate_validation_utility: exact logged candidate_utility",
        "retained_values": "CSV retained_validation_utility: exact logged post-decision best_utility, not a cumulative maximum",
        "source_trace": "Each CSV row records the original journal line and exact newline-inclusive record SHA256; --verify-journal checks every value.",
        "transformations": "Every decision retained. Lines connect consecutive candidate observations; retained values use a post-decision step. No smoothing, resampling, clipping, cumulative maximum or outcome selection.",
        "units": "Logged dimensionless validation-pool utility; larger values preferred. Horizontal coordinate is the journaled decision round.",
        "scope": "Adaptive development-side engineering telemetry; no independent first-proposal, learning-gain, writer/map benefit or serving comparison.",
        "matplotlib_version": matplotlib.__version__, "font_family": "Liberation Serif",
        "font_file_sha256": sha(font.read_bytes()), "figure_inches": [3.25, 2.05],
        "minimum_text_size_pt": 8.5, "fixed_canvas_no_tight_crop": True,
        "reproduce": "python3 figures/scripts/make_adapter_validation_trace.py",
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "round_window": source["round_window"],
                      "figure_sha256": provenance["figure_sha256"], "journal_verified": bool(args.verify_journal)}))


if __name__ == "__main__":
    main()
