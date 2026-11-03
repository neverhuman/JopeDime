#!/usr/bin/env python3
"""Plot all recorded program-promotion margins in one frozen journal window."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MaxNLocator


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    figures = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=figures / "data/program_promotion_margins.csv")
    parser.add_argument("--output", type=Path, default=figures / "fig_program_promotion_margins.pdf")
    parser.add_argument("--provenance", type=Path, default=figures / "data/program_promotion_margins.provenance.json")
    args = parser.parse_args()
    source_path = args.data.with_suffix(".source.json")
    source = json.loads(source_path.read_text())
    data = args.data.read_bytes()
    assert sha(data) == source["csv_sha256"]
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    assert len(rows) == source["rows"] == 101
    start, end = source["source_line_window"]
    assert [int(row["source_line"]) for row in rows] == list(range(start, end + 1))
    assert [int(row["event_index"]) for row in rows] == list(range(1, 102))
    rounds = [int(row["round"]) for row in rows]
    assert rounds == sorted(rounds) and [rounds[0], rounds[-1]] == source["round_window"]
    margins = []
    for row in rows:
        old, new = Decimal(row["previous_context_utility"]), Decimal(row["context_utility"])
        margin = Decimal(row["promotion_margin"])
        assert old.is_finite() and new.is_finite() and margin.is_finite()
        assert margin == new - old and margin > Decimal(source["program_gate_utility_margin"])
        assert row["cv_gate_passed"] == "1"
        assert len(row["source_record_sha256"]) == 64
        int(row["source_record_sha256"], 16)
        margins.append(margin)
    assert str(min(margins)) == source["min_margin"] and str(max(margins)) == source["max_margin"]
    font = Path(font_manager.findfont(font_manager.FontProperties(family="Liberation Serif"), fallback_to_default=False))
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    fig.subplots_adjust(left=0.19, right=0.975, bottom=0.23, top=0.95)
    values = [float(value) for value in margins]
    ax.scatter(rounds, values, s=9, color="#17649a", alpha=0.8, linewidths=0)
    ax.axhline(float(source["program_gate_utility_margin"]), color="0.4", linestyle=":", linewidth=0.7)
    ax.set_yscale("log")
    ax.set_ylim(float(source["program_gate_utility_margin"]) * 0.7, max(values) * 1.8)
    ax.set_xlabel("Logged round identifier")
    ax.set_ylabel("Context utility increment")
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.margins(x=0.025)
    ax.grid(axis="y", color="0.88", linewidth=0.4)
    ax.set_axisbelow(True)
    fixed = datetime(2026, 11, 3, tzinfo=timezone.utc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Title": "Training-program promotion margins", "Author": "",
                "Subject": "Conditional admitted-program engineering telemetry", "CreationDate": fixed, "ModDate": fixed})
    plt.close(fig)
    provenance = {
        "schema": "whitepaper.program-promotion-margins-figure/v1",
        "input_csv_sha256": sha(data), "source_receipt_sha256": sha(source_path.read_bytes()),
        "generator_sha256": sha(Path(__file__).read_bytes()), "figure_sha256": sha(args.output.read_bytes()),
        "source_window_sha256": source["source_window_sha256"], "rows": len(rows), "plotted_points": len(rows),
        "round_window": source["round_window"], "source_line_window": source["source_line_window"],
        "derivation": source["derivation"], "scope": source["scope"], "vertical_scale": "logarithmic",
        "matplotlib_version": matplotlib.__version__, "font_family": "Liberation Serif",
        "font_file_sha256": sha(font.read_bytes()), "figure_inches": [3.25, 2.05],
        "minimum_text_size_pt": 8.5, "fixed_canvas_no_tight_crop": True,
        "reproduce": "python3 figures/scripts/make_program_promotion_margins.py",
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "plotted_points": len(rows), "figure_sha256": provenance["figure_sha256"]}))


if __name__ == "__main__":
    main()
