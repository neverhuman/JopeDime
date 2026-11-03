#!/usr/bin/env python3
"""Show program-journal counts for a frozen set of observed decision rounds."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    figures = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=figures / "data/program_round_counts.csv")
    parser.add_argument("--output", type=Path, default=figures / "fig_program_round_counts.pdf")
    parser.add_argument("--provenance", type=Path, default=figures / "data/program_round_counts.provenance.json")
    args = parser.parse_args()
    source_path, events_path = args.data.with_suffix(".source.json"), args.data.with_suffix(".events.csv")
    source = json.loads(source_path.read_text())
    rd, ed = args.data.read_bytes(), events_path.read_bytes()
    assert sha(rd) == source["rounds_csv_sha256"] and sha(ed) == source["events_csv_sha256"]
    rounds = list(csv.DictReader(io.StringIO(rd.decode())))
    events = list(csv.DictReader(io.StringIO(ed.decode())))
    ids = [int(r["round"]) for r in rounds]
    assert ids == sorted(set(ids)) == source["round_ids"] and len(ids) == 200
    assert [int(r["decision_index"]) for r in rounds] == list(range(1, 201))
    assert len(events) == source["program_records_in_bounding_window"] == 92
    counts = Counter(int(e["round"]) for e in events)
    for e in events:
        assert int(e["decision_observed"]) == (int(e["round"]) in ids)
        assert len(e["program_source_record_sha256"]) == 64
        int(e["program_source_record_sha256"], 16)
    values = [int(r["program_records"]) for r in rounds]
    assert all(v == counts[r] for r, v in zip(ids, values))
    assert sum(values) == source["paired_program_records"] == 91
    assert {str(k): v for k, v in Counter(values).items()} == source["observed_round_count_distribution"]
    missing = source["missing_decision_round_ids"]
    assert missing == sorted(set(range(ids[0], ids[-1] + 1)) - set(ids)) == [4331]
    assert sum(counts[r] for r in missing) == source["unpaired_program_records"] == 1
    font = Path(font_manager.findfont(font_manager.FontProperties(family="Liberation Serif"), fallback_to_default=False))
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 1.30), gridspec_kw={"width_ratios": [2.3, 1]})
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.31, top=0.97, wspace=0.24)
    ax, hist = axes
    ax.bar(ids, values, width=0.85, color="#17649a", linewidth=0)
    for r in missing:
        ax.axvline(r, color="0.55", linestyle=":", linewidth=0.7)
        ax.bar(r, counts[r], width=0.85, facecolor="0.8", edgecolor="0.25", linewidth=0.5)
    ax.set_xlim(ids[0] - 1, ids[-1] + 1)
    ax.set_ylim(0, max(values) * 1.5)
    ax.set_xlabel("Logged round")
    ax.set_ylabel("Records")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.legend(handles=[Patch(facecolor="#17649a", label="Decision recorded"),
                       Line2D([], [], color="0.55", linestyle=":", label="No decision record")],
              loc="upper right", frameon=False, ncol=2, fontsize=8.5, handlelength=1.2, columnspacing=0.8)
    distribution = Counter(values)
    categories = list(range(max(values) + 1))
    heights = [distribution[k] for k in categories]
    hist.bar(categories, heights, width=0.7, color="#17649a", linewidth=0)
    for k, v in zip(categories, heights):
        hist.annotate(str(v), (k, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8.5)
    hist.set_ylim(0, max(heights) * 1.32)
    hist.set_xticks(categories)
    hist.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    hist.set_xlabel("Records per observed round")
    hist.set_ylabel("Decisions")
    for a in axes:
        a.grid(axis="y", color="0.9", linewidth=0.4)
        a.set_axisbelow(True)
    fixed = datetime(2026, 11, 3, tzinfo=timezone.utc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Title": "Program records by logged round", "Author": "",
                "Subject": "Training-side event counts with a missing decision marked", "CreationDate": fixed, "ModDate": fixed})
    plt.close(fig)
    provenance = {"schema": "whitepaper.program-round-counts-figure/v1", "rounds_csv_sha256": sha(rd), "events_csv_sha256": sha(ed),
        "source_receipt_sha256": sha(source_path.read_bytes()), "generator_sha256": sha(Path(__file__).read_bytes()),
        "figure_sha256": sha(args.output.read_bytes()), "admission_window_sha256": source["admission_window_sha256"],
        "program_window_sha256": source["program_window_sha256"], "observed_decisions": len(rounds), "program_records": len(events),
        "paired_program_records": sum(values), "unpaired_program_records": source["unpaired_program_records"],
        "missing_decision_round_ids": missing, "zero_event_observed_rounds": distribution[0],
        "scope": source["scope"], "matplotlib_version": matplotlib.__version__, "font_family": "Liberation Serif",
        "font_file_sha256": sha(font.read_bytes()), "figure_inches": [6.75, 1.30], "minimum_text_size_pt": 8.5,
        "vertical_scale": "linear", "fixed_canvas_no_tight_crop": True,
        "reproduce": "python3 figures/scripts/make_program_round_counts.py"}
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"observed_decisions": len(rounds), "program_records": len(events), "figure_sha256": provenance["figure_sha256"]}))


if __name__ == "__main__":
    main()
