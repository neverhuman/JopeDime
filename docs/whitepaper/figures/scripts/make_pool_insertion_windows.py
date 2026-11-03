#!/usr/bin/env python3
"""Plot insertion counts per logged round in fixed, complete round windows."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

from make_event_counters import read_csv, sha, verify_inputs
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib


def main() -> None:
    figures = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=figures / "data/pool_insertion_windows.csv")
    parser.add_argument("--inputs-root", type=Path, default=figures)
    parser.add_argument("--output", type=Path, default=figures / "fig_pool_insertion_windows.pdf")
    parser.add_argument("--provenance", type=Path, default=figures / "data/pool_insertion_windows.provenance.json")
    args = parser.parse_args()
    source_path = args.data.with_suffix(".source.json")
    source = json.loads(source_path.read_text())
    data = args.data.read_bytes()
    assert sha(data) == source["csv_sha256"]
    for name, digest in source["input_sha256"].items():
        assert sha((args.inputs_root / name).read_bytes()) == digest, name
    parent_source = json.loads((args.inputs_root / "data/event_counters.source.json").read_text())
    parent_bytes = (args.inputs_root / "data/event_counters.csv").read_bytes()
    assert sha(parent_bytes) == parent_source["csv_sha256"]
    parent = read_csv(parent_bytes)
    admissions, events = verify_inputs(parent_source, args.inputs_root)
    rounds = [int(row["round"]) for row in parent]
    assert rounds == [int(row["round"]) for row in admissions] == list(range(4064, 4264))
    counts = Counter(v["round"] for v in events.values())
    for row in parent:
        r = int(row["round"])
        assert int(row["pool_insertions"]) == counts[r]
        assert row["insertion_source_record_lines"] == ";".join(
            str(k) for k, v in sorted(events.items()) if v["round"] == r)
    rows = read_csv(data)
    assert len(parent) == source["input_rounds"] == 200
    assert len(rows) == source["windows"] == 10 and source["window_rounds"] == 20
    totals = []
    for i, row in enumerate(rows):
        window = parent[i * 20:(i + 1) * 20]
        total = sum(int(r["pool_insertions"]) for r in window)
        assert int(row["window"]) == i + 1
        assert int(row["first_round"]) == int(window[0]["round"])
        assert int(row["last_round"]) == int(window[-1]["round"])
        assert int(row["round_count"]) == len(window) == 20
        assert int(row["pool_insertions"]) == total
        assert Decimal(row["insertions_per_round"]) == Decimal(total) / 20
        assert row["input_csv_lines"] == ";".join(str(j + 2) for j in range(i * 20, (i + 1) * 20))
        assert row["insertion_source_record_lines"] == ";".join(
            r["insertion_source_record_lines"] for r in window if r["insertion_source_record_lines"])
        assert 0 <= total <= 20
        totals.append(total)
    assert totals == source["window_insertions"] and sum(totals) == source["total_insertions"] == 89
    font = Path(font_manager.findfont(font_manager.FontProperties(family="Liberation Serif"), fallback_to_default=False))
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.23, top=0.95)
    starts = [int(row["first_round"]) for row in rows]
    rates = [float(row["insertions_per_round"]) for row in rows]
    ax.bar(starts, rates, width=16, color="#17649a", align="center", linewidth=0)
    ax.set_xlabel("First logged round in window")
    ax.set_ylabel("Insertions per logged round")
    ax.set_xticks(starts[::2])
    ax.set_xlim(starts[0] - 12, starts[-1] + 12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.grid(axis="y", color="0.88", linewidth=0.4)
    ax.set_axisbelow(True)
    fixed = datetime(2026, 11, 3, tzinfo=timezone.utc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Title": "Pool insertion frequency in fixed round windows", "Author": "",
                "Subject": "Engineering events per logged round", "CreationDate": fixed, "ModDate": fixed})
    plt.close(fig)
    provenance = {
        "schema": "whitepaper.pool-insertion-windows-figure/v1", "input_commit": source["input_commit"],
        "input_csv_sha256": sha(data), "source_receipt_sha256": sha(source_path.read_bytes()),
        "generator_sha256": sha(Path(__file__).read_bytes()), "figure_sha256": sha(args.output.read_bytes()),
        "helper_script_sha256": source["input_sha256"]["scripts/make_event_counters.py"],
        "input_rounds": 200, "window_rounds": 20, "windows": 10, "plotted_points": 10,
        "round_window": [4064, 4263], "window_insertions": totals, "total_insertions": sum(totals),
        "derivation": source["bin_policy"], "lineage": source["lineage"], "scope": source["scope"],
        "matplotlib_version": matplotlib.__version__, "font_family": "Liberation Serif",
        "font_file_sha256": sha(font.read_bytes()), "figure_inches": [3.25, 2.05],
        "minimum_text_size_pt": 8.5, "fixed_canvas_no_tight_crop": True,
        "reproduce": "python3 figures/scripts/make_pool_insertion_windows.py",
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"input_rounds": 200, "windows": 10, "total_insertions": sum(totals),
                      "figure_sha256": provenance["figure_sha256"]}))


if __name__ == "__main__":
    main()
