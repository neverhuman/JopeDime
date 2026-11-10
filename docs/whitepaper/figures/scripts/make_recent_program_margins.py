#!/usr/bin/env python3
"""Plot all recorded utility gains in a fixed training-program journal window."""
from __future__ import annotations
from flywheel_typography import style_figure, finalize_typography_receipts
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MaxNLocator

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/recent_program_margins.csv")
    parser.add_argument("--output", type=Path, default=root / "fig_recent_program_margins.pdf")
    parser.add_argument("--provenance", type=Path, default=root / "data/recent_program_margins.provenance.json")
    args = parser.parse_args()
    source_path = args.data.with_suffix(".source.json")
    source = json.loads(source_path.read_text())
    raw = args.data.read_bytes()
    assert sha(raw) == source["csv_sha256"]
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    assert len(rows) == source["program_window_records"] == 92
    assert [int(r["event_index"]) for r in rows] == list(range(1, 93))
    assert [int(r["source_line"]) for r in rows] == source["source_line_numbers"]
    gains = [float(r["gain"]) for r in rows]
    assert all(math.isfinite(v) and v > source["threshold"] for v in gains)
    assert all(float(r["new_utility"]) - float(r["old_utility"]) == v for r, v in zip(rows, gains))
    assert all(r["protocol"] == source["protocol"] for r in rows)
    assert [int(r["round"]) for r in rows if not int(r["decision_observed"])] == [4331]
    font = Path(font_manager.findfont(font_manager.FontProperties(family="Liberation Serif"), fallback_to_default=False))
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, cdf) = plt.subplots(1, 2, figsize=(6.75, 1.30), gridspec_kw={"width_ratios": [1.65, 1]})
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.31, top=0.78, wspace=0.30)
    for observed, color, label in [(1, "#17649a", "Decision recorded"), (0, "0.5", "No decision record")]:
        selected = [(int(r["round"]), v) for r, v in zip(rows, gains) if int(r["decision_observed"]) == observed]
        ax.scatter([x for x, v in selected], [v for x, v in selected], s=8, c=color, linewidths=0, label=label, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(8e-7, 0.5)
    ax.set_xlim(source["bounding_round_window"][0] - 1, source["bounding_round_window"][1] + 1)
    ax.axhline(source["threshold"], color="0.5", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Logged round")
    ax.set_ylabel("Utility gain")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0.1, frameon=False, ncol=2, fontsize=8.5, handlelength=0.7, columnspacing=0.6)
    ordered = sorted(gains)
    cdf.step([source["threshold"]] + ordered, [0] + [(i + 1) / len(ordered) for i in range(len(ordered))], where="post", color="#17649a", linewidth=1)
    cdf.set_xscale("log")
    cdf.set_xlim(8e-7, 0.3)
    cdf.set_ylim(0, 1.04)
    cdf.set_yticks([0, 0.5, 1])
    cdf.set_xlabel("Recorded utility gain")
    cdf.set_ylabel("Fraction")
    for a in [ax, cdf]:
        a.grid(axis="y", color="0.9", linewidth=0.4)
        a.set_axisbelow(True)
    fixed = datetime(2026, 11, 3, tzinfo=timezone.utc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    style_figure(fig, args.output)
    fig.savefig(args.output, metadata={"Title": "Recent training-program journal gains", "Author": "",
                "Subject": "Conditional two-fold program-incumbent utility differences", "CreationDate": fixed, "ModDate": fixed})
    plt.close(fig)
    provenance = {"schema": "whitepaper.recent-program-margins-figure/v1", "csv_sha256": sha(raw),
                  "source_receipt_sha256": sha(source_path.read_bytes()), "generator_sha256": sha(Path(__file__).read_bytes()),
                  "figure_sha256": sha(args.output.read_bytes()), "program_window_sha256": source["program_window_sha256"],
                  "records": len(rows), "minimum_gain": min(gains), "maximum_gain": max(gains),
                  "protocol": source["protocol"], "threshold": source["threshold"], "scope": source["scope"],
                  "matplotlib_version": matplotlib.__version__, "font_family": "Liberation Serif", "font_file_sha256": sha(font.read_bytes()),
                  "figure_inches": [6.75, 1.30], "minimum_text_size_pt": 8.5, "gain_axes": "logarithmic", "ecdf_denominator": len(rows),
                  "fixed_canvas_no_tight_crop": True, "reproduce": "python3 figures/scripts/make_recent_program_margins.py"}
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"records": len(rows), "figure_sha256": provenance["figure_sha256"]}))

if __name__ == "__main__":
    main()

finalize_typography_receipts(__file__)
