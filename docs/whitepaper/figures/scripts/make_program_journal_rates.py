#!/usr/bin/env python3
"""Plot every captured program-pool journal addition in fixed clock bins."""
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
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    figures = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=figures / "data/program_journal_rates.csv")
    parser.add_argument("--output", type=Path, default=figures / "fig_program_journal_rates.pdf")
    parser.add_argument("--provenance", type=Path, default=figures / "data/program_journal_rates.provenance.json")
    args = parser.parse_args()
    source_path = args.data.with_suffix(".source.json")
    source = json.loads(source_path.read_text())
    events_path = args.data.with_suffix(".events.csv")
    bins_data, events_data = args.data.read_bytes(), events_path.read_bytes()
    assert sha(bins_data) == source["bins_csv_sha256"]
    assert sha(events_data) == source["events_csv_sha256"]
    rows = list(csv.DictReader(io.StringIO(bins_data.decode())))
    events = list(csv.DictReader(io.StringIO(events_data.decode())))
    assert len(rows) == source["bin_count"] == 21
    assert len(events) == source["complete_prefix_records"] == 3490
    counts = [[0, 0] for _ in rows]
    for i, event in enumerate(events, 1):
        assert int(event["event_index"]) == int(event["source_line"]) == i
        assert len(event["source_record_sha256"]) == 64
        int(event["source_record_sha256"], 16)
        j = int(event["bin_index"])
        assert 0 <= j < len(rows)
        assert event["kind"] in {"promotion", "historical_admission"}
        counts[j][event["kind"] == "historical_admission"] += 1
    for j, row in enumerate(rows):
        assert int(row["bin_index"]) == j
        assert int(row["exposure_us"]) == int(row["end_offset_us"]) - int(row["start_offset_us"]) > 0
        assert counts[j] == [int(row["ordinary_promotions"]), int(row["historical_admissions"])]
        assert sum(counts[j]) == int(row["total_records"])
        for key, count in [("ordinary_per_hour", counts[j][0]), ("historical_per_hour", counts[j][1]), ("records_per_hour", sum(counts[j]))]:
            assert Decimal(row[key]) == Decimal(count) * Decimal(3600000000) / Decimal(row["exposure_us"])
        assert int(row["partial"]) == (int(row["exposure_us"]) < source["bin_duration_us"])
    assert sum(sum(c) for c in counts) == len(events)
    assert sum(c[0] for c in counts) == source["ordinary_promotions"] == 3405
    assert sum(c[1] for c in counts) == source["historical_admissions"] == 85
    assert sum(int(r["partial"]) for r in rows) == source["partial_bins"] == 1
    assert source["unrecorded_protocol_ids"] == 90 and source["missing_protocol_is_a_pass"] is False
    font = Path(font_manager.findfont(font_manager.FontProperties(family="Liberation Serif"), fallback_to_default=False))
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.75, 1.70))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.27, top=0.96)
    centers = [(int(r["start_offset_us"]) + int(r["end_offset_us"])) / 7200000000 for r in rows]
    widths = [int(r["exposure_us"]) / 3600000000 * 0.9 for r in rows]
    normal = [float(r["ordinary_per_hour"]) for r in rows]
    historical = [float(r["historical_per_hour"]) for r in rows]
    ax.bar(centers, normal, widths, color="#17649a", linewidth=0)
    ax.bar(centers, historical, widths, bottom=normal, color="0.85", edgecolor="0.4", hatch="///", linewidth=0.4)
    total = [a + b for a, b in zip(normal, historical)]
    for j, row in enumerate(rows):
        if int(row["partial"]):
            ax.bar(centers[j], total[j], widths[j], facecolor="none", edgecolor="black", linewidth=0.8)
            ax.annotate("partial", (centers[j], total[j]), xytext=(-6, 8), textcoords="offset points", ha="right", fontsize=8.5)
    transition = source["cohort_transition"]
    before = [e for e in events if int(e["round"]) < transition["round"]]
    after = [e for e in events if int(e["round"]) >= transition["round"]]
    left = max(before, key=lambda e: datetime.fromisoformat(e["timestamp_utc"]))
    right = min(after, key=lambda e: datetime.fromisoformat(e["timestamp_utc"]))
    start = datetime.fromisoformat(source["first_timestamp_utc"])
    def offset_us(event):
        dt = datetime.fromisoformat(event["timestamp_utc"]) - start
        return (dt.days * 86400 + dt.seconds) * 1000000 + dt.microseconds
    bracket = [offset_us(left), offset_us(right)]
    assert bracket == transition["clock_bracket_offset_us"]
    assert [int(left["source_line"]), int(right["source_line"])] == [transition["last_warmup_source_line"], transition["first_full_cohort_source_line"]]
    assert [len(before), len(after)] == [transition["warmup_records"], transition["full_cohort_records"]]
    assert left["source_record_sha256"] == transition["last_warmup_record_sha256"]
    assert right["source_record_sha256"] == transition["first_full_cohort_record_sha256"]
    assert sha(events_data) == transition["source_events_csv_sha256"]
    bounds = [u / 3600000000 for u in bracket]
    ax.axvspan(*bounds, color="0.92", zorder=0)
    for bound in bounds:
        ax.axvline(bound, color="0.45", linewidth=0.6, linestyle=":", zorder=1)
    ax.annotate("Round 775 transition\n(warm-up / full cohort)",
                xy=(sum(bounds) / 2, 0.56), xycoords=("data", "axes fraction"),
                xytext=(0.22, 0.66), textcoords="axes fraction", fontsize=8.5,
                ha="left", va="center", arrowprops={"arrowstyle":"-", "linewidth":0.6, "color":"0.35"})
    ax.set_ylim(0, max(total) * 1.24)
    ax.set_xlim(-1, int(rows[-1]["end_offset_us"]) / 3600000000 + 1)
    ax.set_xlabel("Hours since first journal timestamp")
    ax.set_ylabel("Journal records per hour")
    ax.xaxis.set_major_locator(MaxNLocator(7, integer=True))
    ax.grid(axis="y", color="0.9", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor="#17649a", label="Ordinary promotions"),
                       Patch(facecolor="0.85", edgecolor="0.4", hatch="///", label="Historical admissions")],
              loc="upper right", frameon=False, fontsize=8.5, ncol=2)
    fixed = datetime(2026, 11, 3, tzinfo=timezone.utc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Title": "Program-pool journal rates", "Author": "",
                "Subject": "Adaptive training-side journal additions", "CreationDate": fixed, "ModDate": fixed})
    plt.close(fig)
    provenance = {"schema": "whitepaper.program-journal-rates-figure/v1", "bins_csv_sha256": sha(bins_data),
        "events_csv_sha256": sha(events_data), "source_receipt_sha256": sha(source_path.read_bytes()),
        "generator_sha256": sha(Path(__file__).read_bytes()), "figure_sha256": sha(args.output.read_bytes()),
        "source_prefix_sha256": source["complete_prefix_sha256"], "source_records": len(events),
        "plotted_bins": len(rows), "full_bins": source["full_bins"], "partial_bins": source["partial_bins"],
        "ordinary_promotions": source["ordinary_promotions"], "historical_admissions": source["historical_admissions"],
        "unrecorded_protocol_ids": source["unrecorded_protocol_ids"], "scope": source["scope"],
        "matplotlib_version": matplotlib.__version__, "font_family": "Liberation Serif", "font_file_sha256": sha(font.read_bytes()),
        "figure_inches": [6.75, 1.70], "minimum_text_size_pt": 8.5, "vertical_scale": "linear",
        "bar_width_tracks_observed_exposure": True, "fixed_canvas_no_tight_crop": True,
        "cohort_transition": transition,
        "reproduce": "python3 figures/scripts/make_program_journal_rates.py"}
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"source_records": len(events), "plotted_bins": len(rows), "figure_sha256": provenance["figure_sha256"]}))


if __name__ == "__main__":
    main()
