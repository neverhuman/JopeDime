#!/usr/bin/env python3
"""Plot event counters derived from two hash-pinned, committed histories."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from itertools import accumulate
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MaxNLocator


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode())))


def verify_inputs(source: dict, root: Path) -> tuple[list[dict], dict[int, dict]]:
    blobs = {}
    for name, digest in source["input_sha256"].items():
        blobs[name] = (root / name).read_bytes()
        assert sha(blobs[name]) == digest, name
    admissions = read_csv(blobs["data/adapter_validation_trace.csv"])
    intervals = read_csv(blobs["data/pool_insertion_cadence.csv"])
    admission_source = json.loads(blobs["data/adapter_validation_trace.source.json"])
    insertion_source = json.loads(blobs["data/pool_insertion_cadence.source.json"])
    assert admission_source["csv_sha256"] == sha(blobs["data/adapter_validation_trace.csv"])
    assert insertion_source["csv_sha256"] == sha(blobs["data/pool_insertion_cadence.csv"])
    assert len(admissions) == admission_source["rows"] == 200
    assert len(intervals) == insertion_source["intervals"] == 100
    rounds = [int(row["round"]) for row in admissions]
    assert rounds == list(range(source["round_window"][0], source["round_window"][1] + 1))
    events = {}
    for row in intervals:
        for prefix in ["previous_", ""]:
            key = int(row[prefix + "source_line"])
            value = {"round": int(row[prefix + "round"]), "timestamp": row[prefix + "timestamp_utc"],
                     "record_sha256": row["previous_record_sha256" if prefix else "source_record_sha256"]}
            assert key not in events or events[key] == value
            events[key] = value
    assert len(events) == insertion_source["source_records"] == 101
    assert sorted(events) == list(range(insertion_source["window_first_source_line"],
                                       insertion_source["window_last_source_line"] + 1))
    assert min(v["round"] for v in events.values()) < rounds[0]
    assert max(v["round"] for v in events.values()) > rounds[-1]
    return admissions, events


def main() -> None:
    figures = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=figures / "data/event_counters.csv")
    parser.add_argument("--inputs-root", type=Path, default=figures)
    parser.add_argument("--output", type=Path, default=figures / "fig_event_counters.pdf")
    parser.add_argument("--provenance", type=Path, default=figures / "data/event_counters.provenance.json")
    args = parser.parse_args()
    receipt_path = args.data.with_suffix(".source.json")
    source = json.loads(receipt_path.read_text())
    data = args.data.read_bytes()
    assert sha(data) == source["csv_sha256"]
    admissions, events = verify_inputs(source, args.inputs_root)
    counts = Counter(v["round"] for v in events.values())
    rounds = [int(row["round"]) for row in admissions]
    per_round_insertions = [counts[r] for r in rounds]
    assert all(row["accepted"] in {"0", "1"} for row in admissions)
    per_round_admissions = [int(row["accepted"]) for row in admissions]
    insertions = list(accumulate(per_round_insertions))
    accepted = list(accumulate(per_round_admissions))
    rows = read_csv(data)
    assert len(rows) == source["rows"] == 200
    for i, row in enumerate(rows):
        assert int(row["round"]) == rounds[i]
        assert int(row["pool_insertions"]) == per_round_insertions[i]
        assert int(row["adapter_admissions"]) == per_round_admissions[i]
        assert int(row["cumulative_pool_insertions"]) == insertions[i]
        assert int(row["cumulative_adapter_admissions"]) == accepted[i]
        assert int(row["adapter_input_csv_line"]) == i + 2
        expected = ";".join(str(k) for k, v in sorted(events.items()) if v["round"] == rounds[i])
        assert row["insertion_source_record_lines"] == expected
    assert insertions[-1] == source["window_insertions"]
    assert accepted[-1] == source["window_adapter_admissions"]
    font = Path(font_manager.findfont(font_manager.FontProperties(family="Liberation Serif"), fallback_to_default=False))
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    fig.subplots_adjust(left=0.17, right=0.975, bottom=0.23, top=0.82)
    ax.step(rounds, insertions, where="post", color="#17649a", linewidth=1.2, label="Pool insertions")
    ax.step(rounds, accepted, where="post", color="#ad4d20", linestyle="--", linewidth=1.2,
            label="Adapter admissions")
    ax.set_xlabel("Logged round identifier")
    ax.set_ylabel("Cumulative events in window")
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.margins(x=0.015, y=0.08)
    ax.grid(axis="y", color="0.88", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=2, frameon=False,
              handlelength=1.5, columnspacing=1, borderaxespad=0)
    fixed = datetime(2026, 11, 3, tzinfo=timezone.utc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Title": "Committed engineering event counters", "Author": "",
                "Subject": "Distinct insertion and admission counts", "CreationDate": fixed, "ModDate": fixed})
    plt.close(fig)
    provenance = {
        "schema": "whitepaper.event-counters-figure/v1", "input_commit": source["input_commit"],
        "input_csv_sha256": sha(data), "source_receipt_sha256": sha(receipt_path.read_bytes()),
        "generator_sha256": sha(Path(__file__).read_bytes()), "figure_sha256": sha(args.output.read_bytes()),
        "rows": 200, "plotted_points": 400, "round_window": source["round_window"],
        "window_insertions": insertions[-1], "window_adapter_admissions": accepted[-1],
        "derivation": "Exact per-round event counts and prefix sums from hash-pinned committed inputs; overlapping interval endpoints deduplicated by original source line, with duplicate values/hashes required identical. No missing-event interpolation or outcome selection.",
        "zero_count_basis": source["zero_count_basis"],
        "scope": "Round-identifier alignment of distinct engineering events only; not joined evaluator/population/normalization, independent learning gain, proposal quality or speed comparison.",
        "matplotlib_version": matplotlib.__version__, "font_family": "Liberation Serif",
        "font_file_sha256": sha(font.read_bytes()), "figure_inches": [3.25, 2.05],
        "minimum_text_size_pt": 8.5, "fixed_canvas_no_tight_crop": True,
        "reproduce": "python3 figures/scripts/make_event_counters.py",
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"committed_inputs_verified": True, "rows": 200, "plotted_points": 400,
                      "pool_insertions": insertions[-1], "adapter_admissions": accepted[-1],
                      "figure_sha256": provenance["figure_sha256"]}))


if __name__ == "__main__":
    main()
