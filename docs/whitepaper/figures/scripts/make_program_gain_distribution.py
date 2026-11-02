#!/usr/bin/env python3
"""Replay the conditional distribution of logged strict program-event gains."""

import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal, getcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

getcontext().prec = 50
FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data/program_gain_events.csv"
SOURCE = FIGURES / "data/program_gain.source.json"
OUTPUT = FIGURES / "fig_fw_program_gain_distribution.pdf"
PROVENANCE = FIGURES / "data/program_gain.provenance.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads(SOURCE.read_text())
    assert sha(DATA) == source["csv_sha256"], "CSV hash mismatch"
    capture = source["capture"]
    window = capture["rounds"]
    rounds = [r["round"] for r in window]
    assert len(window) == capture["completed_rounds"] == 20
    assert rounds == list(range(rounds[0], rounds[0] + 20))
    assert [rounds[0], rounds[-1]] == capture["round_window"]
    assert capture["all20_summary_journal_counts_match"]
    assert capture["all20_retained_round_scalars_match"]
    with DATA.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(capture["events"]) == capture["program_events"]
    gains, counts = [], Counter()
    for index, (row, event) in enumerate(zip(rows, capture["events"]), 1):
        assert int(row["event"]) == event["event"] == index
        assert int(row["round"]) == event["round"] in rounds
        assert event["cv_gate_passed"] and event["evaluation_scope"] == "full_train_cv"
        for field in ["old_utility", "new_utility", "utility_gain"]:
            assert row[field] == event[field], "CSV/journal scalar mismatch"
        old, new, gain = [
            Decimal(row[k]) for k in ["old_utility", "new_utility", "utility_gain"]
        ]
        assert all(x.is_finite() for x in [old, new, gain])
        assert gain == new - old and gain > Decimal("0.000001")
        gains.append(gain)
        counts[int(row["round"])] += 1
    assert [counts[n] for n in rounds] == [r["program_events"] for r in window]
    assert sum(counts[n] == 0 for n in rounds) == capture["zero_event_rounds"]
    assert sum(r["adapter_accepted"] for r in window) == capture["adapter_admissions"]

    frequencies = Counter(gains)
    unique = sorted(frequencies)
    cumulative, ecdf = 0, []
    for gain in unique:
        cumulative += frequencies[gain]
        ecdf.append(cumulative / len(gains))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    left, right = float(unique[0]) / 1.6, float(unique[-1]) * 1.6
    ax.step(
        [left, *map(float, unique), right],
        [0, *ecdf, 1],
        where="post",
        color="#7451b8",
        linewidth=1.1,
    )
    ax.scatter(list(map(float, unique)), ecdf, s=12, color="#7451b8", zorder=3)
    ax.set(
        xscale="log",
        xlim=(left, right),
        ylim=(0, 1.05),
        xlabel="logged utility increment (log scale)",
        ylabel="fraction of strict events",
    )
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    facts = dict(
        schema="whitepaper.program-gain-figure/v1",
        input_csv_sha256=sha(DATA),
        source_receipt_sha256=sha(SOURCE),
        generator_sha256=sha(Path(__file__)),
        figure_sha256=sha(OUTPUT),
        round_window=capture["round_window"],
        completed_rounds=len(window),
        program_events=len(gains),
        zero_event_rounds=capture["zero_event_rounds"],
        adapter_admissions=capture["adapter_admissions"],
        sorted_utility_gains=[str(g) for g in sorted(gains)],
        definition=source["definition"],
        scope=source["scope"],
        matplotlib_version=matplotlib.__version__,
        reproduce=source["reproduce"],
    )
    PROVENANCE.write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts))


if __name__ == "__main__":
    main()
