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


def rolling_admissions():
    """Plot full trailing windows from a separately captured admission prefix."""
    data = FIGURES / "data/rolling_admissions.csv"
    receipt = FIGURES / "data/rolling_admissions.source.json"
    checksum = data.with_suffix(".csv.sha256")
    output = FIGURES / "fig_fw_rolling_admissions.pdf"
    provenance = FIGURES / "data/rolling_admissions.provenance.json"
    source = json.loads(receipt.read_text())
    capture = source["capture"]
    capture_hash = source["csv_sha256"]
    assert sha(data) == capture_hash
    assert checksum.read_text().split() == [capture_hash, data.name]
    with data.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == capture["rows"] == 1000
    rounds = [int(r["round"]) for r in rows]
    assert rounds == list(range(rounds[0], rounds[-1] + 1))
    assert [rounds[0], rounds[-1]] == capture["round_window"]
    assert capture["rolling_window"] == 100
    assert capture["all_selected_per_round_scalars_match"]
    assert capture["all_retained_summaries_match"]
    assert {r["evaluation_identity_sha256"] for r in rows} == {
        capture["evaluation_identity_sha256"]
    }
    accepted, gains, ties = [], [], []
    last_best = last_identity = None
    for row in rows:
        candidate, previous, best = [
            Decimal(row[k])
            for k in [
                "candidate_validation_utility",
                "previous_best_validation_utility",
                "retained_best_validation_utility",
            ]
        ]
        assert all(v.is_finite() for v in [candidate, previous, best])
        take = int(row["adapter_accepted"])
        identity = int(row["reported_best_round"])
        assert take in (0, 1) and take == int(candidate >= previous)
        assert best == (candidate if take else previous)
        if last_best is not None:
            assert previous == last_best
            assert take or identity == last_identity
        if take:
            assert identity == int(row["round"])
            (gains if candidate > previous else ties).append(int(row["round"]))
        accepted.append(take)
        last_best, last_identity = best, identity
    windows = []
    for end in range(99, len(rows)):
        count = sum(accepted[end - 99 : end + 1])
        windows.append(
            dict(
                first_round=rounds[end - 99],
                last_round=rounds[end],
                observed_rounds=100,
                accepted=count,
                rejected=100 - count,
            )
        )
    ends = [w["last_round"] for w in windows]
    admissions = [w["accepted"] for w in windows]
    rejections = [w["rejected"] for w in windows]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 3.25), sharex=True)
    for ax, values, title, color in zip(
        axes,
        [admissions, rejections],
        ["(a) Accepted", "(b) Rejected"],
        ["#0b5394", "#626b75"],
    ):
        ax.step(ends, values, where="post", lw=1, color=color)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("decisions / 100 rounds")
        ax.set_xticks([ends[0], ends[300], ends[600], ends[-1]])
        ax.grid(axis="y", alpha=0.2, linewidth=0.4)
        ax.set_axisbelow(True)
    axes[0].set(ylim=(-0.1, max(admissions) + 0.3), yticks=[0, 1, 2])
    axes[1].set(ylim=(min(rejections) - 0.3, 100.1), yticks=[98, 99, 100])
    axes[-1].set_xlabel("completed window-end round")
    fig.tight_layout(pad=0.5, h_pad=0.8)
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    facts = dict(
        schema="whitepaper.rolling-admissions-figure/v1",
        input_csv_sha256=sha(data),
        source_receipt_sha256=sha(receipt),
        generator_sha256=sha(Path(__file__)),
        figure_sha256=sha(output),
        rows=len(rows),
        round_window=capture["round_window"],
        rolling_window=100,
        full_windows=len(windows),
        plotted_end_rounds=[ends[0], ends[-1]],
        accepted=sum(accepted),
        rejected=len(rows) - sum(accepted),
        strict_gain_rounds=gains,
        accepted_tie_rounds=ties,
        windows=windows,
        warmup_rows_not_plotted=99,
        definition="Each plotted endpoint uses exactly 100 consecutive observed decisions; accepted plus rejected equals 100. Overlapping descriptive windows, without padding or independent-trial interpretation.",
        scope=source["scope"],
        matplotlib_version=matplotlib.__version__,
        reproduce=source["reproduce"],
    )
    provenance.write_text(json.dumps(facts, indent=2) + "\n")
    print(f"{len(rows)} observed decisions; {len(windows)} full rolling windows")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Replay program gains or rolling adapter decisions."
    )
    parser.add_argument("--rolling-admissions", action="store_true")
    if parser.parse_args().rolling_admissions:
        rolling_admissions()
    else:
        main()
