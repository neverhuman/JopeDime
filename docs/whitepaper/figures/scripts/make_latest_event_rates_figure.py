#!/usr/bin/env python3
"""Replay one captured completed-round event-rate window; never query the live loop."""

import argparse
import csv
import hashlib
import json
from decimal import Decimal, getcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

getcontext().prec = 50
FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data/latest_event_rates.csv"
SOURCE = DATA.with_suffix(".source.json")
PROVENANCE = DATA.with_suffix(".provenance.json")
OUTPUT = FIGURES / "fig_fw_latest_event_rates.pdf"
FIELDS = [
    "round",
    "program_events",
    "wall_seconds",
    "adapter_accepted",
    "reported_best_round",
    "evaluation_identity_sha256",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(rows, capture):
    rounds = [int(row["round"]) for row in rows]
    assert len(rows) == capture["rows"] == 20
    assert rounds == list(range(rounds[0], rounds[-1] + 1))
    assert [rounds[0], rounds[-1]] == capture["round_window"]
    assert len({r["evaluation_identity_sha256"] for r in rows}) == 1
    for row in rows:
        wall = Decimal(row["wall_seconds"])
        assert wall.is_finite() and wall > 0
        assert str(int(row["program_events"])) == str(row["program_events"])
        assert int(row["program_events"]) >= 0
        assert int(row["adapter_accepted"]) in (0, 1)
        assert int(row["reported_best_round"]) <= int(row["round"])


def extract(projection, snapshot):
    meta = json.loads(snapshot.read_text())
    assert sha(projection) == meta["projection_sha256"]
    assert meta["stable_prefix_rereads_verified"] is True
    rows = [json.loads(line) for line in projection.read_text().splitlines()]
    validate(rows, meta)
    with DATA.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    capture = {
        k: meta[k]
        for k in [
            "at",
            "selection",
            "round_window",
            "rows",
            "projection_sha256",
            "stable_prefix_rereads_verified",
        ]
    }
    journal = meta["journal_verification"]
    assert journal["all20_summary_counts_matched"] is True
    assert journal["window_program_events"] == sum(
        int(r["program_events"]) for r in rows
    )
    capture["journal_verification"] = journal
    capture["source_prefixes"] = {
        name: {
            k: meta[key][k]
            for k in [
                "prefix_bytes",
                "prefix_sha256",
                "source_rows",
                "trailing_incomplete_bytes_omitted",
            ]
        }
        for name, key in [
            ("completed_round_summaries", "summary_source"),
            ("adapter_admissions", "admission_source"),
        ]
    }
    SOURCE.write_text(
        json.dumps(
            dict(
                schema="whitepaper.latest-event-rates-source/v1",
                capture=capture,
                csv_sha256=sha(DATA),
                source_binding="Matched aggregate summary/admission fields only; installed source, adapter and population remain unjoined",
                scope="Training-search operating observations; no independent first-proposal or deployment qualification",
            ),
            indent=2,
        )
        + "\n"
    )


def plot():
    source = json.loads(SOURCE.read_text())
    assert sha(DATA) == source["csv_sha256"]
    rows = list(csv.DictReader(DATA.open()))
    validate(rows, source["capture"])
    elapsed = Decimal(0)
    events = 0
    times, counts, rates, running = [], [], [], []
    for row in rows:
        wall = Decimal(row["wall_seconds"])
        count = int(row["program_events"])
        elapsed += wall
        events += count
        times.append(float(elapsed / 60))
        counts.append(events)
        rates.append(float(Decimal(count) * 3600 / wall))
        running.append(float(Decimal(events) * 3600 / elapsed))
    window_rate = Decimal(events) * 3600 / elapsed
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45))
    axes[0].step([0] + times, [0] + counts, where="post", color="#6941c6", lw=1.3)
    axes[0].scatter(times, counts, color="#6941c6", s=8, zorder=3)
    axes[0].set(title="Logged program-incumbent events", ylabel="Cumulative events")
    axes[1].plot(
        times, rates, ".-", color="#7c858e", lw=0.8, ms=3, label="per completed round"
    )
    axes[1].plot(
        times, running, color="#0b5394", lw=1.25, label="cumulative window rate"
    )
    axes[1].axhline(
        float(window_rate),
        color="#212529",
        ls="--",
        lw=0.8,
        label=f"whole window: {window_rate:.2f}/h",
    )
    axes[1].set(title="Event rates on the same clock", ylabel="Events per hour")
    axes[1].legend(frameon=False, loc="upper right", fontsize=6.8)
    for ax in axes:
        ax.set_xlabel("Accumulated completed-round time (min)")
        ax.set_xlim(0, times[-1] * 1.02)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.22, lw=0.4)
    fig.tight_layout(pad=0.5, w_pad=1.1)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    provenance = dict(
        schema="whitepaper.latest-event-rates-figure/v1",
        source_receipt_sha256=sha(SOURCE),
        csv_sha256=sha(DATA),
        generator_sha256=sha(Path(__file__)),
        figure_sha256=sha(OUTPUT),
        rows=len(rows),
        round_window=source["capture"]["round_window"],
        program_events=events,
        logged_wall_seconds=str(elapsed),
        whole_window_events_per_hour=str(window_rate),
        adapter_admissions=sum(int(r["adapter_accepted"]) for r in rows),
        definitions="Per-round rate=3600*events/wall_seconds; cumulative rate=3600*cumulative events/cumulative wall_seconds; clock includes all logged completed-round work and waits",
        transformations="All20 consecutive captured rows including zero-event rounds; cumulative count reset to zero at window start; no gap filling or smoothing",
        source_binding=source["source_binding"],
        scope=source["scope"],
        matplotlib_version=matplotlib.__version__,
        reproduce="python3 figures/scripts/make_latest_event_rates_figure.py",
    )
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: provenance[k]
                for k in [
                    "round_window",
                    "program_events",
                    "logged_wall_seconds",
                    "whole_window_events_per_hour",
                    "adapter_admissions",
                ]
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if bool(args.projection) != bool(args.snapshot):
        parser.error("Supply both projection and capture receipt")
    if args.projection:
        extract(args.projection, args.snapshot)
    plot()


if __name__ == "__main__":
    main()
