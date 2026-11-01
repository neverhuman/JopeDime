#!/usr/bin/env python3
"""Plot operating quantities from separately retained windows."""

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
DATA = FIGURES / "data/window_operating_inputs.csv"
SOURCE = FIGURES / "data/window_operating.source.json"
PROVENANCE = FIGURES / "data/latest_event_rates.provenance.json"
OUTPUT = FIGURES / "fig_fw_latest_event_rates.pdf"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source = json.loads(SOURCE.read_text())
    assert sha(DATA) == source["csv_sha256"]
    rows = list(csv.DictReader(DATA.open()))
    groups = {}
    for row in rows:
        key = (int(row["window_start"]), int(row["window_end"]))
        groups.setdefault(key, []).append(row)
    facts = []
    for key, window in sorted(groups.items()):
        assert len(window) == 20 and [int(r["round"]) for r in window] == list(
            range(key[0], key[1] + 1)
        )
        walls = [Decimal(r["wall_seconds"]) for r in window]
        assert all(x.is_finite() and x > 0 for x in walls)
        events = [int(r["program_events"]) for r in window]
        assert all(x >= 0 for x in events)
        wall = sum(walls)
        count = sum(events)
        facts.append(
            dict(
                round_window=list(key),
                rows=20,
                events=count,
                wall_seconds=str(wall),
                events_per_hour=str(Decimal(count) * 3600 / wall),
                mean_wall_seconds=str(wall / 20),
                adapter_admissions=sum(int(r["adapter_accepted"]) for r in window),
            )
        )
    assert (
        len(facts) == 5
        and facts[-1]["round_window"] == source["capture"]["round_window"]
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.8, 2.55), sharey=True)
    yy = list(range(len(facts)))
    labels = [f"{r['round_window'][0]}–{r['round_window'][1]}" for r in facts]
    ax0.barh(
        yy, [float(r["events_per_hour"]) for r in facts], color="#4c6ef5", height=0.55
    )
    ax1.barh(
        yy, [float(r["mean_wall_seconds"]) for r in facts], color="#e8590c", height=0.55
    )
    ax0.set_yticks(yy, labels)
    ax0.invert_yaxis()
    ax0.set(
        title="Strict program-event rate",
        xlabel="events / logged hour",
        ylabel="separate retained rounds",
    )
    ax1.set(title="Mean completed-round duration", xlabel="logged seconds / round")
    for ax in (ax0, ax1):
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    for y, r in zip(yy, facts):
        ax0.text(
            float(r["events_per_hour"]) + 0.3,
            y,
            f"{r['events']} events",
            va="center",
            fontsize=6,
        )
    ax0.set_xlim(0, max(float(r["events_per_hour"]) for r in facts) * 1.35)
    fig.tight_layout(pad=0.6, w_pad=1.5)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    PROVENANCE.write_text(
        json.dumps(
            dict(
                schema="whitepaper.window-operating-summary/v2",
                input_csv_sha256=sha(DATA),
                source_receipt_sha256=sha(SOURCE),
                generator_sha256=sha(Path(__file__)),
                figure_sha256=sha(OUTPUT),
                windows=facts,
                definitions="Whole-window rate=3600*sum(events)/sum(full round wall seconds); mean wall=sum(wall)/20. No gaps imputed; windows remain separate.",
                scope=source["scope"],
                matplotlib_version=matplotlib.__version__,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(facts))


def make_round_clock():
    """Show every completed round in a separately authenticated live capture."""
    from matplotlib.ticker import MaxNLocator

    data = FIGURES / "data/latest_round_clock.csv"
    receipt = FIGURES / "data/latest_round_clock.source.json"
    output = FIGURES / "fig_fw_latest_round_clock.pdf"
    provenance = FIGURES / "data/latest_round_clock.provenance.json"
    source = json.loads(receipt.read_text())
    assert sha(data) == source["csv_sha256"], "Round-clock CSV hash mismatch"
    with data.open() as stream:
        rows = list(csv.DictReader(stream))
    rounds = [int(row["round"]) for row in rows]
    assert len(rows) == source["capture"]["rows"] == 20
    assert rounds == list(range(rounds[0], rounds[0] + 20))
    assert [rounds[0], rounds[-1]] == source["capture"]["round_window"]
    walls = [Decimal(row["wall_seconds"]) for row in rows]
    events = [int(row["program_events"]) for row in rows]
    admissions = [int(row["adapter_accepted"]) for row in rows]
    assert all(value.is_finite() and value > 0 for value in walls)
    assert all(value >= 0 for value in events)
    assert all(value in (0, 1) for value in admissions)
    assert source["capture"]["journal_verification"]["all20_summary_counts_matched"]
    total_wall = sum(walls)
    total_events = sum(events)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.8, 2.2))
    ax0.bar(rounds, events, width=0.75, color="#4c6ef5")
    ax0.set(title="Events per completed round", ylabel="strict program events")
    ax0.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax0.set_ylim(0, max(events) + 0.5)
    ax1.plot(rounds, [float(value) for value in walls], "o-", color="#495057", ms=2)
    ax1.set(title="Full logged round time", ylabel="seconds per round")
    ax1.set_ylim(0, float(max(walls)) * 1.12)
    for ax in (ax0, ax1):
        ax.set(xlabel="round", xlim=(rounds[0] - 0.75, rounds[-1] + 0.75))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.6, w_pad=1.4)
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    facts = dict(
        schema="whitepaper.latest-round-clock-figure/v1",
        input_csv_sha256=sha(data),
        source_receipt_sha256=sha(receipt),
        generator_sha256=sha(Path(__file__)),
        figure_sha256=sha(output),
        rows=len(rows),
        round_window=[rounds[0], rounds[-1]],
        program_events=total_events,
        logged_wall_seconds=str(total_wall),
        events_per_logged_hour=str(Decimal(total_events) * 3600 / total_wall),
        mean_round_seconds=str(total_wall / len(rows)),
        zero_event_rounds=events.count(0),
        adapter_admissions=sum(admissions),
        definitions="Each bar counts strict program-incumbent events in one completed summary; each point is its full logged wall duration, including all work and waits.",
        scope=source["scope"],
        matplotlib_version=matplotlib.__version__,
        reproduce="python3 figures/scripts/make_latest_event_rates_figure.py --round-clock-only",
    )
    provenance.write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-clock-only", action="store_true")
    args = parser.parse_args()
    if args.round_clock_only:
        make_round_clock()
    else:
        main()
