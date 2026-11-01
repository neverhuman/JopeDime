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


if __name__ == "__main__":
    main()
