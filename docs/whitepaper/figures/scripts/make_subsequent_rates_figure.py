#!/usr/bin/env python3
"""Derive binned work/event rates from the committed subsequent-window records."""

import csv
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent.parent
SOURCE = FIGURES / "data/subsequent_window.csv"
PARENT = SOURCE.with_suffix(".provenance.json")
DATA = FIGURES / "data/subsequent_rates.csv"
PROVENANCE = DATA.with_suffix(".provenance.json")
OUTPUT = FIGURES / "fig_fw_subsequent_rates.pdf"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parent = json.loads(PARENT.read_text())
assert sha(SOURCE) == parent["csv_sha256"]
with SOURCE.open() as stream:
    source_rows = list(csv.DictReader(stream))
rounds = [int(row["round"]) for row in source_rows]
assert len(source_rows) == parent["summary_rows"] == 20
assert rounds == list(range(rounds[0], rounds[-1] + 1))
assert parent["round_window"] == [rounds[0], rounds[-1]]
assert all(int(row["adapter_accepted"]) == 0 for row in source_rows)

elapsed = Decimal(0)
events = 0
rows = []
for source in source_rows:
    duration = Decimal(source["completed_round_wall_seconds"])
    count = int(source["program_promotions"])
    assert duration.is_finite() and duration > 0 and count >= 0
    start = elapsed
    elapsed += duration
    events += count
    assert events == int(source["cumulative_program_promotions"])
    rows.append(
        {
            "round": source["round"],
            "completed_round_wall_seconds": source["completed_round_wall_seconds"],
            "round_start_seconds": format(start, "f"),
            "round_end_seconds": format(elapsed, "f"),
            "strict_training_program_events": count,
            "rounds_per_hour": format(Decimal(3600) / duration, ".12f"),
            "training_program_events_per_hour": format(
                Decimal(3600) * count / duration, ".12f"
            ),
        }
    )
assert events == parent["strict_program_promotions"]
assert math.isclose(
    float(elapsed), parent["completed_round_wall_seconds"], abs_tol=1e-9
)
round_mean = 3600 * len(rows) / float(elapsed)
event_mean = 3600 * events / float(elapsed)
assert math.isclose(round_mean, parent["rounds_per_hour"], abs_tol=1e-9)
assert math.isclose(event_mean, parent["program_promotions_per_hour"], abs_tol=1e-9)
with DATA.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
edges = [float(rows[0]["round_start_seconds"]) / 60] + [
    float(row["round_end_seconds"]) / 60 for row in rows
]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.25))
for ax, field, mean, title, label, color in [
    (
        axes[0],
        "rounds_per_hour",
        round_mean,
        "Completed-round pace",
        "rounds per hour",
        "#0b5394",
    ),
    (
        axes[1],
        "training_program_events_per_hour",
        event_mean,
        "Strict training-program event rate",
        "training-program events per hour",
        "#6740b7",
    ),
]:
    values = [float(row[field]) for row in rows]
    ax.stairs(values, edges, color=color, lw=1.0, baseline=None)
    ax.axhline(mean, color="#6e7781", ls="--", lw=0.85, label="window rate from totals")
    ax.set_title(title, fontsize=8, loc="left", pad=4)
    ax.set_xlabel("accumulated completed-round wall time (min)")
    ax.set_ylabel(label)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=7, handlelength=1.5)
fig.tight_layout(pad=0.45, w_pad=1.5)
fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)

provenance = {
    "schema": "whitepaper.subsequent-rates-view/v1",
    "source_csv": str(SOURCE.relative_to(FIGURES)),
    "source_csv_sha256": sha(SOURCE),
    "parent_provenance_sha256": sha(PARENT),
    "snapshot_at": parent["snapshot"]["at"],
    "round_window": parent["round_window"],
    "summary_rows": len(rows),
    "strict_training_program_events": events,
    "adapter_admissions": 0,
    "completed_round_wall_seconds": format(elapsed, "f"),
    "window_rounds_per_hour": round_mean,
    "window_training_program_events_per_hour": event_mean,
    "selection": "Every row of the committed subsequent-window CSV; no outcome selection or new observations",
    "rate_definition": "Per-round bin averages: 3600 / wall_seconds and 3600 * strict training-program events / wall_seconds",
    "bin_edges": "Cumulative completed-round wall seconds; widths equal the logged round durations",
    "window_rate_definition": "Total completed rounds or strict program events divided by summed round wall time, times 3600; not the unweighted average of bin rates",
    "time_scope": "Logged completed-round wall time includes qualification waits; gaps outside completed rounds are omitted",
    "interpretation": "Development work/event rates, not deployment throughput, adapter admission rate or independent first-proposal performance",
    "source_binding": parent["source_binding"],
    "scope": parent["scope"],
    "csv_sha256": sha(DATA),
    "generator_sha256": sha(Path(__file__)),
    "figure_sha256": sha(OUTPUT),
    "matplotlib_version": matplotlib.__version__,
    "reproduce": "python3 figures/scripts/make_subsequent_rates_figure.py",
}
PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
print(
    f"{len(rows)} retained rounds; {events} strict program events; no new observations"
)
