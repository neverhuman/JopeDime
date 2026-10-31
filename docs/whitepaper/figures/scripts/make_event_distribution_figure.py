#!/usr/bin/env python3
"""Summarize training-program event counts in three committed record windows."""

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FIGURES = Path(__file__).resolve().parent.parent
NAMES = ("recent_development", "followup_development", "later_development")
OUTPUT = FIGURES / "fig_fw_event_distribution.pdf"
PROVENANCE = FIGURES / "data" / "event_distribution.provenance.json"
windows = []
for name in NAMES:
    source = FIGURES / "data" / (name + ".csv")
    parent = source.with_suffix(".provenance.json")
    raw, parent_raw = source.read_bytes(), parent.read_bytes()
    receipt = json.loads(parent_raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == receipt["csv_sha256"], "Source CSV/provenance mismatch"
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    rounds = [int(row["round"]) for row in rows]
    events = [int(row["program_promotions"]) for row in rows]
    assert rounds == list(range(rounds[0], rounds[-1] + 1))
    assert all(value >= 0 for value in events)
    running = 0
    for row, count in zip(rows, events):
        running += count
        assert running == int(row["cumulative_program_promotions"])
    assert receipt["round_window"] == [rounds[0], rounds[-1]]
    assert receipt["summary_rows"] == len(rows) == 20
    assert receipt["strict_program_promotions"] == sum(events)
    bins = Counter(min(count, 3) for count in events)
    windows.append(
        {
            "round_window": [rounds[0], rounds[-1]],
            "completed_rounds": len(rows),
            "strict_program_events": sum(events),
            "round_counts_by_event_bin": [bins[i] for i in range(4)],
            "source_csv": "docs/whitepaper/figures/data/" + source.name,
            "source_csv_sha256": digest,
            "parent_provenance_sha256": hashlib.sha256(parent_raw).hexdigest(),
            "snapshot_at": receipt["snapshot"]["at"],
        }
    )

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
fig, ax = plt.subplots(figsize=(3.4, 2.1))
colors = ("#c7cdd4", "#69a6d1", "#5267a5", "#5f3dc4")
left = [0] * len(windows)
for i, label in enumerate(("0", "1", "2", "3+")):
    counts = [window["round_counts_by_event_bin"][i] for window in windows]
    ax.barh(
        range(len(windows)),
        counts,
        left=left,
        height=0.62,
        color=colors[i],
        edgecolor="white",
        linewidth=0.45,
        label=label,
    )
    for y, count in enumerate(counts):
        if count:
            ax.text(
                left[y] + count / 2,
                y,
                str(count),
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if i >= 2 else "#202830",
            )
    left = [a + b for a, b in zip(left, counts)]
ax.set_yticks(
    range(len(windows)),
    [
        f"{w['round_window'][0]}–{w['round_window'][1]}\n({w['strict_program_events']} events)"
        for w in windows
    ],
)
ax.invert_yaxis()
ax.set_xlim(0, 20)
ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
ax.set_xlabel("completed rounds")
ax.set_ylabel("round window")
ax.legend(
    title="program events per round",
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
    fontsize=7.2,
    title_fontsize=8,
    handlelength=1.1,
    columnspacing=0.9,
)
fig.tight_layout(pad=0.4)
fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)
PROVENANCE.write_text(
    json.dumps(
        {
            "schema": "whitepaper.training-event-distribution/v1",
            "windows": windows,
            "bin_definition": ["0 events", "1 event", "2 events", "3 or more events"],
            "source_columns_used": [
                "round",
                "program_promotions",
                "cumulative_program_promotions",
            ],
            "scope": "Separate retained training-side event windows; no stitching of gaps, independent discovery, matched comparison or first-proposal benefit claim",
            "source_binding": "Aggregate field interpretation only; exact loop/checkpoint qualification remains pending",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "figure_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            "matplotlib_version": matplotlib.__version__,
            "reproduce": "python3 figures/scripts/make_event_distribution_figure.py",
        },
        indent=2,
    )
    + "\n"
)
print(
    "; ".join(
        f"{w['round_window']}: {w['completed_rounds']} rounds, "
        f"{w['strict_program_events']} training-program events"
        for w in windows
    )
)
