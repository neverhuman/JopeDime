#!/usr/bin/env python3
"""Plot two distinct journal event counts in consecutive round windows."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from matplotlib.ticker import MaxNLocator

FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data/pool_insertions.csv"
SOURCE = FIGURES / "data/pool_insertions.source.json"
OUTPUT = FIGURES / "fig_fw_pool_insertions.pdf"
PROVENANCE = FIGURES / "data/pool_insertions.provenance.json"
WINDOW = 20


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads(SOURCE.read_text())
    assert sha(DATA) == source["csv_sha256"]
    with DATA.open() as stream:
        rows = list(csv.DictReader(stream))
    rounds = [int(row["round"]) for row in rows]
    first, last = source["round_window"]
    assert rounds == list(range(first, last + 1))
    assert rounds[0] == 3174 and len(rows) == source["rows"]
    insertions = [int(row["automatic_pool_insertions"]) for row in rows]
    rejected = [int(row["adapter_rejected"]) for row in rows]
    assert all(value >= 0 for value in insertions)
    assert all(value in (0, 1) for value in rejected)
    assert sum(insertions) == source["automatic_pool_insertions"]
    assert sum(rejected) == source["rejected_adapter_candidates"]

    windows = []
    for start in range(0, len(rows), WINDOW):
        stop = min(start + WINDOW, len(rows))
        windows.append(dict(
            first_round=rounds[start], last_round=rounds[stop - 1],
            rounds=stop - start,
            automatic_pool_insertions=sum(insertions[start:stop]),
            rejected_adapter_candidates=sum(rejected[start:stop]),
        ))
    assert all(window["rounds"] == WINDOW for window in windows[:-1])
    assert 1 <= windows[-1]["rounds"] <= WINDOW
    assert sum(window["rounds"] for window in windows) == len(rows)
    centers = [(window["first_round"] + window["last_round"]) / 2 for window in windows]
    widths = [window["rounds"] * 0.85 for window in windows]

    font = Path(findfont("Liberation Serif", fallback_to_default=False))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif"],
        "font.size": 8.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 2.35), sharex=True)
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.20, top=0.90, hspace=0.60)
    series = [
        ("automatic_pool_insertions", "Training-pool insertions", "#1971c2"),
        ("rejected_adapter_candidates", "Rejected adapter candidates", "#7451b8"),
    ]
    for ax, (key, title, color) in zip(axes, series):
        values = [window[key] for window in windows]
        ax.bar(centers, values, width=widths, color=color, linewidth=0)
        ax.set_title(title, fontsize=8.5, loc="left", pad=3)
        ax.set_ylabel("count", fontsize=8.5)
        ax.set_ylim(0, max(1, max(values) * 1.18))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.tick_params(labelsize=8.5)
        ax.grid(axis="y", alpha=0.2, linewidth=0.4)
        ax.set_axisbelow(True)
    padding = max(2, (rounds[-1] - rounds[0]) * 0.04)
    axes[-1].set_xlim(rounds[0] - padding, rounds[-1] + padding)
    axes[-1].set_xticks([rounds[0], rounds[len(rounds) // 2], rounds[-1]])
    axes[-1].set_xlabel("completed round; windows of 20", fontsize=8.5)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    facts = dict(
        schema="whitepaper.pool-insertions-figure/v1",
        input_csv_sha256=sha(DATA), source_receipt_sha256=sha(SOURCE),
        generator_sha256=sha(Path(__file__)), figure_sha256=sha(OUTPUT),
        rows=len(rows), round_window=[rounds[0], rounds[-1]],
        window_size=WINDOW, windows=windows,
        automatic_pool_insertions=sum(insertions),
        rejected_adapter_candidates=sum(rejected),
        transformations="Separate raw count sums in consecutive nonoverlapping windows of 20 rounds; final partial window retained. No smoothing, normalization or outcome subsampling.",
        scope=source["scope"], source_binding=source["source_binding"],
        matplotlib_version=matplotlib.__version__, font_family="Liberation Serif",
        font_file_sha256=sha(font), figure_inches=[3.25, 2.35],
        minimum_text_size_pt=8.5, fixed_canvas_no_tight_crop=True,
        reproduce="python3 figures/scripts/make_pool_insertions_figure.py",
    )
    PROVENANCE.write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps({key: value for key, value in facts.items() if key != "windows"}))


if __name__ == "__main__":
    main()
