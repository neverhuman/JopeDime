#!/usr/bin/env python3
"""Plot one captured window of logged adapter admission decisions."""

import csv
import hashlib
import json
from decimal import Decimal, getcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from matplotlib.ticker import MaxNLocator

getcontext().prec = 50
FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data/live_adapter_decisions.csv"
SOURCE = FIGURES / "data/live_adapter_decisions.source.json"
OUTPUT = FIGURES / "fig_fw_live_adapter_decisions.pdf"
PROVENANCE = FIGURES / "data/live_adapter_decisions.provenance.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads(SOURCE.read_text())
    assert sha(DATA) == source["csv_sha256"]
    with DATA.open() as stream:
        rows = list(csv.DictReader(stream))
    rounds = [int(row["round"]) for row in rows]
    capture = source["capture"]
    assert len(rows) == capture["rows"] == 20
    assert rounds == list(range(rounds[0], rounds[0] + len(rows)))
    assert [rounds[0], rounds[-1]] == capture["round_window"]
    assert capture["all20_candidate_and_decision_flags_matched"]
    assert capture["all20_summary_admission_retained_utility_matched"]
    assert capture["retained_state_contiguous"]
    candidates = [Decimal(row["candidate_utility"]) for row in rows]
    previous = [Decimal(row["previous_best_utility"]) for row in rows]
    retained = [Decimal(row["retained_utility"]) for row in rows]
    margins = [Decimal(row["candidate_minus_previous"]) for row in rows]
    accepted = [int(row["accepted"]) for row in rows]
    assert all(value.is_finite() for values in [candidates, previous, retained, margins]
               for value in values)
    assert all(flag in (0, 1) for flag in accepted)
    assert margins == [candidate-best for candidate, best in zip(candidates, previous)]
    assert retained == [candidate if flag else best
                        for candidate, best, flag in zip(candidates, previous, accepted)]
    assert retained[:-1] == previous[1:]
    assert sum(accepted) == capture["adapter_admissions"]
    assert len(rows)-sum(accepted) == capture["rejected"]
    assert sorted({str(value) for value in retained}) == capture["retained_utility_values"]

    font = Path(findfont("Liberation Serif", fallback_to_default=False))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif"],
        "font.size": 8.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    ax.axhline(0, color="#495057", ls="--", lw=0.8, label="score equality")
    for flag, marker, color, label in [
        (0, "x", "#7451b8", "rejected"), (1, "D", "#2b8a3e", "accepted"),
    ]:
        positions = [i for i, value in enumerate(accepted) if value == flag]
        if positions:
            ax.scatter([rounds[i] for i in positions], [float(margins[i]) for i in positions],
                       marker=marker, c=color, s=20, linewidths=0.8,
                       label=f"{label} ({len(positions)})", zorder=3)
    low, high = min(Decimal(0), min(margins)), max(Decimal(0), max(margins))
    pad = max(high-low, Decimal("0.001")) * Decimal("0.15")
    ax.set(xlabel="completed round", ylabel="candidate − pre-round best",
           xlim=(rounds[0]-0.7, rounds[-1]+0.7), ylim=(float(low-pad), float(high+pad)))
    ax.set_xticks([rounds[0], rounds[6], rounds[12], rounds[-1]])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2,
              fontsize=8, frameon=False, handlelength=1.1, columnspacing=0.7)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    facts = dict(
        schema="whitepaper.live-adapter-decisions-figure/v1",
        input_csv_sha256=sha(DATA), source_receipt_sha256=sha(SOURCE),
        generator_sha256=sha(Path(__file__)), figure_sha256=sha(OUTPUT),
        rows=len(rows), round_window=[rounds[0], rounds[-1]],
        adapter_admissions=sum(accepted), rejected=len(rows)-sum(accepted),
        retained_utility_values=sorted({str(value) for value in retained}),
        minimum_margin=str(min(margins)), maximum_margin=str(max(margins)),
        definitions=source["definitions"], scope=source["scope"],
        matplotlib_version=matplotlib.__version__,
        font_family="Liberation Serif", font_file_sha256=sha(font),
        figure_inches=[3.25, 2.2],
        reproduce="python3 figures/scripts/make_live_adapter_decisions_figure.py",
    )
    PROVENANCE.write_text(json.dumps(facts, indent=2)+"\n")
    print(json.dumps(facts))


if __name__ == "__main__":
    main()
