#!/usr/bin/env python3
"""Render one captured retained-best trajectory and its trailing rejection run."""

import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont

FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data/retained_best_plateau.csv"
SOURCE = FIGURES / "data/retained_best_plateau.source.json"
OUTPUT = FIGURES / "fig_fw_retained_best_plateau.pdf"
PROVENANCE = FIGURES / "data/retained_best_plateau.provenance.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads(SOURCE.read_text())
    assert sha(DATA) == source["csv_sha256"]
    capture = source["capture"]
    with DATA.open() as stream:
        rows = list(csv.DictReader(stream))
    rounds = [int(row["round"]) for row in rows]
    assert len(rows) == capture["rows"]
    assert rounds == list(range(rounds[0], rounds[-1] + 1))
    assert [rounds[0], rounds[-1]] == capture["round_window"]
    candidates = [Decimal(row["candidate_utility"]) for row in rows]
    previous = [Decimal(row["previous_best_utility"]) for row in rows]
    retained = [Decimal(row["retained_utility"]) for row in rows]
    accepted = [int(row["accepted"]) for row in rows]
    assert all(value.is_finite() for values in [candidates, previous, retained]
               for value in values)
    assert all(flag in (0, 1) for flag in accepted)
    assert accepted == [int(candidate >= best)
                        for candidate, best in zip(candidates, previous)]
    assert retained == [candidate if flag else best
                        for candidate, best, flag in zip(candidates, previous, accepted)]
    assert previous[1:] == retained[:-1]
    gains = [r for r, c, p, flag in zip(rounds, candidates, previous, accepted)
             if flag and c > p]
    assert gains and gains[-1] == capture["last_strict_gain_round"]
    gain = gains[-1]
    assert rounds[0] == gain - 1
    admitted = [r for r, flag in zip(rounds, accepted) if flag]
    last_admitted = admitted[-1]
    assert last_admitted == capture["last_admitted_round"]
    rejection_rounds = [r for r, flag in zip(rounds, accepted)
                        if r > last_admitted and not flag]
    assert rejection_rounds == list(range(last_admitted + 1, rounds[-1] + 1))
    assert len(rejection_rounds) == capture["trailing_rejections"]
    assert [last_admitted + 1, rounds[-1]] == capture["trailing_rejection_rounds"]
    assert retained[-1] == Decimal(capture["retained_utility"])
    assert int(rows[-1]["best_round"]) == capture["best_round"]
    assert all(value == retained[-1] for r, value in zip(rounds, retained) if r >= gain)
    assert capture["retained_state_contiguous"]
    assert capture["all20_candidate_and_decision_flags_matched"]
    assert capture["all20_summary_admission_retained_utility_matched"]

    font = Path(findfont("Liberation Serif", fallback_to_default=False))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif"],
        "font.size": 8.5, "pdf.fonttype": 42, "path.simplify": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(3.25, 2.0))
    fig.subplots_adjust(left=0.30, right=0.97, bottom=0.24, top=0.93)
    ax.step(rounds, [float(v) for v in retained], where="post", color="#1971c2", lw=1.2)
    ax.scatter([gain], [float(retained[rounds.index(gain)])], marker="*",
               s=48, color="#c77c02", zorder=4)
    ax.vlines(rejection_rounds, 0.025, 0.075, transform=ax.get_xaxis_transform(),
              colors="#7451b8", linewidth=0.45, alpha=0.85)
    ax.text((last_admitted + 1 + rounds[-1]) / 2, 0.12,
            f"{len(rejection_rounds):,} consecutive rejections",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5)
    span = retained[-1] - retained[0]
    assert span > 0
    padding = max(1, (rounds[-1] - rounds[0]) * 0.018)
    ax.set(xlabel="completed round", ylabel="retained validation utility",
           xlim=(rounds[0] - padding, rounds[-1] + padding),
           ylim=(float(retained[0] - span * Decimal("0.22")),
                 float(retained[-1] + span * Decimal("0.28"))))
    ax.set_yticks([float(retained[0]), float(retained[-1])],
                 labels=[f"{retained[0]:.7f}", f"{retained[-1]:.7f}"])
    ax.set_xticks([gain, (gain + rounds[-1]) // 2, rounds[-1]])
    ax.tick_params(labelsize=8.5)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    facts = dict(
        schema="whitepaper.retained-best-plateau-figure/v1",
        input_csv_sha256=sha(DATA), source_receipt_sha256=sha(SOURCE),
        generator_sha256=sha(Path(__file__)), figure_sha256=sha(OUTPUT),
        rows=len(rows), round_window=[rounds[0], rounds[-1]],
        last_strict_gain_round=gain, last_admitted_round=last_admitted,
        trailing_rejection_rounds=[last_admitted + 1, rounds[-1]],
        trailing_rejections=len(rejection_rounds),
        final_retained_utility=str(retained[-1]), best_round=capture["best_round"],
        baseline_retained_utility=str(retained[0]),
        transformations="Every logged retained score plotted in order as a post-decision step; one rug mark per trailing rejection. No smoothing, cumulative maximum, clipping or outcome selection.",
        scope=source["scope"], source_binding=source["source_binding"],
        matplotlib_version=matplotlib.__version__, font_family="Liberation Serif",
        font_file_sha256=sha(font), figure_inches=[3.25, 2.0],
        minimum_text_size_pt=8.5, fixed_canvas_no_tight_crop=True,
        reproduce="python3 figures/scripts/make_retained_best_plateau_figure.py",
    )
    PROVENANCE.write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts))


if __name__ == "__main__":
    main()
