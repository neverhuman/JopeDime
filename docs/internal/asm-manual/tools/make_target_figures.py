#!/usr/bin/env python3
"""Plot historical development targets from retained comparison metadata only."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/target-history"
FIG = ROOT / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})


def read(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    summary = read("target-quality-1000-summary.csv")
    comparisons = read("target-quality-1000-comparisons.csv")
    assert len(summary) == 3 and len(comparisons) == 159
    arms = ["neural", "generic", "shuffled"]
    summary = [next(row for row in summary if row["arm"] == arm) for arm in arms]
    for row in summary:
        group = [r for r in comparisons if r["arm"] == row["arm"]]
        assert len(group) == int(row["repeated_comparisons"])
        assert all(int(r["completed_evaluations"]) == 1000 for r in group)
        outcomes = {"improved": 0, "unchanged": 0, "worsened": 0}
        for candidate in group:
            first, last = float(candidate["first_score"]), float(candidate["last_score"])
            key = "improved" if last < first else "worsened" if last > first else "unchanged"
            outcomes[key] += 1
        assert all(count == int(row[key]) for key, count in outcomes.items())
        assert statistics.median(float(r["last_relative_improvement"]) for r in group) == float(row["median_relative_improvement"])

    fig, ax = plt.subplots(figsize=(10, 4.8))
    left = [0] * 3
    for key, color in [("improved", "#318c89"), ("unchanged", "#ccd8e1"), ("worsened", "#cb7743")]:
        values = [int(row[key]) for row in summary]
        bars = ax.barh(range(3), values, left=left, color=color, label=key.capitalize())
        for bar, value in zip(bars, values):
            if value:
                ax.text(bar.get_x() + value/2, bar.get_y() + bar.get_height()/2,
                        str(value), va="center", ha="center")
        left = [a+b for a, b in zip(left, values)]
    ax.set_yticks(range(3), [f"{row['arm']} (n={row['repeated_comparisons']})" for row in summary])
    ax.invert_yaxis()
    ax.set_xlabel("Repeated recorded comparisons: first versus last retained score")
    ax.set_title("Historical GP targets improved on a subset of development comparisons")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    fig.text(.025, .025,
             "1,000 completed evaluations per retained search; seeds and elapsed time vary. Denominators differ by arm.\n"
             "Median relative improvement is zero in all three arms. These are adaptive development results, not a neural win rate.\n"
             "Source: target-quality-1000-summary.csv and all 159 comparison rows.", fontsize=9)
    fig.tight_layout(rect=(0, .19, 1, 1))
    fig.savefig(FIG / "target-quality-progress-counts.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rows = read("target-quality-pine-chromium-cut1.csv")
    assert len(rows) == 213
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, arm, color in zip(axes, arms, ["#246b8e", "#318c89", "#cb7743"]):
        group = sorted((row for row in rows if row["arm"] == arm), key=lambda row: int(row["generation"]))
        assert [int(row["generation"]) for row in group] == list(range(71))
        values = [float(row["score"]) for row in group]
        ax.plot(range(71), values, color=color, linewidth=1.2, marker=".", markersize=2)
        ax.set_title(arm.capitalize())
        ax.set_xlabel("Historical generation")
        ax.set_xticks([0, 20, 40, 60, 70])
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Reported best_validation_rmse\n(historical evaluator units)")
    fig.suptitle("Illustrative pine-chromium cut 1: the final best score is shared by all three arms", fontweight="bold")
    fig.text(.025, .02,
             "Post-hoc example selected for the largest observed neural relative improvement; not a representative random sample.\n"
             "All 213 original search scores shown. Adaptive teacher validation; checkpoint exposure unknown.\n"
             "Recorded comparison fields match; RNG and wall time vary. Source: target-quality-pine-chromium-cut1.csv.", fontsize=9)
    fig.tight_layout(rect=(0, .2, 1, .94))
    fig.savefig(FIG / "target-quality-pine-example.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    names = ["target-quality-1000-summary.csv", "target-quality-1000-comparisons.csv", "target-quality-pine-chromium-cut1.csv"]
    outputs = ["target-quality-progress-counts.png", "target-quality-pine-example.png"]
    receipt = {"schema": "jope.manual-target-figures/v1",
               "inputs": {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in names},
               "outputs": {name: hashlib.sha256((FIG/name).read_bytes()).hexdigest() for name in outputs},
               "comparisons_recomputed": 159, "original_example_scores": 213,
               "new_fits": 0, "equal_compute_or_neural_causality_claimed": False}
    (DATA / "figure-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Two target figures generated; all comparison counts independently recomputed.")


if __name__ == "__main__":
    main()
