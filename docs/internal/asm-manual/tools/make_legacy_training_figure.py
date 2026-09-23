#!/usr/bin/env python3
"""Plot retained legacy logs and reported internal ladder; no model execution."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/legacy-training"


def main() -> None:
    names = ["round1.train_metrics.jsonl", "loss_components_by_generation.csv",
             "elo_trajectory.csv", "verification.json", "gate-result.v1.json"]
    inputs = {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
              for name in names}
    verification = json.loads((DATA / "verification.json").read_text())
    for source in verification["sources"]:
        assert inputs[source["copy"]] == source["sha256"]
    epochs = [json.loads(line) for line in
              (DATA / names[0]).read_text().splitlines() if line.strip()]
    with (DATA / names[1]).open(newline="") as stream:
        generations = list(csv.DictReader(stream))
    with (DATA / names[2]).open(newline="") as stream:
        elo = list(csv.DictReader(stream))
    assert [row["epoch"] for row in epochs] == [1, 2, 3, 4]
    assert [int(row["gen"]) for row in generations] == list(range(1, 32))
    assert [int(row["generation"]) for row in elo] == list(range(1, 31))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.hashsalt": "jope-legacy-history-v1"})
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    x = [row["epoch"] for row in epochs]
    for key, label, color in [("train_loss", "Training composite", "#246b8e"),
                              ("val_loss", "Validation composite", "#cb7743")]:
        axes[0].plot(x, [row[key] for row in epochs], "o-", label=label, color=color)
    axes[0].axvline(2.5, color="#777777", linestyle="--", linewidth=1)
    axes[0].set(title="Lineage A round 1: four logged epochs", xlabel="Epoch (phase B starts at 3)", ylabel="Composite loss")
    axes[0].set_xticks(x)
    axes[0].legend(frameon=False, loc="center right")
    axes[1].plot([int(row["gen"]) for row in generations],
                 [float(row["val_seq_nll"]) for row in generations], "o-", color="#246b8e", markersize=3)
    axes[1].set(title="Jope Prime: reported generation summaries", xlabel="Generation (1–31)", ylabel="Validation sequence NLL")
    axes[2].plot([int(row["generation"]) for row in elo],
                 [float(row["elo_top1"]) for row in elo], "o-", color="#318c89", markersize=3)
    axes[2].set(title="Internal self-play Elo: not a benchmark-valid ladder", xlabel="Generation (1–30; no generation-31 Elo row)", ylabel="Reported top-1 Elo")
    for ax in axes:
        ax.grid(alpha=.18)
        ax.set_axisbelow(True)
    fig.text(.03, .02, "Separate populations/objectives. Paper CSVs retain reported ladder values, not a replayed tournament.\nGeneration-31 summary matches epoch 3, not the final epoch. Higher Elo does not imply lower NLL.", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, 1))
    output = ROOT / "figures/training-legacy-round1-generations.svg"
    fig.savefig(output, metadata={"Date": None}, facecolor="white")
    plt.close(fig)
    receipt = {"schema": "jope.manual.legacy-figure/v1", "inputs": inputs,
               "output": str(output.relative_to(ROOT)),
               "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "interpolation": "Straight lines connect original summaries; no invented observations",
               "loss_generations": 31, "internal_elo_generations": 30, "round1_epochs": 4}
    (DATA / "figure-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
