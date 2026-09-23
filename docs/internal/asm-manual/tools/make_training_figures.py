#!/usr/bin/env python3
"""Plot retained training observations; never load models or run training."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/training-history"
FIG = ROOT / "figures"
BLUE, ORANGE, TEAL = "#246b8e", "#cb7743", "#318c89"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})
OUTPUTS: list[str] = []
INPUTS: set[str] = set()


def read(name: str) -> list[dict[str, str]]:
    INPUTS.add(name)
    with (DATA / name).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(math.isfinite(float(row["value"])) for row in rows)
    return rows


def save(fig, name: str, caption: str) -> None:
    fig.text(.025, .018, caption, fontsize=9, va="bottom")
    fig.tight_layout(rect=(0, .115, 1, .96))
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    OUTPUTS.append(name)


def decorate(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.15)
    ax.set_axisbelow(True)


def parent_boundaries(ax) -> None:
    # Model ancestry changes; every block still starts fresh AdamW.
    ax.axvline(60.5, color="#777777", linestyle="--", linewidth=.9, zorder=1)
    ax.axvline(73.5, color=ORANGE, linestyle="--", linewidth=1.0, zorder=1)


def verify_parent_regimes(rows: list[dict[str, str]]) -> None:
    parents: dict[int, str] = {}
    for row in rows:
        cycle = int(row["cycle"])
        parent = row["parent_state_sha256"]
        assert parents.setdefault(cycle, parent) == parent
    assert set(parents) == set(range(108))
    assert parents[0] == "20dca6708ebcffc433b7a63fea3df868c1003adefc19826637367bc4a2fa64fe"
    assert all(parents[c] == "61ba604345641ec4a0b9103886d5b03ed86a1cad475f79ad582bb5e0cb5cb75c"
               for c in range(61, 74))
    assert all(parents[c] == "c2f229c9b8c582e11e0e62ec011a9ccb066820fbcee5e2da9d0ad5d63780160e"
               for c in range(74, 108))
    checkpoints = {int(row["cycle"]): row["checkpoint_state_sha256"] for row in rows}
    assert all(parents[c] == checkpoints[c - 1] for c in range(1, 61))


def historical_blocks() -> None:
    rows = read("block-endpoint-metrics.csv")
    verify_parent_regimes(rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric, label in zip(axes.flat,
                                ["loss", "sequence_nll", "ranking_bt", "runtime_huber"],
                                ["Composite objective", "Sequence NLL", "Ranking BT", "Runtime Huber"]):
        for endpoint, color, marker in [("first", ORANGE, "x"), ("last", BLUE, ".")]:
            selected = sorted((r for r in rows if r["metric"] == metric and r["endpoint"] == endpoint),
                              key=lambda r: int(r["cycle"]))
            assert len(selected) == 108
            ax.scatter([int(r["cycle"]) for r in selected], [float(r["value"]) for r in selected],
                       s=18, alpha=.8, color=color, marker=marker,
                       label="Step 1 minibatch" if endpoint == "first" else "Step 3,000 minibatch")
        ax.set_title(label)
        decorate(ax, "Historical block / cycle", "Recorded loss")
        parent_boundaries(ax)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("108 historical blocks: observed endpoints, not a continuous optimizer", fontweight="bold")
    save(fig, "training-historical-endpoints.png",
         "Fresh AdamW each block. Parents: cycles 1–60 previous; 61–73 cycle 59; 74–107 recovery (cycle 0 separate seed).\n"
         "Gray/orange boundaries mark changed starting parents at 61/74, not continuous learning events.\n"
         "Different minibatch endpoints only; source: block-endpoint-metrics.csv, audit captured 11 September 2026.")

    rows = read("block-adaptive-validation.csv")
    verify_parent_regimes(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    for ax, metric, label in zip(axes, ["sequence_nll", "ranking_bt"], ["Sequence NLL", "Ranking BT"]):
        selected = sorted((r for r in rows if r["metric"] == metric), key=lambda r: int(r["cycle"]))
        assert len(selected) == 108 and all(r["independent_evaluation"] == "False" for r in selected)
        ax.scatter([int(r["cycle"]) for r in selected], [float(r["value"]) for r in selected],
                   s=18, color=BLUE)
        ax.set_title(label)
        decorate(ax, "Historical block / cycle", "Adaptive validation loss")
        parent_boundaries(ax)
    fig.suptitle("Historical validation observations — changing corpus and monitor", fontweight="bold")
    save(fig, "training-adaptive-validation.png",
         "Parents: cycles 1–60 previous; 61–73 cycle 59; 74–107 recovery (cycle 0 separate seed).\n"
         "Gray/orange boundaries mark changed parents at 61/74; the rollback jump is not continuous learning.\n"
         "Changing corpus/monitor observations only. Source: block-adaptive-validation.csv; all identities retained.")


def fixed_monitor() -> None:
    rows = read("fixed-monitor-metrics.csv")
    assert len({r["corpus_sha256"] for r in rows}) == len({r["panel_sha256"] for r in rows}) == 1
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric, label in zip(axes.flat,
                                ["sequence_nll", "ranking_bt", "ranking_listwise", "runtime_huber"],
                                ["Sequence NLL", "Ranking BT", "Listwise cross-entropy", "Runtime Huber"]):
        selected = [r for r in rows if r["metric"] == metric]
        recovery = [float(r["value"]) for r in selected if r["monitor_group"] == "recovery"]
        assert len(recovery) == 2 and recovery[0] == recovery[1]
        values = [recovery[0]] + [float(next(r["value"] for r in selected if r["monitor_group"] == group))
                                  for group in ("cycle25", "last")]
        bars = ax.bar(["Recovery", "Cycle 25", "Cycle 107"], values, color=[TEAL, ORANGE, BLUE])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.6f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(values)*1.17)
        ax.set_title(label)
        decorate(ax, "Checkpoint (categorical)", "Recorded loss — lower is better")
    fig.suptitle("Same corpus and adaptive monitor: later checkpoints did not improve every loss", fontweight="bold")
    save(fig, "training-fixed-monitor.png",
         "Recovery was measured under both relevant executables with identical outputs; each challenger has its control.\n"
         "2,048 examples per evaluation. Adaptive teaching monitor, not sealed or independent evaluation.\n"
         "Source: fixed-monitor-metrics.csv; original record and executable hashes retained.")


def gen31() -> None:
    rows = read("gen31-epoch-metrics.csv")

    def series(metric: str, scope: str = "epoch_summary"):
        selected = sorted((r for r in rows if r["metric"] == metric and r["scope"] == scope),
                          key=lambda r: int(r["epoch"]))
        assert len(selected) == 14
        return [int(r["epoch"]) for r in selected], [float(r["value"]) for r in selected]

    def phase(ax):
        ax.axvline(6.5, color="#888888", linestyle="--", linewidth=.8)
        ax.set_xticks([1, 3, 5, 7, 9, 11, 14])
        ax.set_xlabel("Epoch (phase A: 1–6; phase B: 7–14)", fontsize=9)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric, title in zip(axes.flat,
                               ["total", "seq_nll", "reward_bt", "runtime_huber"],
                               ["Composite loss (changing learned weights)", "Sequence NLL", "Reward BT", "Runtime Huber"]):
        for split, color in [("train", TEAL), ("val", ORANGE)]:
            x, y = series(f"{split}_loss") if metric == "total" else series(metric, f"{split}_loss_components")
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.3, color=color,
                    label="Training" if split == "train" else "Validation")
        ax.set_title(title)
        decorate(ax, "Epoch", "Recorded loss")
        phase(ax)
    axes[0, 0].legend(loc="center right", fontsize=9)
    fig.suptitle("Gen31 / policy_v3.2: fourteen retained epoch summaries", fontweight="bold")
    save(fig, "training-gen31-loss.png",
         "Separate model lineage from accepted step-60000. No event date is embedded in this log.\n"
         "Phase boundary and changing composite weights limit comparisons; raw component curves remain separate.\n"
         "Source: gen31-epoch-metrics.csv and retained original train_metrics.jsonl; no smoothed or invented points.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    specifications = [
        ("val_pairwise_win_accuracy", "Pairwise win accuracy", "Higher is better"),
        ("val_top1_historical_regret", "Top-1 historical regret", "Lower is better"),
        ("val_conditional_advantage", "Conditional advantage", "Higher is better"),
        ("val_reward_cv_correlation", "Reward / recorded CV correlation", "Diagnostic correlation"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.flat, specifications):
        x, y = series(metric)
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.3, color=BLUE)
        ax.set_title(title)
        decorate(ax, "Epoch", ylabel)
        phase(ax)
    fig.suptitle("Gen31 policy diagnostics: quality signals beyond training loss", fontweight="bold")
    save(fig, "training-gen31-quality.png",
         "Diagnostics against retained teacher labels, not a measured improvement in a new GP campaign.\n"
         "No independent-family or serving-promotion claim follows from these curves.\n"
         "Source: gen31-epoch-metrics.csv; exact metric definitions and limitations are in the training-history chapter.")


def carried_checkpoints() -> None:
    rows = read("a100-checkpoint-validation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, metric, title in zip(axes, ["sequence_nll", "ranking_bt"], ["Sequence NLL", "Ranking BT"]):
        selected = [r for r in rows if r["metric"] == metric]
        assert len(selected) == 78
        ax.scatter([int(r["step"]) for r in selected], [float(r["value"]) for r in selected],
                   s=20, color=BLUE, alpha=.65, label="Retained checkpoint")
        accepted = [r for r in selected if r["is_accepted_serving60000"] == "True"]
        assert len(accepted) == 1
        ax.scatter([int(accepted[0]["step"])], [float(accepted[0]["value"])],
                   color=ORANGE, marker="*", s=170, zorder=5, label="Accepted step-60000")
        ax.set_title(title)
        decorate(ax, "Recorded checkpoint optimizer step", "Adaptive validation loss")
        ax.ticklabel_format(style="plain", axis="x")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Later A100 lineage: carried optimizer, 78 retained checkpoints", fontweight="bold")
    save(fig, "training-a100-checkpoints.png",
         "22 corpus identities; changing adaptive monitors. This is not one fixed-evaluation learning curve.\n"
         "Checkpoint step and corpus identity remain separate; the highest step is not automatically the serving model.\n"
         "Source: a100-checkpoint-validation.csv; state/model/optimizer/parent hashes retained.")

    rows = read("rtx60100-step-metrics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric, title in zip(axes.flat,
                                ["loss", "sequence_nll", "ranking_bt", "contextfree_bt"],
                                ["Composite objective", "Sequence NLL", "Ranking BT", "Context-free ablation BT"]):
        selected = sorted((r for r in rows if r["metric"] == metric), key=lambda r: int(r["step"]))
        assert [int(r["step"]) for r in selected] == list(range(60001, 60101))
        ax.plot([int(r["step"]) for r in selected], [float(r["value"]) for r in selected],
                color=BLUE, linewidth=1, marker=".", markersize=2)
        ax.axvline(60006.5, color=ORANGE, linestyle="--", linewidth=.9)
        ax.set_title(title)
        decorate(ax, "Carried optimizer step", "Recorded minibatch loss")
        ax.ticklabel_format(style="plain", useOffset=False, axis="x")
        ax.set_xticks([60001, 60050, 60100])
    fig.suptitle("RTX 3090: 100 actual continuation steps on unchanged corpus", fontweight="bold")
    save(fig, "training-rtx60100-steps.png",
         "14 September 2026; world 1 × microbatch 256 × accumulation 8 = 2,048. Dashed line marks resumed attempt.\n"
         "204,800 committed program-example presentations; zero fresh presentations. Hardware qualification, not promotion.\n"
         "Source: rtx60100-step-metrics.csv; all 100 logged steps shown, without smoothing or reconstructed points.")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    historical_blocks()
    fixed_monitor()
    gen31()
    carried_checkpoints()
    receipt = {"schema": "jope.manual-training-figures/v1",
               "inputs": {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in sorted(INPUTS)},
               "outputs": {name: hashlib.sha256((FIG/name).read_bytes()).hexdigest() for name in OUTPUTS},
               "matplotlib_version": matplotlib.__version__,
               "fits": 0, "training_runs": 0, "interpolated_missing_observations": False}
    (DATA / "figure-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Generated {len(OUTPUTS)} figures from retained training observations.")


if __name__ == "__main__":
    main()
