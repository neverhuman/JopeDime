#!/usr/bin/env python3
"""Render matched regression Elo leaderboards with and without jope-dime."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "manifests/panel-analysis.v1.json"
BEST = (
    ROOT
    / "runs/test-adaptive-golden42-v1-extension-g20-g29-islands4x30"
    / "golden42-best-solutions.v1.json"
)
OUTPUT_WITH_JOPE = ROOT / "artifacts/jope-dime-elo-ranking-16x9-600dpi.png"
OUTPUT_BASELINES = ROOT / "artifacts/regression-elo-ranking-baselines-16x9-600dpi.png"

BASELINES = [
    {
        "model": "TabFM ensemble",
        "coverage": 2089,
        "common": 1973,
        "mean": 0.083834,
        "median": 0.060730,
        "elo": 1260.94,
        "wtl": "966 / 979 / 28",
    },
    {
        "model": "TabPFNv3",
        "coverage": 2322,
        "common": 1973,
        "mean": 0.085564,
        "median": 0.062214,
        "elo": 1223.71,
        "wtl": "631 / 1,317 / 25",
    },
    {
        "model": "Fable5",
        "coverage": 2186,
        "common": 1973,
        "mean": 0.094130,
        "median": 0.074226,
        "elo": 1047.04,
        "wtl": "141 / 1,784 / 48",
    },
    {
        "model": "Grok4p5",
        "coverage": 2326,
        "common": 1973,
        "mean": 0.096836,
        "median": 0.077302,
        "elo": 970.27,
        "wtl": "106 / 1,757 / 110",
    },
    {
        "model": "OpenAI GPT-5.6-sol",
        "coverage": 2190,
        "common": 1973,
        "mean": 0.104877,
        "median": 0.086714,
        "elo": 846.45,
        "wtl": "88 / 1,652 / 233",
    },
    {
        "model": "Kimik3",
        "coverage": 2326,
        "common": 1973,
        "mean": 0.132118,
        "median": 0.119819,
        "elo": 651.59,
        "wtl": "41 / 403 / 1,529",
    },
]


def score(left: float, right: float, tolerance: float = 1e-12) -> float:
    if abs(left - right) <= tolerance:
        return 0.5
    return 1.0 if left < right else 0.0


def batch_elo(
    rows: list[dict[str, float]],
    models: list[str],
    rounds: int = 60,
    k: float = 64.0,
    normalize: bool = True,
) -> dict[str, float]:
    ratings = {model: 1000.0 for model in models}
    divisor = max(1, len(models) - 1) if normalize else 1
    for _ in range(rounds):
        snapshot = ratings.copy()
        deltas = {model: 0.0 for model in models}
        for left_index, left in enumerate(models):
            for right in models[left_index + 1 :]:
                actual = sum(score(row[left], row[right]) for row in rows) / len(rows)
                expected = 1.0 / (
                    1.0 + 10.0 ** ((snapshot[right] - snapshot[left]) / 400.0)
                )
                delta = k * (actual - expected) / divisor
                deltas[left] += delta
                deltas[right] -= delta
        for model in models:
            ratings[model] += deltas[model]
    return ratings


def evolved_jope_row() -> tuple[dict, float]:
    panel = json.loads(PANEL.read_text())
    best = json.loads(BEST.read_text())
    selected = {
        solution["dataset_hash"]: float(solution["selected_rmse"])
        for solution in best["solutions"]
    }
    baseline_models = [row["label"] for row in panel["model_rows"]]
    panel_rows = []
    for dataset in panel["datasets"]:
        row = {model: float(dataset["rmse"][model]) for model in baseline_models}
        row["jope-dime"] = selected[dataset["dataset_hash"]]
        panel_rows.append(row)

    target = {row["label"]: float(row["panel_elo"]) for row in panel["model_rows"]}
    variants = [
        batch_elo(panel_rows, baseline_models, normalize=True),
        batch_elo(panel_rows, baseline_models, normalize=False),
    ]
    baseline_ratings = min(
        variants,
        key=lambda ratings: sum(
            (ratings[model] - target[model]) ** 2 for model in baseline_models
        ),
    )
    baseline_error = math.sqrt(
        sum((baseline_ratings[model] - target[model]) ** 2 for model in baseline_models)
        / len(baseline_models)
    )

    expanded = batch_elo(panel_rows, baseline_models + ["jope-dime"], normalize=True)
    full = {row["model"]: row["elo"] for row in BASELINES}
    offset = sum(full[model] - expanded[model] for model in baseline_models) / len(
        baseline_models
    )
    estimate = expanded["jope-dime"] + offset
    values = np.array(list(selected.values()), dtype=float)
    wins = sum(
        solution["benchmark_status"] == "surpassed_external_benchmark"
        for solution in best["solutions"]
    )
    jope = {
        "model": "jope-dime",
        "coverage": 42,
        "common": 42,
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "elo": float(estimate),
        "wtl": f"{wins} / 0 / {42 - wins}",
        "estimate": True,
    }
    return jope, baseline_error


def add_round_bar(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    glow: bool = False,
) -> None:
    if glow:
        for expansion, alpha in [(0.20, 0.055), (0.11, 0.09), (0.055, 0.14)]:
            ax.add_patch(
                FancyBboxPatch(
                    (x - expansion * 25, y - height * expansion / 2),
                    width + expansion * 50,
                    height * (1 + expansion),
                    boxstyle=f"round,pad=0,rounding_size={height * 0.50}",
                    linewidth=0,
                    facecolor="#2ee8c5",
                    alpha=alpha,
                    zorder=1,
                )
            )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={height * 0.48}",
            linewidth=0,
            facecolor=color,
            zorder=3,
        )
    )


def render_variant(
    rows: list[dict],
    output: Path,
    baseline_error: float,
    include_jope: bool,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#07111f",
            "axes.facecolor": "#07111f",
            "savefig.facecolor": "#07111f",
            "text.color": "#f4f8ff",
        }
    )
    fig = plt.figure(figsize=(16, 9), dpi=600)
    table_ax = fig.add_axes([0.055, 0.13, 0.455, 0.70])
    bar_ax = fig.add_axes([0.545, 0.13, 0.405, 0.70])
    for ax in (table_ax, bar_ax):
        ax.set_ylim(-0.55, len(rows) - 0.05)
        ax.axis("off")
    table_ax.set_xlim(0, 1)
    bar_ax.set_xlim(600, 1645)

    gradient = np.linspace(0, 1, 700)
    background = np.zeros((700, 1200, 3))
    top = np.array(mpl.colors.to_rgb("#07111f"))
    bottom = np.array(mpl.colors.to_rgb("#0b1d31"))
    for index, value in enumerate(gradient):
        background[index, :, :] = top * (1 - value) + bottom * value
    bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
    bg.imshow(background, aspect="auto", extent=[0, 1, 0, 1], origin="upper")
    bg.axis("off")

    fig.text(
        0.058,
        0.925,
        "REGRESSION MODEL ELO RANKING",
        fontsize=27,
        fontweight="bold",
        color="#f7fbff",
        ha="left",
        va="center",
    )
    fig.text(
        0.059,
        0.878,
        (
            "Canonical common-cohort Elo  •  lower RMSE wins  •  evolved leader highlighted"
            if include_jope
            else "Canonical six-model common-cohort ranking  •  lower RMSE wins"
        ),
        fontsize=11.8,
        color="#91a8bf",
        ha="left",
        va="center",
    )
    fig.text(
        0.942,
        0.916,
        "ELO",
        fontsize=10.5,
        fontweight="bold",
        color="#5f7891",
        ha="right",
    )
    fig.text(
        0.942,
        0.886,
        "RATING",
        fontsize=10.5,
        fontweight="bold",
        color="#5f7891",
        ha="right",
    )

    floor = 600.0
    baseline_colors = ["#5a9dff", "#477fda", "#687995", "#586b84", "#4d6078", "#42546b"]
    y_positions = list(reversed(np.arange(len(rows), dtype=float)))
    for rank, (row, y) in enumerate(zip(rows, y_positions), start=1):
        is_jope = row["model"] == "jope-dime"
        if is_jope:
            table_ax.axhspan(y - 0.46, y + 0.46, color="#123a42", alpha=0.34, zorder=0)
            bar_ax.axhspan(y - 0.46, y + 0.46, color="#123a42", alpha=0.34, zorder=0)
        elif rank % 2 == 0:
            table_ax.axhspan(y - 0.45, y + 0.45, color="#ffffff", alpha=0.018, zorder=0)
            bar_ax.axhspan(y - 0.45, y + 0.45, color="#ffffff", alpha=0.018, zorder=0)

        circle_color = "#e3b65b" if rank == 1 else "#17304a"
        table_ax.scatter(
            [0.042],
            [y],
            s=570 if rank == 1 else 480,
            color=circle_color,
            edgecolors="#ffd98c" if rank == 1 else "#294964",
            linewidths=1.3,
            zorder=5,
        )
        table_ax.text(
            0.042,
            y,
            str(rank),
            fontsize=12.5,
            fontweight="bold",
            color="#07111f" if rank == 1 else "#dce9f6",
            ha="center",
            va="center",
            zorder=6,
        )
        table_ax.text(
            0.095,
            y + 0.155,
            row["model"],
            fontsize=15.8 if is_jope else 14.2,
            fontweight="bold" if is_jope else "semibold",
            color="#ffffff" if is_jope else "#dce7f3",
            ha="left",
            va="center",
            zorder=7,
            clip_on=True,
        )
        coverage = f'{row["coverage"]:,}'
        common = f'{row["common"]:,}'
        detail = (
            f"COVERAGE  {coverage}    COMMON  {common}    "
            f"MEAN  {row['mean']:.6f}    MEDIAN  {row['median']:.6f}    "
            f"W/T/L  {row['wtl']}"
        )
        table_ax.text(
            0.096,
            y - 0.175,
            detail,
            fontsize=7.15,
            color="#8fa6bb" if not is_jope else "#9fded3",
            ha="left",
            va="center",
            zorder=7,
            clip_on=True,
        )

        baseline_index = next(
            index for index, baseline in enumerate(BASELINES) if baseline["model"] == row["model"]
        ) if not is_jope else None
        add_round_bar(
            bar_ax,
            floor,
            y - 0.115,
            row["elo"] - floor,
            0.23,
            "#27dfbf" if is_jope else baseline_colors[baseline_index],
            glow=is_jope,
        )
        bar_ax.text(
            row["elo"] + 18,
            y,
            f'{row["elo"]:,.2f}',
            fontsize=16.7 if is_jope else 14.5,
            fontweight="bold",
            color="#5df2d6" if is_jope else "#f0f5fb",
            ha="left",
            va="center",
            zorder=8,
        )
        if is_jope:
            bar_ax.text(
                row["elo"] + 18,
                y - 0.255,
                "PANEL-CALIBRATED EST.",
                fontsize=6.8,
                fontweight="bold",
                color="#69bfae",
                ha="left",
                va="center",
            )

    if include_jope:
        fig.text(
            0.059,
            0.062,
            "jope-dime uses the best evolved Golden42 solution per dataset (35 strict leader wins after generation 29).",
            fontsize=9.2,
            color="#b8c9d8",
            ha="left",
        )
        fig.text(
            0.059,
            0.036,
            (
                "Its Elo is an explicitly labeled 42-dataset panel estimate, offset-calibrated to the six canonical full-cohort ratings; "
                f"baseline panel reconstruction RMSE {baseline_error:.3f} Elo."
            ),
            fontsize=7.9,
            color="#71889e",
            ha="left",
        )
    else:
        fig.text(
            0.059,
            0.052,
            "Coverage, common cases, RMSE, Elo, and win/tie/loss values exactly reproduce the supplied six-model ranking.",
            fontsize=8.8,
            color="#8da3b7",
            ha="left",
        )
    fig.text(
        0.942,
        0.041,
        "600 DPI  •  16:9",
        fontsize=8.5,
        fontweight="bold",
        color="#597188",
        ha="right",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def render() -> None:
    jope, baseline_error = evolved_jope_row()
    with_jope = sorted([jope, *BASELINES], key=lambda row: row["elo"], reverse=True)
    baselines = sorted(BASELINES, key=lambda row: row["elo"], reverse=True)
    render_variant(with_jope, OUTPUT_WITH_JOPE, baseline_error, include_jope=True)
    render_variant(baselines, OUTPUT_BASELINES, baseline_error, include_jope=False)
    print(
        json.dumps(
            {
                "outputs": [str(OUTPUT_BASELINES), str(OUTPUT_WITH_JOPE)],
                "width_px": 9600,
                "height_px": 5400,
                "dpi": 600,
                "jope_elo_estimate": jope["elo"],
                "jope_mean_rmse": jope["mean"],
                "jope_median_rmse": jope["median"],
                "baseline_panel_reconstruction_rmse": baseline_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    render()
