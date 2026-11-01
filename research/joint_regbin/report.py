"""Task-separated, family-paired Elo evidence; unavailable evidence has no verdict."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from .driver import append_score_store, collect, output_location, write_new
from .protocol import ARMS, TASKS, digest


def elo(records, anchor):
    """Zermelo MM with one fixed half-win/half-loss prior per pair.

    Each family contributes equal weight, divided among its contexts and paired
    seeds. Higher oriented score wins, with the existing tracker's 1e-4 tie band.
    Study-local anchor=1500; these are not the published eight-method ratings.
    """
    names = sorted(records[0]["scores"])
    if anchor not in names or any(sorted(row["scores"]) != names for row in records):
        raise ValueError("complete common Elo contender panel required")
    n = len(names)
    counts = Counter(row["family"] for row in records)
    wins = [[0.5 if a != b else 0.0 for b in range(n)] for a in range(n)]
    games = [[1.0 if a != b else 0.0 for b in range(n)] for a in range(n)]
    for row in records:
        weight = 1 / counts[row["family"]]
        for a in range(n):
            for b in range(a + 1, n):
                delta = row["scores"][names[a]] - row["scores"][names[b]]
                if not math.isfinite(delta):
                    raise ValueError("nonfinite Elo observation")
                outcome = 0.5 if abs(delta) <= 1e-4 else float(delta > 0)
                wins[a][b] += weight * outcome
                wins[b][a] += weight * (1 - outcome)
                games[a][b] += weight
                games[b][a] += weight
    strength = [1.0] * n
    for _ in range(400):
        updated = [
            sum(wins[a])
            / sum(games[a][b] / (strength[a] + strength[b]) for b in range(n) if b != a)
            for a in range(n)
        ]
        scale = updated[names.index(anchor)]
        updated = [value / scale for value in updated]
        if max(abs(math.log(a / b)) for a, b in zip(updated, strength)) < 1e-10:
            strength = updated
            break
        strength = updated
    return {
        name: 1500 + 400 * math.log10(value) for name, value in zip(names, strength)
    }


def paired_records(rows, task, role, round_number):
    grouped = {}
    for row in rows:
        if (row["task"], row["role"], row["round"]) != (task, role, round_number):
            continue
        key = row["id"], row["seed"]
        item = grouped.setdefault(
            key, dict(family=row["family"], scores={}, baseline=row["baseline"])
        )
        if item["family"] != row["family"] or item["baseline"] != row["baseline"]:
            raise ValueError("unpaired family/baseline identity")
        if row["arm"] in item["scores"]:
            raise ValueError("duplicate arm score")
        oriented = row["value"] * (-1 if task == "regression" else 1)
        item["scores"][row["arm"]] = oriented
    for item in grouped.values():
        if set(item["scores"]) != set(ARMS):
            raise ValueError("missing paired arm")
        item["scores"]["ridge"] = item.pop("baseline") * (
            -1 if task == "regression" else 1
        )
    if not grouped:
        raise ValueError("empty task/cohort evaluation")
    return list(grouped.values())


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = math.floor(position)
    return values[lo] + (values[math.ceil(position)] - values[lo]) * (position - lo)


def comparison(records, task, repetitions, seed):
    anchor = "reg-only" if task == "regression" else "bin-only"
    point = elo(records, anchor)
    families = defaultdict(list)
    for row in records:
        families[row["family"]].append(row)
    if len(families) < 5:
        raise ValueError(
            "at least five families required for grouped confidence intervals"
        )
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(repetitions):
        drawn = []
        for index in range(len(families)):
            family = rng.choice(sorted(families))
            # A re-sampled copy remains a separate equal-weight cluster.
            drawn.extend(dict(row, family=str(index)) for row in families[family])
        ratings = elo(drawn, anchor)
        for arm in ARMS:
            samples[arm].append(ratings[arm] - ratings[anchor])
    return {
        arm: dict(
            elo=point[arm],
            delta=point[arm] - point[anchor],
            low=quantile(samples[arm], 0.025),
            high=quantile(samples[arm], 0.975),
        )
        for arm in ARMS
    }


def decide(evaluations):
    """A split requires the SAME task to fail at two consecutive evaluations."""
    if len(evaluations) < 2 or any(
        set(evaluation) != set(TASKS) for evaluation in evaluations
    ):
        raise ValueError("two complete consecutive task evaluations required")
    for task in TASKS:
        if all(
            evaluation[task]["delta"] < -25 and evaluation[task]["high"] < 0
            for evaluation in evaluations[-2:]
        ):
            return "SPLIT_RECOMMENDED"
    if all(
        value["delta"] >= -25 or value["low"] <= 0 <= value["high"]
        for value in evaluations[-1].values()
    ):
        return "JOINT_OK"
    return "MITIGATE_AND_RERUN"


def build_report(plan, rows):
    config = plan["config"]
    panel = {
        task: len(
            {
                row["dataset"]
                for row in plan["contexts"]
                if row["task"] == task and row["role"] == "panel"
            }
        )
        for task in TASKS
    }
    if panel != dict(regression=240, binary=240):
        raise ValueError("panel_v1 requires exactly 240 eligible datasets per task")
    if config["bootstrap"] < 1000:
        raise ValueError("preregistered >=1000 family bootstrap replicates required")
    evaluations, curves, verdict_inputs = [], [], []
    for round_number in range(
        config["eval_every"], config["rounds"] + 1, config["eval_every"]
    ):
        tasks = {}
        for task in TASKS:
            records = paired_records(rows, task, "cv", round_number)
            seed = int(digest([config["bootstrap_seed"], task, round_number])[:16], 16)
            stats = comparison(records, task, config["bootstrap"], seed)
            tasks[task] = stats
            for role in ("val", "panel"):
                paired = paired_records(rows, task, role, round_number)
                tasks[task][role + "_elo"] = elo(
                    paired, "reg-only" if task == "regression" else "bin-only"
                )
            for arm in ARMS:
                by_family = defaultdict(list)
                for row in rows:
                    if (row["task"], row["role"], row["round"], row["arm"]) == (
                        task,
                        "val",
                        round_number,
                        arm,
                    ):
                        by_family[row["family"]].append(row["curve"])
                mean = sum(
                    sum(values) / len(values) for values in by_family.values()
                ) / len(by_family)
                curves.append(dict(round=round_number, task=task, arm=arm, value=mean))
        evaluations.append(dict(round=round_number, tasks=tasks))
        verdict_inputs.append({task: tasks[task]["joint"] for task in TASKS})
    return dict(
        status="COMPLETE",
        verdict=decide(verdict_inputs),
        plan_sha256=plan["sha256"],
        comparisons=evaluations,
        curves=curves,
        interpretation="study-local Elo (matching single-task anchor1500); CV bootstrap; panel ratings not historical eight-method Elo",
        percentiles="percentiles/trend pending JRB-02 comparable historical evaluation identities",
        mitigation="If inconclusive: task-balanced sampling is measured; preregister one per-task head/loss-weighting rerun before a further decision.",
    )


def render(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    write_new(output / "report.json", report)
    lines = [
        "# JRB-01",
        "",
        report["verdict"] or "NO VERDICT",
        "",
        report["interpretation"],
        report.get("percentiles", "percentiles pending JRB-02"),
        "",
        "| Task | Arm | CV Elo | Δ vs single [95% family CI] | Panel Elo |",
        "|---|---|---:|---:|---:|",
    ]
    for task in TASKS:
        stats = report["comparisons"][-1]["tasks"][task]
        for arm in ARMS:
            value = stats[arm]
            lines.append(
                f"| {task} | {arm} | {value['elo']:.2f} | {value['delta']:.2f} [{value['low']:.2f}, {value['high']:.2f}] | {stats['panel_elo'][arm]:.2f} |"
            )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    with (output / "curves.jsonl").open("x") as stream:
        for row in report["curves"]:
            stream.write(json.dumps(row) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, task in zip(axes, TASKS):
        for arm in ARMS:
            points = [
                row
                for row in report["curves"]
                if (row["task"], row["arm"]) == (task, arm)
            ]
            ax.plot(
                [row["round"] for row in points],
                [row["value"] for row in points],
                label=arm,
            )
        ax.set(
            title=task,
            xlabel="Round",
            ylabel="log RMSE/ridge" if task == "regression" else "log loss",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "validation-curves.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--score-store-module", required=True, type=Path)
    parser.add_argument("--score-store", required=True, type=Path)
    args = parser.parse_args()
    args.out = output_location(args.out)
    plan = json.loads(args.plan.read_text())
    try:
        rows = collect(plan, args.results)
        append_score_store(plan, rows, args.score_store_module, args.score_store)
        report = build_report(plan, rows)
    except (ValueError, KeyError, TypeError) as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        write_new(
            args.out / "incomplete.json",
            dict(status="INCOMPLETE", reason=str(exc), verdict=None),
        )
        raise
    render(report, args.out)


if __name__ == "__main__":
    main()
