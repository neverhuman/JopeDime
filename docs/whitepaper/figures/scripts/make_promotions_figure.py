#!/usr/bin/env python3
"""Plot retained development logs; optionally extract with --logs SNAPSHOT_DIR.

The default promotion plot is an archived window. The manuscript uses the
live-development and live-admissions generators for its current main panels.
Historical target modes require --source-dir with the retained input tables.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data" / "program_promotions.csv"
PROVENANCE = DATA.with_suffix(".provenance.json")
FIELDS = ["round", "program_promotions", "candidate_validation_utility"]


def target_progress(source):
    """Recompute historical development counts from the merged publication tables."""
    names = ["comparisons.csv", "summary.csv",
             "carry-forward-audit.json"]
    inputs = {name: (source / name).read_bytes() for name in names}
    comparisons = list(csv.DictReader(inputs[names[0]].decode().splitlines()))
    summaries = list(csv.DictReader(inputs[names[1]].decode().splitlines()))
    arms = ["neural", "generic", "shuffled"]
    assert {row["arm"] for row in comparisons} == set(arms)
    assert len({(row["arm"], row["comparison_identity_sha256"])
                for row in comparisons}) == len(comparisons), "Duplicate arm groups"
    assert len(summaries) == 3 and {row["arm"] for row in summaries} == set(arms)
    rows = []
    improved_contexts = set()
    for arm, label in zip(arms, ["Policy-seeded", "Generic", "Shuffled"]):
        group = [row for row in comparisons if row["arm"] == arm]
        counts = Counter()
        reductions = []
        for row in group:
            first, last = float(row["first_score"]), float(row["last_score"])
            assert all(math.isfinite(v) and v > 0 for v in [first, last])
            assert int(row["completed_evaluations"]) == 1000
            assert int(row["first_generation"]) < int(row["last_generation"])
            outcome = "lower" if last < first else "higher" if last > first else "same"
            counts[outcome] += 1
            reductions.append((first - last) / first)
            if outcome == "lower":
                improved_contexts.add((row["dataset"], row["cut"]))
        summary = next(row for row in summaries if row["arm"] == arm)
        assert int(summary["completed_evaluations"]) == 1000
        assert len(group) == int(summary["repeated_comparisons"])
        for key, original in [("lower", "improved"), ("same", "unchanged"),
                              ("higher", "worsened")]:
            assert counts[key] == int(summary[original]), "Summary/count mismatch"
        assert math.isclose(median(reductions), float(summary["median_relative_improvement"]),
                            abs_tol=1e-12)
        rows.append(dict(arm=label, groups=len(group), **{k: counts[k] for k in
                                                        ["lower", "same", "higher"]}))
    contexts = {(row["dataset"], row["cut"]) for row in comparisons}
    audit = json.loads(inputs[names[2]])["independently_recomputed_from_published_comparisons"]
    assert (len(comparisons), len(contexts), len(improved_contexts)) == (
        audit["arm_groups"], audit["distinct_dataset_cuts"], audit["distinct_improved_dataset_cuts"])
    output = FIGURES / "data/target_progress.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "schema": "whitepaper.historical-target-progress/v1",
        "source_commit": "569324b7760a95be26e817ce843b2430e32095fb",
        "inputs": {name: hashlib.sha256(data).hexdigest()
                   for name, data in inputs.items()},
        "selection": "All 159 merged comparison rows; first versus last retained generation per arm group",
        "score": "best_validation_rmse; historical normalized training-fold RMSE units; lower is better",
        "outcome_rule": "Exact parsed score comparisons (<, ==, >); no epsilon or added cumulative minimum",
        "completed_evaluations_per_search": 1000,
        "arm_groups": len(comparisons), "distinct_dataset_cuts": len(contexts),
        "lower_arm_groups": sum(row["lower"] for row in rows),
        "distinct_improved_dataset_cuts": len(improved_contexts),
        "scope": "Historical adaptive development validation only; no sealed or holdout inputs",
        "limitations": ["Shared incumbent/teacher seeds make arms dependent",
                        "RNG and elapsed wall time vary; equal consumed compute is not established",
                        "Lower retained scores do not establish neural-policy causality or transfer",
                        "Original score/seed payloads were not reopened; seed audit is retained metadata"],
        "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reproduce": "python3 figures/scripts/make_promotions_figure.py --target-progress --source-dir INPUT_DIR",
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    left = [0] * len(rows)
    for key, label, color in [("lower", "Last lower", "#2b8a3e"),
                              ("same", "Same", "#adb5bd"),
                              ("higher", "Last higher (0)", "#c92a2a")]:
        values = [row[key] for row in rows]
        bars = ax.barh(range(len(rows)), values, left=left, label=label, color=color,
                       height=0.58)
        for bar, value in zip(bars, values):
            if value:
                ax.text(bar.get_x() + value / 2, bar.get_y() + bar.get_height() / 2,
                        str(value), ha="center", va="center", fontsize=8,
                        color="white" if key == "lower" else "#212529")
        left = [a + b for a, b in zip(left, values)]
    ax.set_yticks(range(len(rows)), [f"{row['arm']} (n={row['groups']})" for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("Dataset/cut/arm groups")
    ax.grid(axis="x", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False,
              fontsize=7.5, handlelength=0.8, handletextpad=0.3, columnspacing=0.7)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURES / "fig_fw_target_progress.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"{len(comparisons)} historical arm groups; {len(contexts)} dataset/cuts; "
          f"{sum(row['lower'] for row in rows)} lower groups on {len(improved_contexts)} dataset/cuts")


def target_history(source):
    """Plot every retained generation of the published post-hoc development example."""
    names = ["selected-historical-cut.csv", "comparisons.csv",
             "carry-forward-audit.json"]
    inputs = {name: (source / name).read_bytes() for name in names}
    history = list(csv.DictReader(inputs[names[0]].decode().splitlines()))
    comparisons = list(csv.DictReader(inputs[names[1]].decode().splitlines()))
    identity_fields = ["run", "dataset", "cut", "comparison_identity_sha256",
                       "prepared_sha256", "vocabulary_sha256", "evaluator_sha256",
                       "evaluator_protocol", "execution_order_protocol", "completed_evaluations"]
    assert len(history) == 213
    assert all(len({row[key] for row in history}) == 1 for key in identity_fields)
    assert all(int(row["completed_evaluations"]) == 1000 for row in history)
    assert {row["arm"] for row in history} == {"neural", "generic", "shuffled"}
    selected = max((row for row in comparisons if row["arm"] == "neural"),
                   key=lambda row: float(row["last_relative_improvement"]))
    assert (selected["dataset"], selected["cut"]) == (history[0]["dataset"], history[0]["cut"])
    rows, first_final_generation = [], {}
    for arm, label in [("neural", "Policy-seeded"), ("generic", "Generic"),
                       ("shuffled", "Shuffled")]:
        group = sorted((row for row in history if row["arm"] == arm),
                       key=lambda row: int(row["generation"]))
        assert [int(row["generation"]) for row in group] == list(range(71))
        assert all(math.isfinite(float(row["score"])) and float(row["score"]) > 0
                   for row in group)
        comparison = next(row for row in comparisons if row["arm"] == arm
                          and row["dataset"] == selected["dataset"] and row["cut"] == selected["cut"])
        assert float(group[0]["score"]) == float(comparison["first_score"])
        assert float(group[-1]["score"]) == float(comparison["last_score"])
        first_final_generation[label] = next(int(row["generation"]) for row in group
                                             if float(row["score"]) == float(group[-1]["score"]))
        rows.extend({"generation": row["generation"], "arm": label,
                     "reported_best_development_rmse": row["score"]} for row in group)
    output = FIGURES / "data/target_history.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "schema": "whitepaper.historical-target-history/v1",
        "source_commit": "54f1023225364cefbb49ba7ee12ad967dddf3ce8",
        "inputs": {name: hashlib.sha256(data).hexdigest()
                   for name, data in inputs.items()},
        "selection": "Post-hoc example: largest first-to-last relative reduction in the policy-seeded 1000-evaluation comparison subset",
        "rows": len(rows), "generations_per_arm": 71,
        "completed_evaluations_per_search": 1000,
        "comparison_identity_sha256": history[0]["comparison_identity_sha256"],
        "score": "best_validation_rmse; historical normalized training-fold RMSE units; lower is better",
        "transformation": "All raw reported scores retained; no cumulative minimum, smoothing or interpolation",
        "first_generation_reporting_final_value": first_final_generation,
        "scope": "Historical adaptive development validation only; no sealed or holdout inputs",
        "limitations": ["Example selected after inspection; not a representative random sample",
                        "Shared incumbent/teacher seeds make arms dependent; RNG and elapsed time vary",
                        "Matching final scores does not establish independent discovery or learned-policy gain",
                        "Seed audit is retained metadata; original score/seed payloads were not reopened",
                        "Checkpoint exposure of improved targets to later optimizer batches remains unjoined"],
        "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reproduce": "python3 figures/scripts/make_promotions_figure.py --target-history --source-dir INPUT_DIR",
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.05), sharex=True, sharey=True)
    for ax, label, color in zip(axes, first_final_generation, ["#0b5394", "#2b8a3e", "#d95f02"]):
        group = [row for row in rows if row["arm"] == label]
        ax.plot([int(row["generation"]) for row in group],
                [float(row["reported_best_development_rmse"]) for row in group],
                ".-", ms=2.5, lw=0.9, color=color)
        ax.set_title(label, fontsize=8, loc="left", pad=3)
        ax.set_xlabel("Generation")
        ax.set_xticks([0, 20, 40, 60, 70])
        ax.set_xlim(-1, 71)
        ax.set_ylim(0.0025, 0.0105)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=0.22, linewidth=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Reported best\ndevelopment RMSE")
    fig.tight_layout(pad=0.4, w_pad=0.7)
    fig.savefig(FIGURES / "fig_fw_target_history.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"{len(rows)} historical scores; first final-value generations {first_final_generation}")


def plot_target_combined():
    """Plot consolidated 4-panel historical GP target figure from committed CSVs."""
    prog_csv = FIGURES / "data/target_progress.csv"
    hist_csv = FIGURES / "data/target_history.csv"
    with prog_csv.open() as stream:
        prog_rows = list(csv.DictReader(stream))
    with hist_csv.open() as stream:
        hist_rows = list(csv.DictReader(stream))
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.95),
                             gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
    ax0 = axes[0]
    left = [0] * len(prog_rows)
    for key, label, color in [("lower", "Lower", "#2b8a3e"),
                              ("same", "Same", "#adb5bd"),
                              ("higher", "Higher (0)", "#c92a2a")]:
        values = [int(row[key]) for row in prog_rows]
        bars = ax0.barh(range(len(prog_rows)), values, left=left, label=label,
                        color=color, height=0.58)
        for bar, value in zip(bars, values):
            if value:
                ax0.text(bar.get_x() + value / 2, bar.get_y() + bar.get_height() / 2,
                         str(value), ha="center", va="center", fontsize=6.8,
                         color="white" if key == "lower" else "#212529")
        left = [a + b for a, b in zip(left, values)]
    ax0.set_yticks(range(len(prog_rows)),
                   [f"{row['arm']} ({row['groups']})" for row in prog_rows])
    ax0.invert_yaxis()
    ax0.set_xlim(0, 60)
    ax0.set_xticks([0, 20, 40, 60])
    ax0.set_xlabel("Comparison groups")
    ax0.set_title("First vs. last (159 groups)", fontsize=7.5, loc="left", pad=2.5)
    ax0.grid(axis="x", alpha=0.22, linewidth=0.4)
    ax0.set_axisbelow(True)
    ax0.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.88,
               edgecolor="none", fontsize=6.2, handlelength=0.8, handletextpad=0.25,
               borderpad=0.2)
    for idx, (arm_label, color) in enumerate([
        ("Policy-seeded", "#0b5394"),
        ("Generic", "#2b8a3e"),
        ("Shuffled", "#d95f02"),
    ], start=1):
        ax = axes[idx]
        group = [row for row in hist_rows if row["arm"] == arm_label]
        ax.plot([int(row["generation"]) for row in group],
                [float(row["reported_best_development_rmse"]) for row in group],
                ".-", ms=2.2, lw=0.85, color=color)
        ax.set_title(f"{arm_label} (71 gen)", fontsize=7.5, loc="left", pad=2.5)
        ax.set_xlabel("Generation")
        ax.set_xticks([0, 20, 40, 70])
        ax.set_xlim(-1, 71)
        ax.set_ylim(0.0025, 0.0105)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        if idx == 1:
            ax.set_ylabel("Reported best\ndev. RMSE")
        else:
            ax.tick_params(labelleft=False)
        ax.grid(alpha=0.22, linewidth=0.4)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.35, w_pad=0.45)
    fig.savefig(FIGURES / "fig_fw_target_history.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    gen_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for p_path in [prog_csv.with_suffix(".provenance.json"),
                   hist_csv.with_suffix(".provenance.json")]:
        if p_path.exists():
            prov = json.loads(p_path.read_text())
            prov["generator_sha256"] = gen_sha
            p_path.write_text(json.dumps(prov, indent=2) + "\n")


def extract(logs):
    inputs = {name: (logs / name).read_bytes()
              for name in ("promotions.jsonl", "summary.jsonl")}
    events = [json.loads(line) for line in inputs["promotions.jsonl"].splitlines()]
    summaries = [json.loads(line) for line in inputs["summary.jsonl"].splitlines()]
    summaries.sort(key=lambda row: row["round"])
    rounds = [row["round"] for row in summaries]
    assert rounds == list(range(min(rounds), max(rounds) + 1)), "Need consecutive summaries"
    selected = [row for row in events if row["kind"] == "promotion"
                and min(rounds) <= row["round"] <= max(rounds)]
    assert all(row["cv_gate_passed"] is True
               and row["evaluation_scope"] == "full_train_cv"
               and row["new_utility"] > row["old_utility"] + 1e-6 for row in selected)
    identities = {(row["round"], row["context_sha256"], row["program_sha256"])
                  for row in selected}
    assert len(identities) == len(selected), "Duplicate promotion events"
    counts = Counter(row["round"] for row in selected)
    rows = []
    for row in summaries:
        assert counts[row["round"]] == row["promotions"], "Journal/summary count mismatch"
        assert row["val_reg"] == row["candidate_dev_utility"]["val_reg"]
        assert math.isfinite(row["val_reg"])
        rows.append(dict(zip(FIELDS, [row["round"], counts[row["round"]], row["val_reg"]])))
    with DATA.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    # Disclosure boundary: keep content hashes and sizes, never host names or host paths.
    snapshot = json.loads((logs / "snapshot.json").read_text())
    snapshot.pop("host", None)
    for entry in snapshot.get("files", []):
        entry.pop("path", None)
        entry.pop("inode", None)
    provenance = {
        "schema": "whitepaper.development-program-promotions/v1",
        "source_run": "retained-development-run",
        "snapshot": snapshot,
        "input_sha256": {name: hashlib.sha256(data).hexdigest()
                         for name, data in inputs.items()},
        "selection": "All consecutive retained summary rounds; journal kind=promotion in that window",
        "round_window": [min(rounds), max(rounds)],
        "summary_rows": len(rows),
        "program_promotions": len(selected),
        "distinct_training_contexts_promoted": len({r["context_sha256"] for r in selected}),
        "promotion_CI_scope": dict(Counter(r["CI_scope"] for r in selected)),
        "evaluation_identities": sorted({r["evaluation_identity"] for r in summaries}),
        "metric_definition": "summary.val_reg = candidate_dev_utility.val_reg; larger logged utility is better",
        "source_binding": "Inspected code explains fields; the installed run source/checkpoint binding remains required",
        "scope": "Development telemetry; no sealed data, final test or one-proposal comparison",
        "csv_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "reproduce": "python3 figures/scripts/make_promotions_figure.py --logs SNAPSHOT_DIR",
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
selection = parser.add_mutually_exclusive_group()
selection.add_argument("--logs", type=Path, help="Read-only complete-line snapshot of the two logs")
selection.add_argument("--target-progress", action="store_true",
                       help="Plot historical GP target progress from the merged publication tables")
selection.add_argument("--target-history", action="store_true",
                       help="Plot all generations of the merged historical GP target example")
parser.add_argument("--source-dir", type=Path, help="Retained historical tables with neutral filenames")
args = parser.parse_args()
if (args.target_history or args.target_progress) and args.source_dir is None:
    parser.error("historical target modes require --source-dir INPUT_DIR")
if args.target_history:
    target_history(args.source_dir)
    raise SystemExit(0)
if args.target_progress:
    target_progress(args.source_dir)
    raise SystemExit(0)
if args.logs:
    extract(args.logs)
with DATA.open() as stream:
    rows = list(csv.DictReader(stream))
rounds = [int(row["round"]) for row in rows]
counts = [int(row["program_promotions"]) for row in rows]
utilities = [float(row["candidate_validation_utility"]) for row in rows]
assert rounds == list(range(min(rounds), max(rounds) + 1))
assert all(count >= 0 for count in counts) and all(math.isfinite(u) for u in utilities)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.7), sharex=True)
axes[0].bar(rounds, counts, width=0.7, color="#5f3dc4")
axes[0].set_title("Logged strict program promotions", fontsize=8, loc="left", pad=3)
axes[0].set_ylabel("count / round")
axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[1].plot(rounds, utilities, "o-", ms=2.5, lw=0.9, color="#0b5394")
axes[1].set_title("Candidate adapter validation", fontsize=8, loc="left", pad=3)
axes[1].set_ylabel("pool utility")
axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
axes[1].set_xlabel("round (later retained summary window)")
axes[1].set_xticks([rounds[0], *rounds[5::5], rounds[-1]])
for ax in axes:
    ax.grid(axis="y", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(rounds[0] - 0.7, rounds[-1] + 0.7)
fig.tight_layout(pad=0.45, h_pad=0.7)
fig.savefig(FIGURES / "fig_fw_program_promotions.pdf",
            metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)
plot_target_combined()
print(f"{len(rows)} summary rounds; {sum(counts)} recorded program promotions")
