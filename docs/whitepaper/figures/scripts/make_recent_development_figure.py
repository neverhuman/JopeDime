#!/usr/bin/env python3
"""Plot the committed development snapshot; re-extract with --logs SNAPSHOT_DIR."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data" / "recent_development.csv"
PROVENANCE = DATA.with_suffix(".provenance.json")
OUTPUT = FIGURES / "fig_fw_recent_development.pdf"
FIELDS = ["round", "program_promotions", "cumulative_program_promotions",
          "candidate_validation_utility", "recorded_best_validation_utility",
          "adapter_accepted", "completed_round_wall_seconds", "reported_best_round"]


def extract(logs):
    inputs = {name: (logs / name).read_bytes()
              for name in ("summary.jsonl", "promotions.jsonl")}
    snapshot = json.loads((logs / "snapshot.json").read_text())
    for item in snapshot["files"]:
        assert hashlib.sha256(inputs[item["name"]]).hexdigest() == item["sha256"]
    summaries = [json.loads(line) for line in inputs["summary.jsonl"].splitlines()]
    rounds = [row["round"] for row in summaries]
    assert rounds == list(range(min(rounds), max(rounds) + 1)), "Need consecutive summaries"
    events = [json.loads(line) for line in inputs["promotions.jsonl"].splitlines()]
    selected = [row for row in events if row["kind"] == "promotion"
                and rounds[0] <= row["round"] <= rounds[-1]]
    assert all(row["cv_gate_passed"] is True
               and row["evaluation_scope"] == "full_train_cv"
               and row["new_utility"] > row["old_utility"] + 1e-6 for row in selected)
    assert len({(r["round"], r["context_sha256"], r["program_sha256"])
                for r in selected}) == len(selected), "Duplicate promotion event"
    counts = Counter(row["round"] for row in selected)
    identities = set()
    rows, cumulative = [], 0
    for row in summaries:
        admission = row["adapter_admission"]
        candidate, best = row["candidate_dev_utility"], row["dev_utility"]
        assert row["round"] == admission["round"]
        assert counts[row["round"]] == row["promotions"], "Journal/summary count mismatch"
        assert candidate["n_bin"] == best["n_bin"] == 0, "Require regression-only aggregates"
        assert candidate["n_reg"] == best["n_reg"] > 0
        assert row["val_reg"] == candidate["val_reg"] == admission["candidate_utility"]
        assert best["val_reg"] == best["mean"] == admission["best_utility"]
        assert isinstance(admission["accepted"], bool)
        assert math.isfinite(row["wall_seconds"]) and row["wall_seconds"] > 0
        assert isinstance(admission["best_round"], int)
        assert admission["evaluation_identity"] == row["evaluation_identity"]
        identities.add(row["evaluation_identity"])
        cumulative += counts[row["round"]]
        rows.append(dict(zip(FIELDS, [row["round"], counts[row["round"]], cumulative,
                                     row["val_reg"], admission["best_utility"],
                                     int(admission["accepted"]), row["wall_seconds"],
                                     admission["best_round"]])))
    assert len(identities) == 1, "Do not join different evaluation identities"
    wall = sum(row["completed_round_wall_seconds"] for row in rows)
    rolling = summaries[-20:]
    rolling_rate = (3600 * sum(row["promotions"] for row in rolling)
                    / sum(row["wall_seconds"] for row in rolling))
    assert summaries[-1]["promotions_rate_scope"] == (
        "up to 20 completed rounds since process start; includes qualification wait")
    assert math.isclose(rolling_rate, summaries[-1]["promotions_per_hour"],
                        rel_tol=1e-12), "Derived rate differs from final logged rolling rate"
    with DATA.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    # Keep prefix hashes and sizes; host names, paths and inode numbers stay private.
    snapshot.pop("host", None)
    for item in snapshot["files"]:
        item.pop("path", None)
        item.pop("inode", None)
    provenance = {
        "schema": "whitepaper.recent-development/v1",
        "snapshot": snapshot,
        "input_sha256": {name: hashlib.sha256(data).hexdigest()
                         for name, data in inputs.items()},
        "selection": "All complete consecutive summary rounds; journal kind=promotion in that window",
        "round_window": [rounds[0], rounds[-1]],
        "summary_rows": len(rows),
        "strict_program_promotions": len(selected),
        "adapter_admissions": sum(row["adapter_accepted"] for row in rows),
        "reported_best_rounds": sorted({row["reported_best_round"] for row in rows}),
        "completed_round_wall_seconds": wall,
        "rounds_per_hour": 3600 * len(rows) / wall,
        "program_promotions_per_hour": 3600 * len(selected) / wall,
        "rate_scope": "Retained completed rounds; logged wall time includes qualification waits; not deployment throughput",
        "promotion_CI_scope": dict(Counter(r.get("CI_scope", "unrecorded") for r in selected)),
        "evaluation_identities": sorted(identities),
        "metric_definition": {
            "program_promotions": "Strict training-program incumbent events, checked against summary counts",
            "cumulative_program_promotions": "Running sum within this window only; starts from zero",
            "candidate_validation_utility": "summary.val_reg = candidate_dev_utility.val_reg = adapter_admission.candidate_utility",
            "recorded_best_validation_utility": "dev_utility.val_reg = adapter_admission.best_utility",
            "completed_round_wall_seconds": "summary.wall_seconds; rates divide total completed rounds or strict program promotions by their summed wall time",
            "reported_best_round": "adapter_admission.best_round; a logged identity, not the first strict increase in utility",
        },
        "interpretation_source_commit": ("a13afb4fe8f98b5f9b17fbd8c77650b4c8ba9343"
                                         if args.later_window else
                                         "b2c04a846194bda01310d762d646c8a2a91d26cf"),
        "source_binding": "Field interpretation only; installed run source/checkpoint qualification remains required",
        "scope": "Development telemetry only; no sealed data or independent first-proposal comparison",
        "csv_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "extraction_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reproduce": ("python3 figures/scripts/make_recent_development_figure.py"
                      + (" --follow-up" if args.follow_up else
                         " --later-window" if args.later_window else "")
                      + " --logs SNAPSHOT_DIR"),
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--logs", type=Path, help="Stable complete-line development-log snapshot")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--follow-up", action="store_true", help="Plot the separate follow-up snapshot")
mode.add_argument("--later-window", action="store_true", help="Plot the separate later snapshot")
args = parser.parse_args()
if args.follow_up:
    DATA = FIGURES / "data" / "followup_development.csv"
    PROVENANCE = DATA.with_suffix(".provenance.json")
    OUTPUT = FIGURES / "fig_fw_followup_development.pdf"
elif args.later_window:
    DATA = FIGURES / "data" / "later_development.csv"
    PROVENANCE = DATA.with_suffix(".provenance.json")
    OUTPUT = FIGURES / "fig_fw_later_development.pdf"
if args.logs:
    extract(args.logs)
with DATA.open() as stream:
    rows = list(csv.DictReader(stream))
rounds = [int(row["round"]) for row in rows]
counts = [int(row["program_promotions"]) for row in rows]
cumulative = [int(row["cumulative_program_promotions"]) for row in rows]
candidate = [float(row["candidate_validation_utility"]) for row in rows]
best = [float(row["recorded_best_validation_utility"]) for row in rows]
assert rounds == list(range(min(rounds), max(rounds) + 1))
assert all(n >= 0 for n in counts)
assert cumulative == [sum(counts[:i + 1]) for i in range(len(counts))]
assert all(math.isfinite(v) for v in candidate + best)
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
wide = args.follow_up or args.later_window
round_axis = ("round (later retained window)" if args.later_window
              else "round (follow-up retained window)")
if wide:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 1.9), sharex=True)
    axes[0].bar(rounds, counts, color="#5f3dc4", width=0.7)
    axes[0].set_title("Left: Training-program promotions", fontsize=8, loc="left", pad=3)
    axes[0].set_ylabel("promotions / round")
    axes[0].set_ylim(-0.1, max(counts) + 0.5)
    axes[0].set_xlabel(round_axis)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    axes[1].plot(rounds, candidate, "o-", color="#6e7781", ms=2.5, lw=0.8,
                 label="candidate adapter")
    axes[1].step(rounds, best, where="post", color="#0b5394", lw=1.15,
                 label="recorded best")
    axes[1].set_title("Right: Adapter validation", fontsize=8, loc="left", pad=3)
    axes[1].set_ylabel("validation\npool utility")
    axes[1].set_ylim(min(candidate + best) - 0.004, max(candidate + best) + 0.0013)
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].legend(loc="lower left", ncol=2, fontsize=7.2, frameon=False,
                   handlelength=1.3, handletextpad=0.3, columnspacing=0.7, borderaxespad=0.3)
    axes[1].set_xlabel(round_axis)
    axes[1].set_xticks(list(dict.fromkeys([rounds[0], *rounds[5::5], rounds[-1]])))
    for ax in axes:
        ax.set_xlim(rounds[0] - 0.4, rounds[-1] + 0.4)
        ax.grid(axis="y", alpha=0.22, linewidth=0.4)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4, h_pad=0.65)
else:
    window_specs = [
        ("recent_development.csv", "Rounds 2005–2024"),
        ("followup_development.csv", "Rounds 2082–2101"),
        ("later_development.csv", "Rounds 2343–2362"),
        ("live_development.csv", "Rounds 2644–2663"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(6.8, 2.85), sharex="col", sharey="row")
    for col, (csv_name, title_prefix) in enumerate(window_specs):
        with (FIGURES / "data" / csv_name).open() as stream:
            w_rows = list(csv.DictReader(stream))
        w_rounds = [int(r["round"]) for r in w_rows]
        w_counts = [int(r["program_promotions"]) for r in w_rows]
        w_cand = [float(r["candidate_validation_utility"]) for r in w_rows]
        w_best = [float(r["recorded_best_validation_utility"]) for r in w_rows]
        has_wall = "completed_round_wall_seconds" in w_rows[0]
        w_wall = sum(float(r["completed_round_wall_seconds"]) for r in w_rows) if has_wall else None
        w_adm = sum(int(r["adapter_accepted"]) for r in w_rows)
        assert w_rounds == list(range(w_rounds[0], w_rounds[-1] + 1))
        assert w_adm == 0
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        ax_top.bar(w_rounds, w_counts, color="#5f3dc4", width=0.72)
        subtitle = (f"{sum(w_counts)} events, {w_wall / 60:.1f} min"
                    if w_wall is not None else f"{sum(w_counts)} events, 0 adm.")
        ax_top.set_title(
            f"{title_prefix}\n({subtitle})",
            fontsize=7.2, loc="left", pad=2.5,
        )
        ax_top.set_ylim(0, max(5, max(w_counts) + 0.6))
        ax_top.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax_bot.plot(w_rounds, w_cand, "o-", color="#6e7781", ms=2.2, lw=0.8,
                    label="candidate adapter")
        ax_bot.step(w_rounds, w_best, where="post", color="#0b5394", lw=1.1,
                    label="recorded best (1.1378)")
        ax_bot.set_ylim(1.124, 1.1395)
        ax_bot.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax_bot.set_xlabel("round")
        ax_bot.set_xticks([w_rounds[0], w_rounds[9], w_rounds[-1]])
        for ax in (ax_top, ax_bot):
            ax.set_xlim(w_rounds[0] - 0.6, w_rounds[-1] + 0.6)
            ax.grid(axis="y", alpha=0.22, linewidth=0.4)
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("promotions /\nround")
    axes[1, 0].set_ylabel("validation\npool utility")
    axes[1, 0].legend(loc="lower left", fontsize=6.4, frameon=True, facecolor="white",
                      framealpha=0.88, edgecolor="none", handlelength=1.1,
                      handletextpad=0.25, borderpad=0.2)
    fig.tight_layout(pad=0.35, w_pad=0.45, h_pad=0.5)
fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)
if PROVENANCE.exists():
    provenance = json.loads(PROVENANCE.read_text())
    provenance["figure_sha256"] = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    provenance["extraction_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance["matplotlib_version"] = matplotlib.__version__
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
print(f"{len(rows)} summary rounds; {sum(counts)} strict program promotions")
