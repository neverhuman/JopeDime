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
          "adapter_accepted"]


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
        assert admission["evaluation_identity"] == row["evaluation_identity"]
        identities.add(row["evaluation_identity"])
        cumulative += counts[row["round"]]
        rows.append(dict(zip(FIELDS, [row["round"], counts[row["round"]], cumulative,
                                     row["val_reg"], admission["best_utility"],
                                     int(admission["accepted"])])))
    assert len(identities) == 1, "Do not join different evaluation identities"
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
        "promotion_CI_scope": dict(Counter(r.get("CI_scope", "unrecorded") for r in selected)),
        "evaluation_identities": sorted(identities),
        "metric_definition": {
            "program_promotions": "Strict training-program incumbent events, checked against summary counts",
            "cumulative_program_promotions": "Running sum within this window only; starts from zero",
            "candidate_validation_utility": "summary.val_reg = candidate_dev_utility.val_reg = adapter_admission.candidate_utility",
            "recorded_best_validation_utility": "dev_utility.val_reg = adapter_admission.best_utility",
        },
        "code_references": ["research/flywheel_selfplay/automatic_promotion.py:206-273",
                            "research/flywheel_selfplay/dev_incumbent.py:524-570",
                            "crates/jope-telemetry/src/hooks/flywheel.rs:362-365"],
        "interpretation_source_commit": "b2c04a846194bda01310d762d646c8a2a91d26cf",
        "source_binding": "Field interpretation only; installed run source/checkpoint qualification remains required",
        "scope": "Development telemetry only; no sealed data or independent first-proposal comparison",
        "csv_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "extraction_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reproduce": ("python3 figures/scripts/make_recent_development_figure.py"
                      + (" --follow-up" if args.follow_up else "") + " --logs SNAPSHOT_DIR"),
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--logs", type=Path, help="Stable complete-line development-log snapshot")
parser.add_argument("--follow-up", action="store_true", help="Plot the separate follow-up snapshot")
args = parser.parse_args()
if args.follow_up:
    DATA = FIGURES / "data" / "followup_development.csv"
    PROVENANCE = DATA.with_suffix(".provenance.json")
    OUTPUT = FIGURES / "fig_fw_followup_development.pdf"
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
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
if args.follow_up:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 1.9), sharex=True)
    axes[0].bar(rounds, counts, color="#5f3dc4", width=0.7)
    axes[0].set_title("Left: Training-program promotions", fontsize=8, loc="left", pad=3)
    axes[0].set_ylabel("promotions / round")
    axes[0].set_ylim(-0.1, max(counts) + 0.5)
    axes[0].set_xlabel("round (follow-up retained window)")
else:
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.85), sharex=True)
    axes[0].step(rounds, cumulative, where="post", color="#5f3dc4", lw=1.1)
    axes[0].plot(rounds, cumulative, ".", color="#5f3dc4", ms=2.5)
    axes[0].set_title("Top: Training-program promotions", fontsize=8, loc="left", pad=3)
    axes[0].set_ylabel("cumulative\npromotions")
    axes[0].set_ylim(-0.5, max(cumulative) + max(1, max(cumulative) * 0.1))
axes[0].yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
axes[1].plot(rounds, candidate, "o-", color="#6e7781", ms=2.5, lw=0.8,
             label="candidate adapter")
axes[1].step(rounds, best, where="post", color="#0b5394", lw=1.15,
             label="recorded best")
axes[1].set_title("Right: Adapter validation" if args.follow_up else "Bottom: Adapter validation",
                  fontsize=8, loc="left", pad=3)
axes[1].set_ylabel("validation\npool utility")
axes[1].set_ylim(min(candidate + best) - 0.004, max(candidate + best) + 0.0013)
axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
axes[1].legend(loc="lower left", ncol=2, fontsize=7.2, frameon=False,
               handlelength=1.3, handletextpad=0.3, columnspacing=0.7, borderaxespad=0.3)
axes[1].set_xlabel("round (follow-up retained window)" if args.follow_up
                   else "round (additional retained window)")
axes[1].set_xticks([rounds[0], *rounds[5::5], rounds[-1]])
for ax in axes:
    ax.set_xlim(rounds[0] - 0.4, rounds[-1] + 0.4)
    ax.grid(axis="y", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)
fig.tight_layout(pad=0.4, h_pad=0.65)
fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)
print(f"{len(rows)} summary rounds; {sum(counts)} strict program promotions")
