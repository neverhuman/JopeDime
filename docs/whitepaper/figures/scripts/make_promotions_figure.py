#!/usr/bin/env python3
"""Plot retained development logs; optionally extract with --logs SNAPSHOT_DIR."""
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
DATA = FIGURES / "data" / "program_promotions.csv"
PROVENANCE = DATA.with_suffix(".provenance.json")
FIELDS = ["round", "program_promotions", "candidate_validation_utility"]


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
    provenance = {
        "schema": "whitepaper.development-program-promotions/v1",
        "source_run": "/home/ubuntu/jope-data/flywheel-full-20261030/run-20261030",
        "source_host": "xbabe1",
        "snapshot": json.loads((logs / "snapshot.json").read_text()),
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
        "code_references": ["research/flywheel_selfplay/automatic_promotion.py:206",
                            "crates/jope-telemetry/src/hooks/flywheel.rs:362"],
        "source_binding": "Inspected code explains fields; the installed run source/checkpoint binding remains required",
        "scope": "Development telemetry; no sealed data, final test or one-proposal comparison",
        "csv_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "reproduce": "python3 figures/scripts/make_promotions_figure.py --logs SNAPSHOT_DIR",
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--logs", type=Path, help="Read-only complete-line snapshot of the two logs")
args = parser.parse_args()
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
print(f"{len(rows)} summary rounds; {sum(counts)} recorded program promotions")
