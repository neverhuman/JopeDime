#!/usr/bin/env python3
"""Replay the fresh admission-prefix figures; optionally extract a captured log."""

import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data/live_admissions.csv"
SOURCE = FIGURES / "data/live_admissions.source.json"
PROVENANCE = FIGURES / "data/live_admissions.provenance.json"
REFERENCE = FIGURES / "data/admissions.csv"
TRAJECTORY = FIGURES / "fig_fw_live_validation.pdf"
SHORTFALL = FIGURES / "fig_fw_live_shortfall.pdf"
FIELDS = [
    "round",
    "candidate_val",
    "best_val",
    "accepted",
    "best_round",
    "timestamp",
    "utility_shortfall",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(log, snapshot):
    """Read only the captured aggregate prefix and its capture receipt."""
    metadata = json.loads(snapshot.read_text())
    assert sha(log) == metadata["prefix_sha256"]
    source = [
        json.loads(line, parse_float=Decimal) for line in log.read_text().splitlines()
    ]
    assert len(source) == metadata["source_rows"]
    rows = []
    for record in source:
        assert isinstance(record["accepted"], bool)
        candidate = record["candidate_utility"]
        best = record["best_utility"]
        rows.append(
            {
                "round": record["round"],
                "candidate_val": str(candidate),
                "best_val": str(best),
                "accepted": int(record["accepted"]),
                "best_round": record["best_round"],
                "timestamp": str(record["timestamp"]),
                "utility_shortfall": str(best - candidate),
            }
        )
    with DATA.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    source_receipt = {
        "schema": "whitepaper.live-admission-prefix/v1",
        "snapshot": metadata,
        "evaluation_identities": sorted({r["evaluation_identity"] for r in source}),
        "selection": "Every complete admission record in the captured stable prefix, in log order; no deduplication or outcome selection",
        "field_mapping": {
            "candidate_val": "candidate_utility",
            "best_val": "best_utility (post-decision recorded best)",
            "best_round": "best_round (logged checkpoint identity, not first strict gain)",
            "utility_shortfall": "Decimal(best_utility) - Decimal(candidate_utility)",
        },
        "csv_sha256": sha(DATA),
        "historical_reference_csv_sha256": sha(REFERENCE),
        "source_binding": "Aggregate field interpretation only; installed loop/adapter/checkpoint qualification remains required",
        "scope": "Adaptive development admission aggregates; no independent first-proposal or deployment evidence",
    }
    SOURCE.write_text(json.dumps(source_receipt, indent=2) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--log", type=Path)
parser.add_argument("--snapshot", type=Path)
args = parser.parse_args()
assert bool(args.log) == bool(args.snapshot), (
    "Supply both captured log and snapshot receipt"
)
if args.log:
    extract(args.log, args.snapshot)

source = json.loads(SOURCE.read_text())
assert sha(DATA) == source["csv_sha256"]
assert sha(REFERENCE) == source["historical_reference_csv_sha256"]
with DATA.open() as stream:
    rows = list(csv.DictReader(stream))
with REFERENCE.open() as stream:
    reference = list(csv.DictReader(stream))
rounds = [int(row["round"]) for row in rows]
assert rounds == list(range(rounds[0], rounds[-1] + 1))
assert len(rows) == source["snapshot"]["source_rows"]
for old, new in zip(reference, rows):
    assert old["round"] == new["round"] and old["accepted"] == new["accepted"]
    for field in ["candidate_val", "best_val", "timestamp"]:
        assert Decimal(old[field]) == Decimal(new[field])
assert len(rows) >= len(reference)

candidates = [Decimal(row["candidate_val"]) for row in rows]
bests = [Decimal(row["best_val"]) for row in rows]
gaps = [Decimal(row["utility_shortfall"]) for row in rows]
gains, ties, accepted = [], [], []
for i, row in enumerate(rows):
    assert row["accepted"] in ["0", "1"]
    assert all(v.is_finite() for v in [candidates[i], bests[i], gaps[i]])
    assert gaps[i] == bests[i] - candidates[i] and gaps[i] >= 0
    assert int(row["best_round"]) <= rounds[i]
    if i:
        assert bests[i] >= bests[i - 1]
        if row["accepted"] == "0":
            assert bests[i] == bests[i - 1]
    if row["accepted"] == "1":
        assert i > 0 and candidates[i] == bests[i]
        accepted.append(rounds[i])
        (gains if bests[i] > bests[i - 1] else ties).append(i)
    elif not i:
        assert row["accepted"] == "0"
assert accepted, "Last-acceptance annotation requires an accepted record"
last_acceptance = accepted[-1]
rejected_tail = len([r for r in rounds if r > last_acceptance])
assert all(
    row["accepted"] == "0" for row in rows if int(row["round"]) > last_acceptance
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
fig, ax = plt.subplots(figsize=(3.4, 2.15))
ax.scatter(
    rounds,
    [float(v) for v in candidates],
    s=3,
    color="#6e7781",
    alpha=0.45,
    linewidths=0,
    label="candidate adapter",
)
ax.step(
    rounds,
    [float(v) for v in bests],
    where="post",
    color="#0b5394",
    lw=1.1,
    label="recorded best",
)
ax.scatter(
    [rounds[i] for i in gains],
    [float(candidates[i]) for i in gains],
    s=28,
    marker="*",
    color="#c92a2a",
    zorder=5,
    label=f"strict gains ({len(gains)})",
)
ax.scatter(
    [rounds[i] for i in ties],
    [float(candidates[i]) for i in ties],
    s=18,
    marker="D",
    facecolors="none",
    edgecolors="#d9480f",
    lw=0.9,
    zorder=4,
    label=f"accepted ties ({len(ties)})",
)
ax.text(
    0.97,
    0.08,
    f"{len(rows):,} logged rounds\nlast acceptance: round {last_acceptance}",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=7.1,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.5),
)
ax.set(
    xlabel="round",
    ylabel="validation pool utility",
    xlim=(rounds[0] - 10, rounds[-1] + 10),
)
ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
ax.grid(axis="y", alpha=0.22, lw=0.4)
ax.legend(
    loc="lower left",
    frameon=True,
    facecolor="white",
    framealpha=0.88,
    edgecolor="none",
    handletextpad=0.3,
    borderpad=0.25,
)
fig.tight_layout(pad=0.4)
fig.savefig(TRAJECTORY, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.05))
ax.step(
    [0.0, *sorted(float(v) for v in gaps)],
    [0.0, *[(i + 1) / len(rows) for i in range(len(rows))]],
    where="post",
    color="#0b5394",
    lw=1.15,
)
ax.set(
    xlabel="Recorded best − candidate\n(validation pool utility)",
    ylabel="Share of logged rounds",
    xlim=(0, float(max(gaps)) * 1.05),
    ylim=(0, 1.03),
)
ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.grid(axis="y", alpha=0.22, lw=0.4)
ax.text(
    0.97,
    0.10,
    f"{len(rows):,} logged rounds",
    transform=ax.transAxes,
    ha="right",
    fontsize=7.2,
)
fig.tight_layout(pad=0.4)
fig.savefig(SHORTFALL, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)

provenance = {
    "schema": "whitepaper.live-admission-figures/v1",
    "source_receipt_sha256": sha(SOURCE),
    "csv_sha256": sha(DATA),
    "generator_sha256": sha(Path(__file__)),
    "figure_sha256": {p.name: sha(p) for p in [TRAJECTORY, SHORTFALL]},
    "round_window": [rounds[0], rounds[-1]],
    "rows": len(rows),
    "historical_reference_rows": len(reference),
    "additional_logged_rows": len(rows) - len(reference),
    "accepted_rounds": accepted,
    "strict_gain_rounds": [rounds[i] for i in gains],
    "accepted_tie_rounds": [rounds[i] for i in ties],
    "last_accepted_round": last_acceptance,
    "rejected_rounds_after_last_acceptance": rejected_tail,
    "final_recorded_best": str(bests[-1]),
    "final_recorded_best_round": int(rows[-1]["best_round"]),
    "zero_gap_rounds": [r for r, g in zip(rounds, gaps) if g == 0],
    "ECDF": "At x, count(gap <= x) / rows; each adaptive round once, repeated values retained; no duration weighting or independence claim",
    "zero_scope": "Post-decision best equals candidate on strict gains and accepted ties; zero is not necessarily a tie with the previous best",
    "source_binding": source["source_binding"],
    "scope": source["scope"],
    "matplotlib_version": matplotlib.__version__,
    "reproduce": "python3 figures/scripts/make_live_admissions_figures.py",
}
PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
print(
    f"{len(rows)} rounds; {len(accepted)} accepted; {rejected_tail} rejected after round {last_acceptance}"
)
