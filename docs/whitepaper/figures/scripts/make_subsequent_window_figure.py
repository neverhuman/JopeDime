#!/usr/bin/env python3
"""Plot the consolidated 4-panel subsequent development window (rounds 2508-2527) from pinned summaries."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data" / "subsequent_window.csv"
PROVENANCE = DATA.with_suffix(".provenance.json")
RATES_DATA = FIGURES / "data" / "subsequent_rates.csv"
OUTPUT = FIGURES / "fig_fw_subsequent_window.pdf"
FIELDS = [
    "round",
    "program_promotions",
    "cumulative_program_promotions",
    "candidate_validation_utility",
    "recorded_best_validation_utility",
    "adapter_accepted",
    "completed_round_wall_seconds",
    "reported_best_round",
]


def extract(logs):
    inputs = {
        name: (logs / name).read_bytes()
        for name in ("summary.jsonl", "promotions.jsonl")
    }
    snapshot = json.loads((logs / "snapshot.json").read_text())
    for item in snapshot["files"]:
        assert hashlib.sha256(inputs[item["name"]]).hexdigest() == item["sha256"]
    summaries = [json.loads(line) for line in inputs["summary.jsonl"].splitlines()]
    rounds = [row["round"] for row in summaries]
    assert rounds == list(range(min(rounds), max(rounds) + 1)), (
        "Need consecutive summaries"
    )
    events = [json.loads(line) for line in inputs["promotions.jsonl"].splitlines()]
    selected = [
        row
        for row in events
        if row["kind"] == "promotion" and rounds[0] <= row["round"] <= rounds[-1]
    ]
    assert all(
        row["cv_gate_passed"] is True
        and row["evaluation_scope"] == "full_train_cv"
        and row["new_utility"] > row["old_utility"] + 1e-6
        for row in selected
    )
    assert len(
        {(r["round"], r["context_sha256"], r["program_sha256"]) for r in selected}
    ) == len(selected), "Duplicate promotion event"
    counts = Counter(row["round"] for row in selected)
    identities = set()
    rows, cumulative = [], 0
    for row in summaries:
        admission = row["adapter_admission"]
        candidate, best = row["candidate_dev_utility"], row["dev_utility"]
        assert row["round"] == admission["round"]
        assert counts[row["round"]] == row["promotions"], (
            "Journal/summary count mismatch"
        )
        assert candidate["n_bin"] == best["n_bin"] == 0, (
            "Require regression-only aggregates"
        )
        assert candidate["n_reg"] == best["n_reg"] > 0
        assert row["val_reg"] == candidate["val_reg"] == admission["candidate_utility"]
        assert best["val_reg"] == best["mean"] == admission["best_utility"]
        assert isinstance(admission["accepted"], bool)
        assert math.isfinite(row["wall_seconds"]) and row["wall_seconds"] > 0
        assert isinstance(admission["best_round"], int)
        assert admission["evaluation_identity"] == row["evaluation_identity"]
        identities.add(row["evaluation_identity"])
        cumulative += counts[row["round"]]
        rows.append(
            dict(
                zip(
                    FIELDS,
                    [
                        row["round"],
                        counts[row["round"]],
                        cumulative,
                        row["val_reg"],
                        admission["best_utility"],
                        int(admission["accepted"]),
                        row["wall_seconds"],
                        admission["best_round"],
                    ],
                )
            )
        )
    assert len(identities) == 1, "Do not join different evaluation identities"
    wall = sum(row["completed_round_wall_seconds"] for row in rows)
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
        "schema": "whitepaper.subsequent-window/v1",
        "snapshot": snapshot,
        "input_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()
        },
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
        "promotion_CI_scope": dict(
            Counter(r.get("CI_scope", "unrecorded") for r in selected)
        ),
        "evaluation_identities": sorted(identities),
        "metric_definition": {
            "program_promotions": "Strict training-program incumbent events, checked against summary counts",
            "cumulative_program_promotions": "Running sum within this window only; starts from zero",
            "candidate_validation_utility": "summary.val_reg = candidate_dev_utility.val_reg = adapter_admission.candidate_utility",
            "recorded_best_validation_utility": "dev_utility.val_reg = adapter_admission.best_utility",
            "completed_round_wall_seconds": "summary.wall_seconds; rates divide total completed rounds or strict program promotions by their summed wall time",
            "reported_best_round": "adapter_admission.best_round; a logged identity, not the first strict increase in utility",
        },
        "interpretation_source_commit": "fea10d2c15ae8544cfc851ce4d79755041120473",
        "source_binding": "Field interpretation only; installed run source/checkpoint qualification remains required",
        "scope": "Development telemetry only; no sealed data or independent first-proposal comparison",
        "csv_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "extraction_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "reproduce": "python3 figures/scripts/make_subsequent_window_figure.py --logs SNAPSHOT_DIR",
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--logs", type=Path, help="Stable complete-line development-log snapshot"
)
args = parser.parse_args()
if args.logs:
    extract(args.logs)
provenance = json.loads(PROVENANCE.read_text())
assert hashlib.sha256(DATA.read_bytes()).hexdigest() == provenance["csv_sha256"]
with DATA.open() as stream:
    rows = list(csv.DictReader(stream))
rounds = [int(row["round"]) for row in rows]
counts = [int(row["program_promotions"]) for row in rows]
cumulative = [int(row["cumulative_program_promotions"]) for row in rows]
candidate = [float(row["candidate_validation_utility"]) for row in rows]
best = [float(row["recorded_best_validation_utility"]) for row in rows]
assert rounds == list(range(min(rounds), max(rounds) + 1))
assert all(n >= 0 for n in counts)
assert cumulative == [sum(counts[: i + 1]) for i in range(len(counts))]
assert all(math.isfinite(v) for v in candidate + best)
assert provenance["round_window"] == [rounds[0], rounds[-1]]
assert provenance["summary_rows"] == len(rows)
assert provenance["strict_program_promotions"] == sum(counts)
gap = [b - c for b, c in zip(best, candidate)]
assert all(g >= 0 for g in gap)

elapsed = Decimal(0)
rate_rows = []
for row in rows:
    duration = Decimal(row["completed_round_wall_seconds"])
    count = int(row["program_promotions"])
    assert duration.is_finite() and duration > 0
    start = elapsed
    elapsed += duration
    rate_rows.append(
        {
            "start_min": float(start) / 60.0,
            "end_min": float(elapsed) / 60.0,
            "rounds_per_hour": float(Decimal(3600) / duration),
            "events_per_hour": float(Decimal(3600) * count / duration),
        }
    )
edges = [rate_rows[0]["start_min"]] + [r["end_min"] for r in rate_rows]
round_mean = 3600.0 * len(rows) / float(elapsed)
event_mean = 3600.0 * sum(counts) / float(elapsed)
assert math.isclose(round_mean, provenance["rounds_per_hour"], abs_tol=1e-9)
assert math.isclose(event_mean, provenance["program_promotions_per_hour"], abs_tol=1e-9)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.85))
xticks = list(dict.fromkeys([rounds[0], *rounds[5::5], rounds[-1]]))

# Panel 1: Logged validation utility & gap to recorded best
axes[0].fill_between(rounds, candidate, best, color="#0b5394", alpha=0.10, lw=0)
axes[0].plot(
    rounds, candidate, "o-", color="#6e7781", ms=2.2, lw=0.75, label="candidate"
)
axes[0].step(
    rounds, best, where="post", color="#0b5394", lw=1.05, label="best (rd 882)"
)
axes[0].set_title("(a) Validation utility & gap", fontsize=7.5, loc="left", pad=3)
axes[0].set_ylabel("validation pool utility")
axes[0].set_xlabel("round (2508–2527)")
axes[0].set_ylim(min(candidate + best) - 0.0008, max(candidate + best) + 0.0008)
axes[0].set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)
axes[0].set_xticks(xticks)
axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
axes[0].legend(
    loc="lower left",
    fontsize=6.3,
    frameon=False,
    handlelength=1.1,
    handletextpad=0.3,
)

# Panel 2: Per-round and cumulative training-program events
axes[1].bar(
    rounds, counts, width=0.65, color="#d0bfff", edgecolor="#5f3dc4", lw=0.4, label="per round"
)
axes[1].step(
    rounds, cumulative, where="post", color="#5f3dc4", lw=1.05, label="cumulative (15)"
)
axes[1].plot(rounds, cumulative, ".", color="#5f3dc4", ms=2.0)
axes[1].set_title("(b) Program-incumbent events", fontsize=7.5, loc="left", pad=3)
axes[1].set_ylabel("strict program events")
axes[1].set_xlabel("round (2508–2527)")
axes[1].set_ylim(-0.5, max(cumulative) + 1.8)
axes[1].set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)
axes[1].set_xticks(xticks)
axes[1].yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
axes[1].legend(
    loc="upper left",
    fontsize=6.3,
    frameon=False,
    handlelength=1.1,
    handletextpad=0.3,
)

# Panel 3: Completed-round pace over accumulated wall time
r_vals = [r["rounds_per_hour"] for r in rate_rows]
axes[2].stairs(r_vals, edges, color="#0b5394", lw=0.95, baseline=None)
axes[2].axhline(
    round_mean, color="#6e7781", ls="--", lw=0.8, label=f"window ({round_mean:.1f}/hr)"
)
axes[2].set_title("(c) Completed-round pace", fontsize=7.5, loc="left", pad=3)
axes[2].set_xlabel("accumulated round time (min)")
axes[2].set_ylabel("rounds per hour")
axes[2].set_xlim(edges[0], edges[-1])
axes[2].set_ylim(0, max(r_vals) * 1.22)
axes[2].legend(loc="upper right", frameon=False, fontsize=6.3, handlelength=1.2)

# Panel 4: Strict training-program event rate over accumulated wall time
e_vals = [r["events_per_hour"] for r in rate_rows]
axes[3].stairs(e_vals, edges, color="#6740b7", lw=0.95, baseline=None)
axes[3].axhline(
    event_mean, color="#6e7781", ls="--", lw=0.8, label=f"window ({event_mean:.1f}/hr)"
)
axes[3].set_title("(d) Program-event rate", fontsize=7.5, loc="left", pad=3)
axes[3].set_xlabel("accumulated round time (min)")
axes[3].set_ylabel("events per hour")
axes[3].set_xlim(edges[0], edges[-1])
axes[3].set_ylim(0, max(e_vals) * 1.22)
axes[3].legend(loc="upper right", frameon=False, fontsize=6.3, handlelength=1.2)

for ax in axes:
    ax.grid(axis="y", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)

fig.tight_layout(pad=0.35, w_pad=0.75)
fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)

provenance["extraction_script_sha256"] = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()
provenance["figure_sha256"] = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
provenance["matplotlib_version"] = matplotlib.__version__
provenance["figure_definition"] = {
    "a": "logged candidate and recorded-best validation pool utility (best labeled round 882) with paired utility shortfall shaded",
    "b": "per-round and cumulative strict training-program incumbent events (15 total; reset to zero for this window)",
    "c": "completed rounds per hour over accumulated completed-round wall time (41.58/hr whole-window rate)",
    "d": "strict training-program incumbent events per hour over accumulated completed-round wall time (31.19/hr whole-window rate)",
}
PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
print(f"{len(rows)} summary rounds; {sum(counts)} strict program promotions")
