#!/usr/bin/env python3
"""Replay the latest retained program-event window; extract with --logs DIR."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data/live_development.csv"
PROVENANCE = DATA.with_suffix(".provenance.json")
OUTPUT = FIGURES / "fig_fw_live_development.pdf"
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(logs):
    snapshot = json.loads((logs / "snapshot.json").read_text())
    snapshot.pop("host", None)
    inputs = {
        name: (logs / name).read_bytes()
        for name in ["summary.jsonl", "promotions.jsonl"]
    }
    for item in snapshot["files"]:
        assert hashlib.sha256(inputs[item["name"]]).hexdigest() == item["sha256"]
        prefix = item.get("source_prefix")
        if isinstance(prefix, dict):
            prefix.pop("host", None)
            prefix.pop("inode", None)
            if "path" in prefix:
                prefix["path"] = Path(prefix["path"]).name
    summaries, events = (
        [json.loads(line, parse_float=Decimal) for line in inputs[name].splitlines()]
        for name in ["summary.jsonl", "promotions.jsonl"]
    )
    rounds = [r["round"] for r in summaries]
    assert len(rounds) == 20 and rounds == list(range(rounds[0], rounds[-1] + 1))
    assert snapshot["round_window"] == [rounds[0], rounds[-1]]
    assert all(
        r["kind"] == "promotion"
        and rounds[0] <= r["round"] <= rounds[-1]
        and r["cv_gate_passed"] is True
        and r["evaluation_scope"] == "full_train_cv"
        and r["new_utility"] > r["old_utility"] + Decimal("0.000001")
        for r in events
    )
    assert len(
        {(r["round"], r["context_sha256"], r["program_sha256"]) for r in events}
    ) == len(events)
    counts, rows, cumulative, identities = (
        Counter(r["round"] for r in events),
        [],
        0,
        set(),
    )
    admissions = {
        int(r["round"]): r
        for r in csv.DictReader((FIGURES / "data/live_admissions.csv").open())
    }
    for r in summaries:
        a, c, b = r["adapter_admission"], r["candidate_dev_utility"], r["dev_utility"]
        assert r["round"] == a["round"] and counts[r["round"]] == r["promotions"]
        assert c["n_bin"] == b["n_bin"] == 0 and c["n_reg"] == b["n_reg"] > 0
        assert r["val_reg"] == c["val_reg"] == a["candidate_utility"]
        assert b["val_reg"] == b["mean"] == a["best_utility"]
        assert isinstance(a["accepted"], bool) and r["wall_seconds"].is_finite()
        assert (
            r["wall_seconds"] > 0
            and a["evaluation_identity"] == r["evaluation_identity"]
        )
        old = admissions[r["round"]]
        assert Decimal(old["candidate_val"]) == a["candidate_utility"]
        assert Decimal(old["best_val"]) == a["best_utility"]
        assert (
            int(old["accepted"]) == int(a["accepted"])
            and int(old["best_round"]) == a["best_round"]
        )
        identities.add(r["evaluation_identity"])
        cumulative += counts[r["round"]]
        rows.append(
            dict(
                zip(
                    FIELDS,
                    [
                        r["round"],
                        counts[r["round"]],
                        cumulative,
                        r["val_reg"],
                        a["best_utility"],
                        int(a["accepted"]),
                        r["wall_seconds"],
                        a["best_round"],
                    ],
                )
            )
        )
    assert len(identities) == 1
    wall = sum(r["completed_round_wall_seconds"] for r in rows)
    rate = Decimal(3600) * len(events) / wall
    assert summaries[-1]["promotions_rate_scope"] == (
        "up to 20 completed rounds since process start; includes qualification wait"
    )
    assert abs(rate - summaries[-1]["promotions_per_hour"]) < Decimal("1e-10")
    with DATA.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    PROVENANCE.write_text(
        json.dumps(
            {
                "schema": "whitepaper.live-development/v1",
                "snapshot": snapshot,
                "round_window": [rounds[0], rounds[-1]],
                "summary_rows": len(rows),
                "strict_program_promotions": len(events),
                "adapter_admissions": sum(r["adapter_accepted"] for r in rows),
                "reported_best_rounds": sorted(
                    {r["reported_best_round"] for r in rows}
                ),
                "completed_round_wall_seconds": str(wall),
                "rounds_per_hour": str(Decimal(3600) * len(rows) / wall),
                "program_promotions_per_hour": str(rate),
                "promotion_CI_scope": dict(
                    Counter(r.get("CI_scope", "unrecorded") for r in events)
                ),
                "evaluation_identities": sorted(identities),
                "csv_sha256": sha(DATA),
                "admission_csv_sha256": sha(FIGURES / "data/live_admissions.csv"),
                "generator_sha256": sha(Path(__file__)),
                "source_binding": "Aggregate field interpretation only; installed source/checkpoint qualification remains required",
                "scope": "Latest retained completed-round development telemetry; program events and adapter admissions are distinct, not independent first-proposal or deployment evidence",
                "reproduce": "python3 figures/scripts/make_live_development_figure.py",
            },
            indent=2,
        )
        + "\n"
    )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--logs", type=Path)
args = parser.parse_args()
if args.logs:
    extract(args.logs)
p = json.loads(PROVENANCE.read_text())
assert sha(DATA) == p["csv_sha256"]
rows = list(csv.DictReader(DATA.open()))
rounds = [int(r["round"]) for r in rows]
counts = [int(r["program_promotions"]) for r in rows]
cumulative = [int(r["cumulative_program_promotions"]) for r in rows]
assert len(rows) == 20 and rounds == list(range(rounds[0], rounds[-1] + 1))
assert cumulative == [sum(counts[: i + 1]) for i in range(len(counts))]
assert sum(counts) == p["strict_program_promotions"] and all(n >= 0 for n in counts)
candidate = [float(r["candidate_validation_utility"]) for r in rows]
best = [float(r["recorded_best_validation_utility"]) for r in rows]
assert all(
    Decimal(r[k]).is_finite()
    for r in rows
    for k in ["candidate_validation_utility", "recorded_best_validation_utility"]
)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.7), sharex=True)
axes[0].bar(rounds, counts, color="#5f3dc4", width=0.7)
axes[0].set_title("Top: Training-program promotions", fontsize=8, loc="left", pad=3)
axes[0].set_ylabel("promotions / round")
axes[0].set_ylim(-0.1, max(counts) + 0.5)
axes[0].yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
axes[1].plot(
    rounds, candidate, "o-", color="#6e7781", ms=2.5, lw=0.8, label="candidate adapter"
)
axes[1].step(
    rounds, best, where="post", color="#0b5394", lw=1.15, label="recorded best"
)
axes[1].set_title("Bottom: Adapter validation", fontsize=8, loc="left", pad=3)
axes[1].set_ylabel("validation\npool utility")
axes[1].set_ylim(min(candidate + best) - 0.004, max(candidate + best) + 0.0013)
axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
axes[1].legend(
    loc="lower left",
    ncol=2,
    fontsize=7.2,
    frameon=False,
    handlelength=1.3,
    handletextpad=0.3,
    columnspacing=0.7,
    borderaxespad=0.3,
)
axes[1].set_xlabel("round (latest retained window)")
axes[1].set_xticks(list(dict.fromkeys([rounds[0], *rounds[5::5], rounds[-1]])))
for ax in axes:
    ax.set_xlim(rounds[0] - 0.4, rounds[-1] + 0.4)
    ax.grid(axis="y", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)
fig.tight_layout(pad=0.4, h_pad=0.65)
fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)
if args.logs:
    p.update(figure_sha256=sha(OUTPUT), matplotlib_version=matplotlib.__version__)
    PROVENANCE.write_text(json.dumps(p, indent=2) + "\n")
print(f"{len(rows)} completed rounds; {sum(counts)} strict program promotions")
