#!/usr/bin/env python3
"""Plot logged rolling/per-round promotions per hour and candidate vs. recorded-best validation utility."""

import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data/promotions_per_hour.csv"
PROVENANCE = DATA.with_suffix(".provenance.json")
PARENT_CSV = FIGURES / "data/live_development.csv"
PARENT_PROV = FIGURES / "data/live_development.provenance.json"
OUTPUT = FIGURES / "fig_fw_promotions_per_hour.pdf"
FIELDS = [
    "round",
    "completed_round_wall_seconds",
    "program_promotions",
    "round_promotions_per_hour",
    "rolling_promotions_per_hour",
    "candidate_validation_utility",
    "recorded_best_validation_utility",
    "adapter_accepted",
    "reported_best_round",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(logs: Path) -> None:
    snapshot = json.loads((logs / "snapshot.json").read_text())
    snapshot.pop("host", None)
    raw_summary = (logs / "summary.jsonl").read_bytes()
    summary_sha = hashlib.sha256(raw_summary).hexdigest()
    for item in snapshot.get("files", []):
        if item.get("name") == "summary.jsonl":
            assert item["sha256"] == summary_sha
        prefix = item.get("source_prefix")
        if isinstance(prefix, dict):
            prefix.pop("host", None)
            prefix.pop("inode", None)
            if "path" in prefix:
                prefix["path"] = Path(prefix["path"]).name

    summaries = [
        json.loads(line, parse_float=Decimal) for line in raw_summary.splitlines()
    ]
    parent_rows = {
        int(r["round"]): r for r in csv.DictReader(PARENT_CSV.open())
    }
    assert len(summaries) == len(parent_rows) == 20
    rounds = [int(r["round"]) for r in summaries]
    assert rounds == list(range(rounds[0], rounds[-1] + 1))

    out_rows = []
    for r in summaries:
        rnd = int(r["round"])
        pr = parent_rows[rnd]
        wall = Decimal(r["wall_seconds"])
        promos = int(r["promotions"])
        rolling = Decimal(r["promotions_per_hour"])
        cand = Decimal(r["val_reg"])
        best = Decimal(r["dev_utility"]["val_reg"])
        accepted = int(bool(r["adapter_admission"]["accepted"]))
        best_round = int(r["adapter_admission"]["best_round"])

        assert wall == Decimal(pr["completed_round_wall_seconds"])
        assert promos == int(pr["program_promotions"])
        assert cand == Decimal(pr["candidate_validation_utility"])
        assert best == Decimal(pr["recorded_best_validation_utility"])
        assert accepted == int(pr["adapter_accepted"])
        assert best_round == int(pr["reported_best_round"])
        assert r["promotions_rate_scope"] == (
            "up to 20 completed rounds since process start; includes qualification wait"
        )
        round_rate = Decimal(3600) * promos / wall
        out_rows.append(
            {
                "round": rnd,
                "completed_round_wall_seconds": str(wall),
                "program_promotions": promos,
                "round_promotions_per_hour": f"{round_rate:.12f}",
                "rolling_promotions_per_hour": str(rolling),
                "candidate_validation_utility": str(cand),
                "recorded_best_validation_utility": str(best),
                "adapter_accepted": accepted,
                "reported_best_round": best_round,
            }
        )

    with DATA.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--logs", type=Path)
args = parser.parse_args()
if args.logs:
    extract(args.logs)

parent_prov = json.loads(PARENT_PROV.read_text())
assert sha(PARENT_CSV) == parent_prov["csv_sha256"]
parent_by_round = {int(r["round"]): r for r in csv.DictReader(PARENT_CSV.open())}

rows = list(csv.DictReader(DATA.open()))
rounds = [int(r["round"]) for r in rows]
assert len(rows) == 20 and rounds == list(range(rounds[0], rounds[-1] + 1))
assert parent_prov["round_window"] == [rounds[0], rounds[-1]]

walls = [Decimal(r["completed_round_wall_seconds"]) for r in rows]
promos = [int(r["program_promotions"]) for r in rows]
round_rates = [Decimal(r["round_promotions_per_hour"]) for r in rows]
rolling_rates = [Decimal(r["rolling_promotions_per_hour"]) for r in rows]
candidates = [Decimal(r["candidate_validation_utility"]) for r in rows]
bests = [Decimal(r["recorded_best_validation_utility"]) for r in rows]

for r, w, p_cnt, rr, roll, c, b in zip(
    rows, walls, promos, round_rates, rolling_rates, candidates, bests
):
    pr = parent_by_round[int(r["round"])]
    assert w == Decimal(pr["completed_round_wall_seconds"]) and w > 0
    assert p_cnt == int(pr["program_promotions"]) and p_cnt >= 0
    assert abs(rr - (Decimal(3600) * p_cnt / w)) < Decimal("1e-9")
    assert roll.is_finite() and roll >= 0
    assert c == Decimal(pr["candidate_validation_utility"])
    assert b == Decimal(pr["recorded_best_validation_utility"])
    assert r["adapter_accepted"] == pr["adapter_accepted"] == "0"

total_wall = sum(walls)
total_promos = sum(promos)
window_rate = Decimal(3600) * total_promos / total_wall
assert abs(window_rate - rolling_rates[-1]) < Decimal("1e-10")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2))

# Left panel: Promotions per hour over rounds
axes[0].bar(
    rounds,
    [float(v) for v in round_rates],
    width=0.62,
    color="#d0bfff",
    edgecolor="#5f3dc4",
    linewidth=0.5,
    label="per-round rate",
    zorder=2,
)
axes[0].plot(
    rounds,
    [float(v) for v in rolling_rates],
    "o-",
    color="#5f3dc4",
    ms=2.7,
    lw=1.1,
    label="logged rolling rate (≤20 rds)",
    zorder=4,
)
axes[0].axhline(
    float(window_rate),
    color="#2b8a3e",
    ls="--",
    lw=0.85,
    label=f"window rate ({float(window_rate):.1f}/hr)",
    zorder=3,
)
axes[0].set_title("Training-program promotions per hour", loc="left", fontsize=8)
axes[0].set_xlabel("round (retained window 2644–2663)")
axes[0].set_ylabel("promotions per hour")
axes[0].set_ylim(0, max(float(v) for v in round_rates) * 1.18)
axes[0].legend(loc="upper right", frameon=False, handlelength=1.4)

# Right panel: Candidate validation utility vs recorded best over rounds
axes[1].plot(
    rounds,
    [float(v) for v in candidates],
    "o-",
    color="#6e7781",
    ms=2.7,
    lw=0.85,
    label="candidate adapter",
)
axes[1].step(
    rounds,
    [float(v) for v in bests],
    where="post",
    color="#0b5394",
    lw=1.15,
    label="recorded best (round 882)",
)
axes[1].set_title("Candidate vs. recorded-best validation", loc="left", fontsize=8)
axes[1].set_xlabel("round (retained window 2644–2663)")
axes[1].set_ylabel("validation pool utility")
axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
vals = [float(v) for v in candidates + bests]
axes[1].set_ylim(min(vals) - 0.004, max(vals) + 0.0014)
axes[1].legend(loc="lower left", frameon=False, handlelength=1.4)

xticks = list(dict.fromkeys([rounds[0], *rounds[5::5], rounds[-1]]))
for ax in axes:
    ax.set_xticks(xticks)
    ax.set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)
    ax.grid(axis="y", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)

fig.tight_layout(pad=0.45, w_pad=1.2)
fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)

provenance = {
    "schema": "whitepaper.promotions-per-hour/v1",
    "source_csv": "figures/data/live_development.csv",
    "source_csv_sha256": sha(PARENT_CSV),
    "parent_provenance": "figures/data/live_development.provenance.json",
    "parent_provenance_sha256": sha(PARENT_PROV),
    "csv": "figures/data/promotions_per_hour.csv",
    "csv_sha256": sha(DATA),
    "figure": "figures/fig_fw_promotions_per_hour.pdf",
    "figure_sha256": sha(OUTPUT),
    "generator": "figures/scripts/make_promotions_per_hour_figure.py",
    "generator_sha256": sha(Path(__file__)),
    "round_window": [rounds[0], rounds[-1]],
    "rows": len(rows),
    "strict_program_promotions": total_promos,
    "completed_round_wall_seconds": str(total_wall),
    "initial_rolling_promotions_per_hour": str(rolling_rates[0]),
    "final_rolling_promotions_per_hour": str(rolling_rates[-1]),
    "window_promotions_per_hour": str(window_rate),
    "adapter_admissions": 0,
    "reported_best_round": 882,
    "metrics": {
        "round_promotions_per_hour": "3600 * program_promotions / completed_round_wall_seconds for each completed round",
        "rolling_promotions_per_hour": "Logged summary.promotions_per_hour over up to 20 completed rounds since process start, including qualification wait",
        "validation_utility": "Logged summary.val_reg (candidate adapter) versus dev_utility.val_reg (recorded best)",
    },
    "scope": "Development operating telemetry from rounds 2644-2663; program promotions and adapter admissions are distinct events, not independent first-proposal or deployment comparisons",
    "matplotlib_version": matplotlib.__version__,
    "reproduce": "python3 figures/scripts/make_promotions_per_hour_figure.py",
}
PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
print(
    f"Plotted {len(rows)} rounds ({rounds[0]}..{rounds[-1]}); "
    f"rolling rate {float(rolling_rates[0]):.2f} -> {float(rolling_rates[-1]):.2f}/hr"
)
