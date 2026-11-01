#!/usr/bin/env python3
"""Replay retained adapter utility and completed-round throughput; import with --logs DIR."""

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
DATA = FIGURES / "data/round_throughput.csv"
PROVENANCE = DATA.with_suffix(".provenance.json")
OUTPUT = FIGURES / "fig_fw_round_throughput.pdf"
FIELDS = [
    "round",
    "wall_seconds",
    "distinct_training_contexts",
    "logged_datasets_per_second",
    "candidate_validation_utility",
    "recorded_best_validation_utility",
    "adapter_accepted",
    "reported_best_round",
    "evaluation_identity",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(logs):
    snapshot = json.loads((logs / "snapshot.json").read_text())
    snapshot.pop("host", None)
    names = {"summary.jsonl", "admissions.jsonl"}
    assert {f["name"] for f in snapshot["files"]} == names
    for item in snapshot["files"]:
        assert sha(logs / item["name"]) == item["sha256"]
        prefix = item.get("source_prefix")
        if isinstance(prefix, dict):
            prefix.pop("host", None)
            prefix.pop("inode", None)
            if "path" in prefix:
                prefix["path"] = Path(prefix["path"]).name
    summaries = [
        json.loads(s) for s in (logs / "summary.jsonl").read_text().splitlines()
    ]
    admissions = {
        a["round"]: a
        for a in map(json.loads, (logs / "admissions.jsonl").read_text().splitlines())
    }
    rounds = [r["round"] for r in summaries]
    assert len(summaries) == len(admissions) == 20
    assert rounds == list(range(rounds[0], rounds[-1] + 1))
    assert snapshot["round_window"] == [rounds[0], rounds[-1]]
    for r in summaries:
        a = admissions[r["round"]]
        assert r["candidate_validation_utility"] == a["candidate_utility"]
        assert r["recorded_best_validation_utility"] == a["best_utility"]
        assert r["adapter_accepted"] == a["accepted"]
        assert r["reported_best_round"] == a["best_round"]
        assert r["evaluation_identity"] == a["evaluation_identity"]
        assert r["n_reg"] == 91 and r["n_bin"] == 0
    with DATA.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {k: int(r[k]) if k == "adapter_accepted" else r[k] for k in FIELDS}
            for r in summaries
        )
    PROVENANCE.write_text(
        json.dumps(
            {
                "schema": "whitepaper.round-throughput.v1",
                "csv_sha256": sha(DATA),
                "generator_sha256": sha(Path(__file__)),
                "source_snapshot": snapshot,
                "rows": len(summaries),
                "round_window": snapshot["round_window"],
                "adapter_admissions": sum(r["adapter_accepted"] for r in summaries),
                "metrics": {
                    "validation": "Candidate adapter utility and recorded best; larger is preferred in the admission log",
                    "throughput": "Distinct training contexts within each completed round / that round's entire logged wall_seconds; independently checked against datasets_per_second",
                },
                "scope": "Development operating observations; repeated contexts across rounds are not independent tasks. Runtime/checkpoint binding and independent first-proposal/deployment comparisons remain pending.",
                "reproduce": "python3 figures/scripts/make_round_throughput_figure.py",
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
provenance = json.loads(PROVENANCE.read_text())
assert sha(DATA) == provenance["csv_sha256"]
assert sha(Path(__file__)) == provenance["generator_sha256"]
rows = list(csv.DictReader(DATA.open()))
rounds = [int(r["round"]) for r in rows]
assert len(rows) == provenance["rows"] == 20
assert rounds == list(range(rounds[0], rounds[-1] + 1))
assert provenance["round_window"] == [rounds[0], rounds[-1]]
rates = []
for r in rows:
    wall = Decimal(r["wall_seconds"])
    contexts = int(r["distinct_training_contexts"])
    assert wall.is_finite() and wall > 0 and contexts > 0
    rate = Decimal(contexts) / wall
    assert abs(rate - Decimal(r["logged_datasets_per_second"])) < Decimal("1e-12")
    assert all(
        Decimal(r[k]).is_finite()
        for k in ["candidate_validation_utility", "recorded_best_validation_utility"]
    )
    assert r["adapter_accepted"] in {"0", "1"}
    rates.append(float(rate))
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
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.25))
axes[0].plot(
    rounds,
    [float(r["candidate_validation_utility"]) for r in rows],
    "o-",
    color="#6e7781",
    ms=2.7,
    lw=0.8,
    label="candidate adapter",
)
axes[0].step(
    rounds,
    [float(r["recorded_best_validation_utility"]) for r in rows],
    where="post",
    color="#0b5394",
    lw=1.1,
    label="recorded best",
)
axes[0].set_title("Adapter validation", loc="left", fontsize=8)
axes[0].set_ylabel("validation pool utility")
axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
utilities = [
    float(r[k])
    for r in rows
    for k in ["candidate_validation_utility", "recorded_best_validation_utility"]
]
axes[0].set_ylim(min(utilities) - 0.004, max(utilities) + 0.0013)
axes[0].legend(frameon=False, loc="lower left", fontsize=7.2)
axes[1].plot(rounds, rates, "o-", color="#2b8a3e", ms=2.7, lw=0.8)
axes[1].set_title("Completed-round throughput", loc="left", fontsize=8)
axes[1].set_ylabel("distinct training contexts /\nfull-round s")
axes[1].set_ylim(bottom=0)
for ax in axes:
    ax.set_xlabel("round (later retained window)")
    ax.set_xticks(list(dict.fromkeys([rounds[0], *rounds[5::5], rounds[-1]])))
    ax.set_xlim(rounds[0] - 0.4, rounds[-1] + 0.4)
    ax.grid(axis="y", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)
fig.tight_layout(pad=0.5, w_pad=1.1)
fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)
if args.logs:
    provenance.update(
        figure_sha256=sha(OUTPUT), matplotlib_version=matplotlib.__version__
    )
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
print(
    f"{len(rows)} retained rounds; throughput agrees with the logged count/time ratio"
)
