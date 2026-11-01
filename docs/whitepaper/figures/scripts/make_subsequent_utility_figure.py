#!/usr/bin/env python3
"""Show logged utility on its original scale for the same rows as Figure 20."""

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data" / "subsequent_window.csv"
PARENT = DATA.with_suffix(".provenance.json")
OUTPUT = FIGURES / "fig_fw_subsequent_utility.pdf"
PROVENANCE = FIGURES / "data" / "subsequent_utility.provenance.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parent = json.loads(PARENT.read_text())
assert sha(DATA) == parent["csv_sha256"]
with DATA.open() as stream:
    rows = list(csv.DictReader(stream))
rounds = [int(row["round"]) for row in rows]
candidate = [float(row["candidate_validation_utility"]) for row in rows]
best = [float(row["recorded_best_validation_utility"]) for row in rows]
assert len(rows) == parent["summary_rows"] == 20
assert rounds == list(range(rounds[0], rounds[-1] + 1))
assert parent["round_window"] == [rounds[0], rounds[-1]]
assert all(math.isfinite(v) for v in candidate + best)
assert (
    sorted({int(row["reported_best_round"]) for row in rows})
    == parent["reported_best_rounds"]
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
fig, ax = plt.subplots(figsize=(3.4, 2.1))
ax.plot(
    rounds, candidate, "o-", color="#6e7781", ms=2.5, lw=0.8, label="candidate adapter"
)
ax.step(rounds, best, where="post", color="#0b5394", lw=1.1, label="recorded best")
ax.set_title(
    "Logged candidate and recorded-best utility", fontsize=8, loc="left", pad=3
)
ax.set_ylabel("validation pool utility")
ax.set_xlabel("round (same retained window)")
ax.set_ylim(min(candidate + best) - 0.001, max(candidate + best) + 0.001)
ax.set_xlim(rounds[0] - 0.4, rounds[-1] + 0.4)
ax.set_xticks(list(dict.fromkeys([rounds[0], *rounds[5::5], rounds[-1]])))
ax.ticklabel_format(axis="y", style="plain", useOffset=False)
ax.grid(axis="y", alpha=0.22, linewidth=0.4)
ax.set_axisbelow(True)
ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.0, 0.88),
    ncol=2,
    fontsize=7.2,
    frameon=False,
    handlelength=1.3,
    handletextpad=0.3,
    columnspacing=0.7,
)
fig.tight_layout(pad=0.4)
fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)
provenance = {
    "schema": "whitepaper.subsequent-utility-view/v1",
    "source_csv": str(DATA.relative_to(FIGURES)),
    "csv_sha256": sha(DATA),
    "parent_provenance_sha256": sha(PARENT),
    "snapshot_at": parent["snapshot"]["at"],
    "round_window": parent["round_window"],
    "summary_rows": len(rows),
    "columns_used": [
        "round",
        "candidate_validation_utility",
        "recorded_best_validation_utility",
    ],
    "reported_best_rounds": parent["reported_best_rounds"],
    "selection": "All rows of the committed subsequent-window CSV; same observations as Figure 20",
    "scope": "An alternate utility-scale view, with no additional observations or independent comparison",
    "generator_sha256": sha(Path(__file__)),
    "figure_sha256": sha(OUTPUT),
    "matplotlib_version": matplotlib.__version__,
}
PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
print(f"{len(rows)} existing summary rows; no additional observations")
