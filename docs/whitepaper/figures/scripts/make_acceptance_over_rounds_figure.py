#!/usr/bin/env python3
"""Plot recorded-best validation utility and cumulative adapter acceptance rate over rounds."""

import csv
import hashlib
import io
import json
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data/live_admissions.csv"
PARENT = FIGURES / "data/live_admissions.provenance.json"
PROVENANCE = FIGURES / "data/acceptance_over_rounds.provenance.json"
OUTPUT = FIGURES / "fig_fw_acceptance_over_rounds.pdf"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


raw_csv = DATA.read_bytes()
raw_parent = PARENT.read_bytes()
parent_meta = json.loads(raw_parent)
csv_digest = hashlib.sha256(raw_csv).hexdigest()
assert csv_digest == parent_meta["csv_sha256"], "CSV sha256 mismatch with parent provenance"

rows = list(csv.DictReader(io.StringIO(raw_csv.decode("utf-8"))))
rounds = [int(r["round"]) for r in rows]
assert rounds == list(range(rounds[0], rounds[-1] + 1))
assert len(rows) == parent_meta["rows"] == 1889

bests = [Decimal(r["best_val"]) for r in rows]
candidates = [Decimal(r["candidate_val"]) for r in rows]
accepted_flags = [int(r["accepted"]) for r in rows]

gains, ties, accepted_rounds = [], [], []
cum_accepted = []
cum_rate_pct = []
running_acc = 0

for i, (rnd, b, c, acc) in enumerate(zip(rounds, bests, candidates, accepted_flags)):
    assert acc in (0, 1)
    assert b.is_finite() and c.is_finite()
    if i > 0:
        assert b >= bests[i - 1]
    if acc == 1:
        assert i > 0 and c == b
        accepted_rounds.append(rnd)
        if b > bests[i - 1]:
            gains.append(i)
        else:
            ties.append(i)
    running_acc += acc
    cum_accepted.append(running_acc)
    cum_rate_pct.append(100.0 * running_acc / (i + 1))

assert accepted_rounds == parent_meta["accepted_rounds"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.15))

# Panel 1: Recorded-best validation utility over rounds
axes[0].step(
    rounds,
    [float(v) for v in bests],
    where="post",
    color="#0b5394",
    lw=1.15,
    label="recorded best",
)
axes[0].scatter(
    [rounds[i] for i in gains],
    [float(bests[i]) for i in gains],
    s=28,
    marker="*",
    color="#c92a2a",
    zorder=5,
    label=f"strict gains ({len(gains)})",
)
axes[0].scatter(
    [rounds[i] for i in ties],
    [float(bests[i]) for i in ties],
    s=18,
    marker="D",
    facecolors="none",
    edgecolors="#d9480f",
    lw=0.9,
    zorder=4,
    label=f"accepted ties ({len(ties)})",
)
axes[0].set_title("Recorded-best validation utility", loc="left", fontsize=8)
axes[0].set_xlabel("round")
axes[0].set_ylabel("recorded-best pool utility")
axes[0].set_xlim(rounds[0] - 15, rounds[-1] + 15)
axes[0].set_ylim(float(min(bests)) - 0.0025, float(max(bests)) + 0.003)
axes[0].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
axes[0].grid(axis="y", alpha=0.22, lw=0.4)
axes[0].legend(loc="lower right", frameon=False)

# Panel 2: Cumulative adapter acceptance rate over rounds
axes[1].plot(
    rounds,
    cum_rate_pct,
    color="#2b8a3e",
    lw=1.1,
    label="cumulative acceptance rate",
)
axes[1].scatter(
    [rounds[i] for i in gains],
    [cum_rate_pct[i] for i in gains],
    s=28,
    marker="*",
    color="#c92a2a",
    zorder=5,
    label=f"strict gains ({len(gains)})",
)
axes[1].scatter(
    [rounds[i] for i in ties],
    [cum_rate_pct[i] for i in ties],
    s=18,
    marker="D",
    facecolors="none",
    edgecolors="#d9480f",
    lw=0.9,
    zorder=4,
    label=f"accepted ties ({len(ties)})",
)
axes[1].set_title("Cumulative adapter acceptance rate", loc="left", fontsize=8)
axes[1].set_xlabel("round")
axes[1].set_ylabel("accepted share of rounds (%)")
axes[1].set_xlim(rounds[0] - 15, rounds[-1] + 15)
axes[1].set_ylim(-0.6, max(cum_rate_pct) * 1.12)
axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
axes[1].grid(axis="y", alpha=0.22, lw=0.4)
axes[1].text(
    0.97,
    0.84,
    f"peak: {max(cum_rate_pct):.2f}% (round {rounds[cum_rate_pct.index(max(cum_rate_pct))]})\n"
    f"final: {cum_rate_pct[-1]:.2f}% ({running_acc}/{len(rows):,})",
    transform=axes[1].transAxes,
    ha="right",
    va="top",
    fontsize=7.1,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.5),
)

fig.tight_layout(pad=0.45, w_pad=1.2)
fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)

provenance = {
    "schema": "whitepaper.acceptance-over-rounds-view/v1",
    "source_csv": "data/live_admissions.csv",
    "source_csv_sha256": csv_digest,
    "parent_provenance_sha256": hashlib.sha256(raw_parent).hexdigest(),
    "round_window": [rounds[0], rounds[-1]],
    "rows": len(rows),
    "accepted_rounds": accepted_rounds,
    "strict_gain_rounds": [rounds[i] for i in gains],
    "accepted_tie_rounds": [rounds[i] for i in ties],
    "initial_recorded_best": str(bests[0]),
    "final_recorded_best": str(bests[-1]),
    "peak_cumulative_acceptance_pct": round(max(cum_rate_pct), 6),
    "peak_cumulative_acceptance_round": rounds[cum_rate_pct.index(max(cum_rate_pct))],
    "last_acceptance_cumulative_pct": round(cum_rate_pct[rounds.index(accepted_rounds[-1])], 6),
    "final_cumulative_acceptance_pct": round(cum_rate_pct[-1], 6),
    "definition": "At round r (index k >= 1 from 775), cumulative acceptance rate is 100 * sum_{j<=k} accepted_j / k; recorded best is the post-decision best_val step trajectory.",
    "scope": "Alternate view of the 1,889 retained admission rounds (775-2663); adds no observations and makes no independent first-proposal or deployment claim.",
    "source_binding": parent_meta["source_binding"],
    "generator_sha256": sha(Path(__file__)),
    "figure_sha256": sha(OUTPUT),
    "matplotlib_version": matplotlib.__version__,
    "reproduce": "python3 figures/scripts/make_acceptance_over_rounds_figure.py",
}
PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
print(
    f"Plotted {len(rows)} rounds ({rounds[0]}..{rounds[-1]}); "
    f"peak acceptance {max(cum_rate_pct):.2f}%, final {cum_rate_pct[-1]:.2f}%"
)
