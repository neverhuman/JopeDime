#!/usr/bin/env python3
"""Plot the consolidated 3-panel earlier admission prefix (rounds 775-2663): trajectory, cumulative acceptance rate, and shortfall ECDF."""

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
gaps = [Decimal(r["utility_shortfall"]) for r in rows]
accepted_flags = [int(r["accepted"]) for r in rows]

gains, ties, accepted_rounds = [], [], []
cum_accepted = []
cum_rate_pct = []
running_acc = 0

for i, (rnd, b, c, g, acc) in enumerate(zip(rounds, bests, candidates, gaps, accepted_flags)):
    assert acc in (0, 1)
    assert b.is_finite() and c.is_finite() and g.is_finite()
    assert g == b - c and g >= 0
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
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)

fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.95))

# Panel 1: Candidate and post-decision recorded-best validation utility over rounds
axes[0].scatter(
    rounds,
    [float(v) for v in candidates],
    s=2.5,
    color="#6e7781",
    alpha=0.45,
    linewidths=0,
    label="candidate adapter",
)
axes[0].step(
    rounds,
    [float(v) for v in bests],
    where="post",
    color="#0b5394",
    lw=1.1,
    label="recorded best",
)
axes[0].scatter(
    [rounds[i] for i in gains],
    [float(bests[i]) for i in gains],
    s=24,
    marker="*",
    color="#c92a2a",
    zorder=5,
    label=f"strict gains ({len(gains)})",
)
axes[0].scatter(
    [rounds[i] for i in ties],
    [float(bests[i]) for i in ties],
    s=15,
    marker="D",
    facecolors="none",
    edgecolors="#d9480f",
    lw=0.85,
    zorder=4,
    label=f"accepted ties ({len(ties)})",
)
axes[0].set_title("(a) Candidate & recorded-best utility", loc="left", fontsize=7.5)
axes[0].set_xlabel("round (775–2663)")
axes[0].set_ylabel("validation pool utility")
axes[0].set_xlim(rounds[0] - 15, rounds[-1] + 15)
axes[0].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
axes[0].grid(axis="y", alpha=0.22, lw=0.4)
axes[0].legend(
    loc="lower left",
    frameon=True,
    facecolor="white",
    framealpha=0.88,
    edgecolor="none",
    handletextpad=0.3,
    borderpad=0.2,
)

# Panel 2: Cumulative adapter acceptance rate over rounds
axes[1].plot(
    rounds,
    cum_rate_pct,
    color="#2b8a3e",
    lw=1.05,
    label="cumulative rate",
)
axes[1].scatter(
    [rounds[i] for i in gains],
    [cum_rate_pct[i] for i in gains],
    s=24,
    marker="*",
    color="#c92a2a",
    zorder=5,
    label=f"strict gains ({len(gains)})",
)
axes[1].scatter(
    [rounds[i] for i in ties],
    [cum_rate_pct[i] for i in ties],
    s=15,
    marker="D",
    facecolors="none",
    edgecolors="#d9480f",
    lw=0.85,
    zorder=4,
    label=f"accepted ties ({len(ties)})",
)
axes[1].set_title("(b) Cumulative acceptance rate", loc="left", fontsize=7.5)
axes[1].set_xlabel("round (775–2663)")
axes[1].set_ylabel("accepted share of rounds (%)")
axes[1].set_xlim(rounds[0] - 15, rounds[-1] + 15)
axes[1].set_ylim(-0.6, max(cum_rate_pct) * 1.12)
axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
axes[1].grid(axis="y", alpha=0.22, lw=0.4)
axes[1].text(
    0.97,
    0.86,
    f"peak: {max(cum_rate_pct):.2f}% (rd {rounds[cum_rate_pct.index(max(cum_rate_pct))]})\n"
    f"final: {cum_rate_pct[-1]:.2f}% ({running_acc}/{len(rows):,})",
    transform=axes[1].transAxes,
    ha="right",
    va="top",
    fontsize=6.5,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.2),
)

# Panel 3: Empirical CDF of candidate utility shortfall (recorded best - candidate)
axes[2].step(
    [0.0, *sorted(float(v) for v in gaps)],
    [0.0, *[(i + 1) / len(rows) for i in range(len(rows))]],
    where="post",
    color="#0b5394",
    lw=1.1,
)
axes[2].set_title("(c) Candidate shortfall ECDF", loc="left", fontsize=7.5)
axes[2].set_xlabel("recorded best − candidate utility")
axes[2].set_ylabel("share of logged rounds")
axes[2].set_xlim(0, float(max(gaps)) * 1.05)
axes[2].set_ylim(0, 1.03)
axes[2].xaxis.set_major_locator(MaxNLocator(nbins=4))
axes[2].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axes[2].grid(axis="y", alpha=0.22, lw=0.4)
axes[2].text(
    0.97,
    0.10,
    f"{len(rows):,} logged rounds\nlast accept: rd {accepted_rounds[-1]}",
    transform=axes[2].transAxes,
    ha="right",
    fontsize=6.5,
)

fig.tight_layout(pad=0.35, w_pad=0.85)
fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)

provenance = {
    "schema": "whitepaper.acceptance-over-rounds-view/v2",
    "source_csv": "figures/data/live_admissions.csv",
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
    "definition": "Consolidated 3-panel view of the 1,889-round prefix (775-2663): (a) candidate and post-decision recorded-best utility, (b) cumulative acceptance rate 100 * sum_{j<=k} accepted_j / k, (c) empirical CDF of utility_shortfall = best_val - candidate_val.",
    "scope": "Consolidated view of the 1,889 retained admission rounds (775-2663); adds no observations and makes no independent first-proposal or deployment claim.",
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
