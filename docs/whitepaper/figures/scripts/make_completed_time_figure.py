#!/usr/bin/env python3
"""Plot the committed end-of-round aggregates against accumulated round time."""
import csv
import hashlib
import io
import json
import math
from itertools import accumulate
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FIGURES = Path(__file__).resolve().parent.parent
SOURCE = FIGURES / "data" / "later_development.csv"
PARENT = SOURCE.with_suffix(".provenance.json")
OUTPUT = FIGURES / "fig_fw_completed_time.pdf"
PROVENANCE = FIGURES / "data" / "completed_time.provenance.json"

source_bytes = SOURCE.read_bytes()
parent_bytes = PARENT.read_bytes()
parent = json.loads(parent_bytes)
source_hash = hashlib.sha256(source_bytes).hexdigest()
assert source_hash == parent["csv_sha256"], "Source CSV/provenance mismatch"
rows = list(csv.DictReader(io.StringIO(source_bytes.decode())))
assert len(rows) >= 2
rounds = [int(row["round"]) for row in rows]
events = [int(row["program_promotions"]) for row in rows]
accepted = [int(row["adapter_accepted"]) for row in rows]
wall = [float(row["completed_round_wall_seconds"]) for row in rows]
candidate = [float(row["candidate_validation_utility"]) for row in rows]
best = [float(row["recorded_best_validation_utility"]) for row in rows]
assert rounds == list(range(rounds[0], rounds[-1] + 1))
assert all(n >= 0 for n in events) and set(accepted) <= {0, 1}
assert all(math.isfinite(t) and t > 0 for t in wall)
assert all(math.isfinite(v) for v in candidate + best)
assert all(a <= b for a, b in zip(best, best[1:]))
elapsed_seconds = list(accumulate(wall))
minutes = [value / 60 for value in elapsed_seconds]
cumulative_events = list(accumulate(events))
assert cumulative_events == [int(row["cumulative_program_promotions"]) for row in rows]
assert parent["round_window"] == [rounds[0], rounds[-1]]
assert parent["summary_rows"] == len(rows)
assert parent["strict_program_promotions"] == sum(events)
assert parent["adapter_admissions"] == sum(accepted)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.85), sharex=True)
axes[0].plot(minutes, candidate, "o-", color="#6e7781", ms=2.5, lw=0.8,
             label="candidate utility")
axes[0].step(minutes, best, where="post", color="#0b5394", lw=1.15,
             label="recorded best")
axes[0].set_title("Logged validation pool utility", fontsize=8, loc="left", pad=3)
axes[0].set_ylabel("logged validation\npool utility")
axes[0].set_ylim(min(candidate + best) - 0.004, max(candidate + best) + 0.0013)
axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
axes[0].legend(loc="lower left", ncol=2, fontsize=7, frameon=False,
               handlelength=1.3, handletextpad=0.3, columnspacing=0.7,
               borderaxespad=0.3)
axes[1].step([0, *minutes], [0, *cumulative_events], where="post",
             color="#5f3dc4", lw=1.1)
axes[1].plot([0, *minutes], [0, *cumulative_events], ".", color="#5f3dc4", ms=2.5)
axes[1].set_title("Strict training-program events", fontsize=8, loc="left", pad=3)
axes[1].set_ylabel("cumulative\nprogram events")
axes[1].set_ylim(-0.6, cumulative_events[-1] + max(1, cumulative_events[-1] * 0.1))
axes[1].yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
axes[1].xaxis.set_major_locator(MaxNLocator(nbins=6))
axes[1].set_xlabel("accumulated completed-round time (min)")
for ax in axes:
    ax.set_xlim(0, minutes[-1] * 1.02)
    ax.grid(axis="y", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)
fig.tight_layout(pad=0.4, h_pad=0.65)
fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)

provenance = {
    "schema": "whitepaper.completed-round-time/v1",
    "source_csv": "docs/whitepaper/figures/data/later_development.csv",
    "source_csv_sha256": source_hash,
    "parent_provenance_sha256": hashlib.sha256(parent_bytes).hexdigest(),
    "snapshot_at": parent["snapshot"]["at"],
    "round_window": [rounds[0], rounds[-1]],
    "summary_rows": len(rows),
    "strict_program_events": sum(events),
    "adapter_admissions": sum(accepted),
    "accumulated_completed_round_seconds": elapsed_seconds[-1],
    "x_definition": "Cumulative summary.wall_seconds at each round end / 60; includes qualification waits; not whole-run calendar time",
    "event_curve_definition": "Counts reported at round end; starts from zero for this window; not exact within-round event times or independent discoveries",
    "utility_definition": parent["metric_definition"],
    "source_binding": parent["source_binding"],
    "scope": "Retained development observations only; no independent first-proposal, transfer or deployment claim",
    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "figure_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    "matplotlib_version": matplotlib.__version__,
    "reproduce": "python3 figures/scripts/make_completed_time_figure.py",
}
PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
print(f"{len(rows)} completed rounds; {sum(events)} program events; "
      f"{sum(accepted)} adapter admissions; {elapsed_seconds[-1]:.1f} seconds")
