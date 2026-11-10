#!/usr/bin/env python3
"""Plot the cumulative count of recorded promotion-context identifiers."""
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/program_promotion_contexts.csv"
SOURCE = ROOT / "data/program_promotion_contexts.source.json"
OUTPUT = ROOT / "fig_program_promotion_contexts.pdf"
PROVENANCE = ROOT / "data/program_promotion_contexts.provenance.json"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

raw = DATA.read_bytes()
source = json.loads(SOURCE.read_text())
assert hashlib.sha256(raw).hexdigest() == source["csv_sha256"]
rows = list(csv.DictReader(io.StringIO(raw.decode())))
assert len(rows) == source["records"] == 3529
rounds = [int(r["round"]) for r in rows]
counts = [int(r["cumulative_context_identifiers"]) for r in rows]
assert [int(r["event_index"]) for r in rows] == list(range(1, len(rows) + 1))
assert [int(r["source_line"]) for r in rows] == list(range(183, 3712))
assert rounds == sorted(rounds) and [rounds[0], rounds[-1]] == [775, 5047]
assert counts[0] == 1 and counts[-1] == source["distinct_logged_context_identifiers"] == 975
assert all(b - a in (0, 1) for a, b in zip(counts, counts[1:]))

plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": 9.0,
    "axes.labelsize": 9.5, "xtick.labelsize": 9.0, "ytick.labelsize": 9.0,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(7.0, 2.15))
ax.scatter(rounds, counts, s=3.0, color="#0b5394", alpha=0.7, linewidths=0)
ax.set_xlabel("Recorded round")
ax.set_ylabel("Distinct recorded\ncontext identifiers")
ax.set_xlim(775, 5049)
ax.set_ylim(0, 1020)
ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax.grid(axis="y", alpha=0.22, linewidth=0.4)
ax.text(0.97, 0.12, "975 identifiers\n3,529 promotion records", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9.0)
fig.tight_layout(pad=0.6)
fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)

font = Path(font_manager.findfont("Liberation Serif", fallback_to_default=False))
provenance = {
    "schema": "whitepaper.program-promotion-contexts-view/v1",
    "source_csv": "figures/data/program_promotion_contexts.csv",
    "input_csv_sha256": sha(DATA), "source_receipt_sha256": sha(SOURCE),
    "generator_sha256": sha(Path(__file__)), "figure_sha256": sha(OUTPUT),
    "records": len(rows), "distinct_logged_context_identifiers": counts[-1],
    "selection_round_window": [775, 5049], "event_round_window": [775, 5047],
    "canvas_inches": [7.0, 2.15], "minimum_source_text_pt": 9.0,
    "font_family": "Liberation Serif", "font_file_sha256": sha(font),
    "matplotlib_version": matplotlib.__version__, "unknowns_imputed": False,
    "smoothing": False, "jitter": False, "connecting_lines": False,
    "definition": "One point per retained promotion record. Count distinct recorded context identifiers observed from the first selected record through the current record in source order.",
    "scope": source["scope"], "source_binding": source["source_binding"],
    "reproduce": "python3 figures/scripts/make_program_promotion_contexts.py",
}
PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
print(f"Plotted {len(rows)} retained promotion records and {counts[-1]} recorded identifiers.")
