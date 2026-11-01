#!/usr/bin/env python3
"""Plot the committed, source-bound adapter admission trace."""
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data" / "candidate_retained_trace.csv"
SOURCE = DATA.with_name("candidate_retained_trace.source.json")
CHECKSUM = DATA.with_suffix(".csv.sha256")
OUTPUT = FIGURES / "fig_fw_candidate_retained_trace.pdf"
PROVENANCE = DATA.with_name("candidate_retained_trace.provenance.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    source = json.loads(SOURCE.read_text())
    expected = source["csv_sha256"]
    if digest(DATA) != expected:
        raise ValueError("CSV hash mismatch")
    if CHECKSUM.read_text().split() != [expected, DATA.name]:
        raise ValueError("checksum sidecar mismatch")
    with DATA.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    for row in raw:
        item = {
            "round": int(row["round"]),
            "candidate": Decimal(row["candidate_validation_utility"]),
            "previous": Decimal(row["previous_best_validation_utility"]),
            "best": Decimal(row["retained_best_validation_utility"]),
            "accepted": int(row["adapter_accepted"]),
            "best_round": int(row["reported_best_round"]),
        }
        if not all(item[k].is_finite() for k in ["candidate", "previous", "best"]):
            raise ValueError("nonfinite utility")
        if item["accepted"] not in [0, 1]:
            raise ValueError("invalid admission flag")
        if item["accepted"]:
            if item["candidate"] != item["best"] or item["best"] < item["previous"]:
                raise ValueError("accepted decision is inconsistent")
            if item["best_round"] != item["round"]:
                raise ValueError("accepted identity is inconsistent")
        elif item["best"] != item["previous"]:
            raise ValueError("rejected decision changes retained best")
        rows.append(item)
    if len(rows) < 2 or len(rows) != source["capture"]["rows"]:
        raise ValueError("row-count mismatch")
    rounds = [r["round"] for r in rows]
    if rounds != list(range(rounds[0], rounds[-1] + 1)):
        raise ValueError("rounds must be complete and consecutive")
    if [rounds[0], rounds[-1]] != source["capture"]["round_window"]:
        raise ValueError("round-window mismatch")
    if {r["evaluation_identity_sha256"] for r in raw} != {
        source["capture"]["evaluation_identity_sha256"]
    }:
        raise ValueError("mixed evaluation identities")
    for before, after in zip(rows, rows[1:]):
        if before["best"] != after["previous"]:
            raise ValueError("retained-best chain mismatch")
        if not after["accepted"] and before["best_round"] != after["best_round"]:
            raise ValueError("rejected decision changes identity")
    return source, rows


def main():
    _, rows = load()
    rounds = [r["round"] for r in rows]
    candidate = [float(r["candidate"]) for r in rows]
    best = [float(r["best"]) for r in rows]
    accepted = [r for r in rows if r["accepted"]]
    ties = [r for r in accepted if r["best"] == r["previous"]]
    gains = [r for r in accepted if r["best"] > r["previous"]]
    spans, start = [], None
    for row in rows:
        if not row["accepted"] and start is None:
            start = row["round"]
        if row["accepted"] and start is not None:
            spans.append([start, row["round"] - 1])
            start = None
    if start is not None:
        spans.append([start, rounds[-1]])
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.8, 2.35))
    ax.plot(rounds, candidate, ".-", color="#737b85", ms=2.2, lw=0.55,
            label="candidate adapter")
    ax.step(rounds, best, where="post", color="#0b5394", lw=1.35,
            label="retained best (post-decision)")
    if ties:
        ax.scatter([r["round"] for r in ties], [float(r["best"]) for r in ties],
                   marker="D", s=29, facecolors="white", edgecolors="#c66b00",
                   linewidths=1.0, zorder=5, label="accepted exact tie")
    if gains:
        ax.scatter([r["round"] for r in gains], [float(r["best"]) for r in gains],
                   marker="*", s=48, c="#b62828", zorder=5,
                   label="accepted strict gain")
    for first, last in spans:
        if last - first + 1 >= 50:
            ax.axvspan(first - 0.5, last + 0.5, color="#adb5bd", alpha=0.12, zorder=0)
            ax.text((first + last) / 2, 0.96, f"{last - first + 1} rejected rounds",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5)
    low, high = min(candidate + best), max(candidate + best)
    padding = max((high - low) * 0.18, 0.0004)
    ax.set_ylim(low - padding, high + padding)
    ax.set_xlim(rounds[0] - 4, rounds[-1] + 4)
    ax.set_xlabel("round")
    ax.set_ylabel("validation pool utility")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", ncol=3, frameon=True, fontsize=6.8,
              facecolor="white", framealpha=0.95, edgecolor="none")
    fig.tight_layout(pad=0.45)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    provenance = {
        "schema": "whitepaper.candidate-retained-figure/v1",
        "input_csv_sha256": digest(DATA),
        "source_receipt_sha256": digest(SOURCE),
        "generator_sha256": digest(Path(__file__)),
        "figure_sha256": digest(OUTPUT),
        "rows": len(rows),
        "round_window": [rounds[0], rounds[-1]],
        "accepted_exact_ties": [r["round"] for r in ties],
        "accepted_strict_gains": [r["round"] for r in gains],
        "rejected_round_spans": spans,
        "final_retained_utility": str(rows[-1]["best"]),
        "matplotlib_version": matplotlib.__version__,
        "scope": "Adaptive adapter admission telemetry, not independent first-proposal or deployment performance",
        "reproduce": "python3 figures/scripts/make_candidate_retained_trace.py",
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + chr(10))
    print(f"{len(rows)} complete rows; {len(ties)} accepted ties; {len(gains)} strict gains")


if __name__ == "__main__":
    main()
