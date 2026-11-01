#!/usr/bin/env python3
"""Plot source-bound adapter admission frequencies in observed-round blocks."""
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data" / "admission_block_rates.csv"
SOURCE = DATA.with_name("admission_block_rates.source.json")
CHECKSUM = DATA.with_suffix(".csv.sha256")
OUTPUT = FIGURES / "fig_fw_admission_block_rates.pdf"
PROVENANCE = DATA.with_name("admission_block_rates.provenance.json")


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


def blocks(rows):
    result = []
    for first in range(0, len(rows), 100):
        part = rows[first:first + 100]
        accepted = [r for r in part if r["accepted"]]
        ties = sum(r["best"] == r["previous"] for r in accepted)
        gains = len(accepted) - ties
        rejected = len(part) - len(accepted)
        result.append({
            "first_round": part[0]["round"], "last_round": part[-1]["round"],
            "observed_rounds": len(part), "accepted_ties": ties,
            "accepted_strict_gains": gains, "rejected": rejected,
            "tie_percent": str(Decimal(100 * ties) / len(part)),
            "strict_gain_percent": str(Decimal(100 * gains) / len(part)),
            "rejected_percent": str(Decimal(100 * rejected) / len(part)),
        })
    return result


def main():
    source, rows = load()
    if source["capture"]["block_size"] != 100:
        raise ValueError("expected fixed 100-round blocks")
    grouped = blocks(rows)
    ties = [float(b["tie_percent"]) for b in grouped]
    gains = [float(b["strict_gain_percent"]) for b in grouped]
    rejected = [float(b["rejected_percent"]) for b in grouped]
    x = list(range(len(grouped)))
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 1.85), sharex=True)
    left, right = axes
    left.bar(x, gains, color="#0b5394", width=0.75, label="strict gain")
    bars = left.bar(x, ties, bottom=gains, color="#d28722", width=0.75,
                    label="exact tie")
    left.set_ylabel("Accepted decisions (%)")
    left.set_ylim(0, max(t + g for t, g in zip(ties, gains)) + 0.6)
    left.legend(loc="upper right", fontsize=7, frameon=False)
    right.plot(x, rejected, ".-", color="#626b75", ms=3.0, lw=0.8)
    right.set_ylabel("Rejected decisions (%)")
    right.set_ylim(96, 100.3)
    right.set_yticks([96, 97, 98, 99, 100])
    if grouped[-1]["observed_rounds"] != 100:
        bars[-1].set_hatch("///")
        bars[-1].set_edgecolor("#5b3c0d")
        right.scatter(x[-1], rejected[-1], marker="D", s=17,
                      facecolors="white", edgecolors="#626b75", zorder=4)
    ticks = sorted({0, len(x) // 3, 2 * len(x) // 3, len(x) - 1})
    for ax in axes:
        ax.set_xticks(ticks, [str(grouped[i]["first_round"]) for i in ticks])
        ax.set_xlabel("Block start round")
        ax.set_xlim(-0.7, len(x) - 0.3)
        ax.grid(axis="y", alpha=0.2, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.45, w_pad=1.0)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    provenance = {
        "schema": "whitepaper.admission-block-rates-figure/v1",
        "input_csv_sha256": digest(DATA), "source_receipt_sha256": digest(SOURCE),
        "generator_sha256": digest(Path(__file__)), "figure_sha256": digest(OUTPUT),
        "rows": len(rows), "round_window": [rows[0]["round"], rows[-1]["round"]],
        "block_size": 100, "blocks": grouped,
        "accepted_ties": sum(b["accepted_ties"] for b in grouped),
        "accepted_strict_gains": sum(b["accepted_strict_gains"] for b in grouped),
        "rejected": sum(b["rejected"] for b in grouped),
        "matplotlib_version": matplotlib.__version__,
        "scope": "Observed adaptive adapter admission frequencies, not independent probabilities or a learning-performance estimate",
        "reproduce": "python3 figures/scripts/make_admission_block_rates.py",
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + chr(10))
    print(f"{len(rows)} observed rounds in {len(grouped)} blocks")


if __name__ == "__main__":
    main()

