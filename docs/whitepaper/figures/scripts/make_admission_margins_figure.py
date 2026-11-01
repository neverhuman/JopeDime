#!/usr/bin/env python3
"""Plot candidate-minus-retained validation margins per round (accepted vs rejected) and near-zero tie breakdown."""

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
DATA = FIGURES / "data/latest_admissions.csv"
SOURCE = FIGURES / "data/latest_admissions.source.json"
PROVENANCE = FIGURES / "data/admission_margins.provenance.json"
OUTPUT = FIGURES / "fig_fw_admission_margins.pdf"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    raw_csv = DATA.read_bytes()
    raw_source = SOURCE.read_bytes()
    source_meta = json.loads(raw_source)
    csv_digest = hashlib.sha256(raw_csv).hexdigest()
    assert csv_digest == source_meta["csv_sha256"], "CSV sha256 mismatch with source receipt"

    rows = list(csv.DictReader(io.StringIO(raw_csv.decode("utf-8"))))
    rounds = [int(r["round"]) for r in rows]
    assert rounds == list(range(rounds[0], rounds[-1] + 1))
    assert len(rows) == source_meta["capture"]["source_rows"] == 2063

    candidates = [Decimal(r["candidate_val"]) for r in rows]
    previous = [Decimal(r["previous_best_val"]) for r in rows]
    bests = [Decimal(r["best_val"]) for r in rows]
    accepted_flags = [int(r["accepted"]) for r in rows]
    decisions = [r["decision"] for r in rows]

    margins = [c - p for c, p in zip(candidates, previous)]

    gains_idx, ties_idx, rej_idx = [], [], []
    for i, (m, acc, dec) in enumerate(zip(margins, accepted_flags, decisions)):
        assert m.is_finite()
        assert acc in (0, 1)
        assert (acc == 1) == (m >= Decimal("0"))
        if acc == 1:
            if m > Decimal("0"):
                assert dec == "accept_improvement"
                gains_idx.append(i)
            else:
                assert m == Decimal("0")
                assert dec == "accept_tie"
                ties_idx.append(i)
        else:
            assert dec == "restore_best_model_and_optimizer"
            rej_idx.append(i)

    assert [rounds[i] for i in gains_idx] == [785, 859, 2750]
    assert [rounds[i] for i in ties_idx] == [782, 882, 2795]
    assert len(rej_idx) == 2057

    gains_ge_1e3 = [i for i in gains_idx if margins[i] >= Decimal("0.001")]
    gains_sub_1e3 = [i for i in gains_idx if Decimal("0") < margins[i] < Decimal("0.001")]
    rej_band_1e4 = [i for i in rej_idx if margins[i] >= Decimal("-0.0001")]
    rej_band_1e3 = [
        i for i in rej_idx if Decimal("-0.001") <= margins[i] < Decimal("-0.0001")
    ]
    rej_below_1e3 = [i for i in rej_idx if margins[i] < Decimal("-0.001")]

    assert len(gains_ge_1e3) == 1 and len(gains_sub_1e3) == 2
    assert len(rej_band_1e4) == 26
    assert len(rej_band_1e3) == 48
    assert len(rej_below_1e3) == 1983
    assert len(rej_band_1e4) + len(rej_band_1e3) + len(rej_below_1e3) == len(rej_idx)

    rej_margins_sorted = sorted(margins[i] for i in rej_idx)
    median_rej = rej_margins_sorted[len(rej_margins_sorted) // 2]
    min_rej_i = min(rej_idx, key=lambda i: margins[i])
    max_rej_i = max(rej_idx, key=lambda i: margins[i])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
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

    fig, axes = plt.subplots(
        1, 2, figsize=(6.8, 2.2), gridspec_kw={"width_ratios": [1.38, 1.0]}
    )

    # Panel 1: Pre-decision margin (candidate - previous retained best) across rounds
    ax0 = axes[0]
    ax0.axhline(0.0, color="#212529", lw=0.85, ls="--", zorder=2, label="threshold (0)")
    ax0.axhspan(-0.001, 0.001, color="#fff3bf", alpha=0.55, zorder=1, label=r"near-zero ($\pm 10^{-3}$)")

    ax0.scatter(
        [rounds[i] for i in rej_idx],
        [float(margins[i]) for i in rej_idx],
        s=4.5,
        color="#7c858e",
        alpha=0.45,
        linewidths=0,
        zorder=3,
        label=f"rejected (< 0; n={len(rej_idx):,})",
    )
    ax0.scatter(
        [rounds[i] for i in gains_sub_1e3],
        [float(margins[i]) for i in gains_sub_1e3],
        s=34,
        marker="*",
        color="#c92a2a",
        zorder=6,
        label=f"strict gains ({len(gains_idx)})",
    )
    ax0.scatter(
        [rounds[i] for i in ties_idx],
        [float(margins[i]) for i in ties_idx],
        s=22,
        marker="D",
        facecolors="none",
        edgecolors="#d9480f",
        lw=1.0,
        zorder=7,
        label=f"exact ties ({len(ties_idx)})",
    )

    # Off-scale annotation for round 785 (+0.030218)
    r785 = rounds[gains_ge_1e3[0]]
    m785 = margins[gains_ge_1e3[0]]
    ax0.annotate(
        f"r{r785}: +{m785:.4f} (off-scale)",
        xy=(r785, 0.00165),
        xytext=(960, 0.00122),
        fontsize=6.6,
        color="#c92a2a",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color="#c92a2a", lw=0.7),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.6),
    )
    # Near-zero gain annotations
    ax0.annotate(
        "r859 (+2.0e-4)",
        xy=(859, float(margins[gains_sub_1e3[0]])),
        xytext=(1080, -0.00095),
        fontsize=6.5,
        color="#c92a2a",
        arrowprops=dict(arrowstyle="->", color="#c92a2a", lw=0.6),
    )
    ax0.annotate(
        "r2750 (+5.7e-4)",
        xy=(2750, float(margins[gains_sub_1e3[1]])),
        xytext=(2560, -0.00095),
        ha="right",
        va="center",
        fontsize=6.5,
        color="#c92a2a",
        arrowprops=dict(arrowstyle="->", color="#c92a2a", lw=0.6),
    )

    ax0.set_title("Candidate minus retained-best utility", loc="left", fontsize=8)
    ax0.set_xlabel("round")
    ax0.set_ylabel("pre-decision margin")
    ax0.set_xlim(rounds[0] - 15, rounds[-1] + 15)
    ax0.set_ylim(-0.0108, 0.00185)
    ax0.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax0.grid(axis="y", alpha=0.22, lw=0.4)
    ax0.legend(
        loc="lower left",
        bbox_to_anchor=(0.01, 0.02),
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="#dee2e6",
        ncol=2,
        fontsize=6.2,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.7,
        borderpad=0.3,
    )

    # Panel 2: Margin regime breakdown (accepted ties/near-zero vs rejected bands)
    ax1 = axes[1]
    categories = [
        r"Accepted gain $\geq 10^{-3}$",
        r"Accepted gain $(0, 10^{-3})$",
        r"Accepted exact tie ($0$)",
        r"Rejected $[-10^{-4}, 0)$",
        r"Rejected $[-10^{-3}, -10^{-4})$",
        r"Rejected $< -10^{-3}$",
    ]
    counts = [
        len(gains_ge_1e3),
        len(gains_sub_1e3),
        len(ties_idx),
        len(rej_band_1e4),
        len(rej_band_1e3),
        len(rej_below_1e3),
    ]
    colors = ["#c92a2a", "#e03131", "#d9480f", "#495057", "#868e96", "#adb5bd"]
    y_pos = list(range(len(categories)))[::-1]

    bars = ax1.barh(y_pos, counts, color=colors, height=0.62, edgecolor="none")
    ax1.set_xscale("symlog", linthresh=10)
    ax1.set_xlim(0, 6500)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(categories, fontsize=6.9)
    ax1.axhline(2.5, color="#212529", lw=0.65, ls=":")
    ax1.set_title("Margin regime round counts", loc="left", fontsize=8)
    ax1.set_xlabel("rounds (symlog scale)")
    ax1.grid(axis="x", alpha=0.22, lw=0.4)

    notes = [
        "1 (r785: +0.0302)",
        "2 (r859, r2750)",
        "3 (r782, r882, r2795)",
        "26 (max -4.3e-5)",
        "48 (74 in [-1e-3, 0))",
        "1,983 (med -0.0034)",
    ]
    for bar, note in zip(bars, notes):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2.0
        if w >= 500:
            ax1.text(
                w * 0.82,
                y,
                note,
                va="center",
                ha="right",
                fontsize=6.4,
                color="#212529",
            )
        else:
            ax1.text(
                w * 1.15 if w >= 10 else w + 0.55,
                y,
                note,
                va="center",
                ha="left",
                fontsize=6.4,
                color="#212529",
            )

    fig.tight_layout(pad=0.45, w_pad=1.5)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    provenance = {
        "schema": "whitepaper.admission-margins-figure/v1",
        "source_csv": "figures/data/latest_admissions.csv",
        "source_csv_sha256": csv_digest,
        "source_receipt_sha256": hashlib.sha256(raw_source).hexdigest(),
        "generator_sha256": sha(Path(__file__)),
        "figure_sha256": sha(OUTPUT),
        "round_window": [rounds[0], rounds[-1]],
        "rows": len(rows),
        "accepted_rows": len(gains_idx) + len(ties_idx),
        "accepted_strict_gain_rounds": [rounds[i] for i in gains_idx],
        "accepted_strict_gains_ge_1e_3_rounds": [rounds[i] for i in gains_ge_1e3],
        "accepted_strict_gains_sub_1e_3_rounds": [rounds[i] for i in gains_sub_1e3],
        "accepted_exact_tie_rounds": [rounds[i] for i in ties_idx],
        "accepted_strict_gain_margins": {
            str(rounds[i]): str(margins[i]) for i in gains_idx
        },
        "rejected_rows": len(rej_idx),
        "rejected_in_minus_1e_4_to_0": len(rej_band_1e4),
        "rejected_in_minus_1e_3_to_minus_1e_4": len(rej_band_1e3),
        "rejected_below_minus_1e_3": len(rej_below_1e3),
        "closest_rejected_round": rounds[max_rej_i],
        "closest_rejected_margin": str(margins[max_rej_i]),
        "min_rejected_round": rounds[min_rej_i],
        "min_rejected_margin": str(margins[min_rej_i]),
        "median_rejected_margin": str(median_rej),
        "definition": "Pre-decision validation margin Delta_r = candidate_val - previous_best_val in exact Decimal; accepted iff Delta_r >= 0 (strict gain > 0, exact tie == 0).",
        "scope": "Development admission telemetry over rounds 775-2837 (2,063 records); not an independent first-proposal or deployment comparison.",
        "source_binding": source_meta["source_binding"],
        "matplotlib_version": matplotlib.__version__,
        "reproduce": "python3 figures/scripts/make_admission_margins_figure.py",
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: provenance[k] for k in [
        "rows",
        "round_window",
        "accepted_strict_gain_rounds",
        "accepted_exact_tie_rounds",
        "rejected_in_minus_1e_4_to_0",
        "rejected_in_minus_1e_3_to_minus_1e_4",
        "rejected_below_minus_1e_3",
    ]}))


if __name__ == "__main__":
    main()
