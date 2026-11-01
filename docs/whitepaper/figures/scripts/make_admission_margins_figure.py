#!/usr/bin/env python3
"""Plot trailing candidate-margin quantiles and cumulative near-tie rejections."""

import csv
import hashlib
import io
import json
from decimal import Decimal, getcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

getcontext().prec = 50

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
    assert csv_digest == source_meta["csv_sha256"], (
        "CSV sha256 mismatch with source receipt"
    )

    rows = list(csv.DictReader(io.StringIO(raw_csv.decode("utf-8"))))
    rounds = [int(r["round"]) for r in rows]
    assert rounds == list(range(rounds[0], rounds[-1] + 1))
    assert len(rows) == source_meta["capture"]["source_rows"]
    assert rounds[0] == source_meta["capture"]["first_round"]
    assert rounds[-1] == source_meta["capture"]["last_round"]

    candidates = [Decimal(r["candidate_val"]) for r in rows]
    previous = [Decimal(r["previous_best_val"]) for r in rows]
    bests = [Decimal(r["best_val"]) for r in rows]
    accepted_flags = [int(r["accepted"]) for r in rows]
    decisions = [r["decision"] for r in rows]

    margins = [c - p for c, p in zip(candidates, previous)]

    gains_idx, ties_idx, rej_idx = [], [], []
    for i, (m, acc, dec) in enumerate(zip(margins, accepted_flags, decisions)):
        assert all(
            value.is_finite() for value in [candidates[i], previous[i], bests[i]]
        )
        assert bests[i] == (candidates[i] if acc else previous[i])
        if i:
            assert previous[i] == bests[i - 1]
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

    assert len(gains_idx) + len(ties_idx) + len(rej_idx) == len(rows)
    assert rej_idx, "This diagnostic requires rejected candidates"

    gains_ge_1e3 = [i for i in gains_idx if margins[i] >= Decimal("0.001")]
    gains_sub_1e3 = [
        i for i in gains_idx if Decimal("0") < margins[i] < Decimal("0.001")
    ]
    rej_band_1e4 = [i for i in rej_idx if margins[i] >= Decimal("-0.0001")]
    rej_band_1e3 = [
        i for i in rej_idx if Decimal("-0.001") <= margins[i] < Decimal("-0.0001")
    ]
    rej_below_1e3 = [i for i in rej_idx if margins[i] < Decimal("-0.001")]

    assert len(gains_ge_1e3) + len(gains_sub_1e3) == len(gains_idx)
    assert len(rej_band_1e4) + len(rej_band_1e3) + len(rej_below_1e3) == len(rej_idx)

    rej_margins_sorted = sorted(margins[i] for i in rej_idx)
    median_rej = (
        rej_margins_sorted[(len(rej_idx) - 1) // 2]
        + rej_margins_sorted[len(rej_idx) // 2]
    ) / 2
    min_rej_i = min(rej_idx, key=lambda i: margins[i])
    max_rej_i = max(rej_idx, key=lambda i: margins[i])

    # Complete trailing windows only; all candidate margins are included.
    def quantile(xs, p):
        at = p * (len(xs) - 1)
        lo = int(at)
        hi = min(lo + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (at - lo)

    cumulative4 = cumulative3 = 0
    derived = []
    for i, (round_id, margin, accepted) in enumerate(
        zip(rounds, margins, accepted_flags)
    ):
        if not accepted:
            cumulative4 += int(Decimal("-.0001") <= margin < 0)
            cumulative3 += int(Decimal("-.001") <= margin < Decimal("-.0001"))
        qs = (
            [
                str(quantile(sorted(margins[i - 49 : i + 1]), q))
                for q in [Decimal(".25"), Decimal(".5"), Decimal(".75"), Decimal(".9")]
            ]
            if i >= 49
            else [""] * 4
        )
        derived.append(
            dict(
                round=round_id,
                q25=qs[0],
                q50=qs[1],
                q75=qs[2],
                q90=qs[3],
                near_1e4=cumulative4,
                near_1e3_to_1e4=cumulative3,
            )
        )
    rolling_path = DATA.with_name("rolling_margin_quantiles.csv")
    with rolling_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(derived[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(derived)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.8, 2.35))
    rr = rounds[49:]
    dd = derived[49:]
    ax0.fill_between(
        rr,
        [float(r["q25"]) for r in dd],
        [float(r["q75"]) for r in dd],
        color="#748ffc",
        alpha=0.28,
        label="50-round IQR",
    )
    ax0.plot(rr, [float(r["q50"]) for r in dd], color="#1c7ed6", lw=1, label="median")
    ax0.plot(
        rr,
        [float(r["q90"]) for r in dd],
        color="#e8590c",
        lw=0.8,
        ls="-.",
        label="90th percentile",
    )
    ax0.axhline(0, color="#495057", lw=0.7, ls="--")
    for i in gains_idx:
        ax0.axvline(rounds[i], color="#c92a2a", lw=0.5, alpha=0.45, ls=":")
    ax0.set(
        title="Trailing candidate-margin quantiles",
        xlabel="round",
        ylabel="candidate minus prior-best utility",
    )
    ax0.legend(fontsize=6, loc="lower left")
    ax0.grid(axis="y", alpha=0.2)
    ax1.step(
        rounds,
        [r["near_1e4"] for r in derived],
        where="post",
        lw=1,
        color="#c92a2a",
        label=r"$[-10^{-4},0)$",
    )
    ax1.step(
        rounds,
        [r["near_1e3_to_1e4"] for r in derived],
        where="post",
        lw=1,
        color="#495057",
        ls="--",
        label=r"$[-10^{-3},-10^{-4})$",
    )
    ax1.set(
        title="Cumulative near-tie rejections", xlabel="round", ylabel="rejected rounds"
    )
    ax1.legend(fontsize=6, loc="upper left")
    ax1.grid(axis="y", alpha=0.2)
    for ax in (ax0, ax1):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    fig.tight_layout(pad=0.6, w_pad=1)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    provenance = {
        "schema": "whitepaper.rolling-margin-quantiles-figure/v2",
        "source_csv": "figures/data/latest_admissions.csv",
        "source_csv_sha256": csv_digest,
        "source_receipt_sha256": hashlib.sha256(raw_source).hexdigest(),
        "generator_sha256": sha(Path(__file__)),
        "rolling_csv_sha256": sha(rolling_path),
        "quantile_window": 50,
        "quantiles": "Linear interpolation at p*(n-1), all50 candidate margins, Decimal precision50; no partial windows",
        "near_tie_counts": "Cumulative rejected rows in two exclusive bands; no acceptance trials inferred",
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
        "scope": f"Development admission telemetry over rounds {rounds[0]}-{rounds[-1]} ({len(rows):,} records); not an independent first-proposal or deployment comparison.",
        "source_binding": source_meta["source_binding"],
        "matplotlib_version": matplotlib.__version__,
        "reproduce": "python3 figures/scripts/make_admission_margins_figure.py",
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: provenance[k]
                for k in [
                    "rows",
                    "round_window",
                    "accepted_strict_gain_rounds",
                    "accepted_exact_tie_rounds",
                    "rejected_in_minus_1e_4_to_0",
                    "rejected_in_minus_1e_3_to_minus_1e_4",
                    "rejected_below_minus_1e_3",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
