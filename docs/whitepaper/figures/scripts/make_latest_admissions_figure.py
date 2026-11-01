#!/usr/bin/env python3
"""Plot the latest retained admission prefix; optionally extract a captured log."""

import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FIGURES = Path(__file__).resolve().parent.parent
DATA = FIGURES / "data/latest_admissions.csv"
SOURCE = DATA.with_suffix(".source.json")
PROVENANCE = DATA.with_suffix(".provenance.json")
OUTPUT = FIGURES / "fig_fw_latest_admissions.pdf"
FIELDS = [
    "round",
    "candidate_val",
    "previous_best_val",
    "best_val",
    "accepted",
    "decision",
    "best_round",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(log, snapshot):
    metadata = json.loads(snapshot.read_text())
    assert sha(log) == metadata["prefix_sha256"]
    records = [
        json.loads(line, parse_float=Decimal) for line in log.read_text().splitlines()
    ]
    assert len(records) == metadata["source_rows"]
    identities = {str(record["evaluation_identity"]) for record in records}
    assert len(identities) == 1, "Admission evaluation identity changed"
    rows = []
    for record in records:
        assert isinstance(record["accepted"], bool)
        rows.append(
            dict(
                zip(
                    FIELDS,
                    [
                        record["round"],
                        str(record["candidate_utility"]),
                        str(record["previous_best_utility"]),
                        str(record["best_utility"]),
                        int(record["accepted"]),
                        record["action"],
                        record["best_round"],
                    ],
                )
            )
        )
    with DATA.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    # Explicit allowlist: no source location, checkpoint name or machine identity.
    capture = {
        key: metadata[key]
        for key in [
            "at",
            "observed_bytes",
            "prefix_bytes",
            "prefix_sha256",
            "trailing_incomplete_bytes_omitted",
            "source_rows",
            "first_round",
            "last_round",
        ]
    }
    source = dict(
        schema="whitepaper.latest-admissions-source/v1",
        capture=capture,
        stable_prefix_reread_verified=True,
        evaluation_identity_sha256=sorted(
            hashlib.sha256(s.encode()).hexdigest() for s in identities
        ),
        csv_sha256=sha(DATA),
        selection="Every complete admission row in stable prefix, in log order; no outcome selection or deduplication",
        field_mapping="candidate/previous best/post-decision best utility, acceptance/action and best-round identity as logged",
        source_binding="Aggregate field interpretation only; exact installed source, adapter and evaluation population remain unjoined",
        scope="Adaptive development admissions; independent first-proposal and deployment evaluation remain required",
    )
    SOURCE.write_text(json.dumps(source, indent=2) + "\n")


def plot():
    source = json.loads(SOURCE.read_text())
    assert sha(DATA) == source["csv_sha256"]
    with DATA.open() as stream:
        rows = list(csv.DictReader(stream))
    rounds = [int(row["round"]) for row in rows]
    assert rounds == list(range(rounds[0], rounds[-1] + 1))
    assert len(rows) == source["capture"]["source_rows"]
    assert [rounds[0], rounds[-1]] == [
        source["capture"]["first_round"],
        source["capture"]["last_round"],
    ]
    candidates = [Decimal(row["candidate_val"]) for row in rows]
    previous = [Decimal(row["previous_best_val"]) for row in rows]
    bests = [Decimal(row["best_val"]) for row in rows]
    accepted, gains, ties, rejected = [], [], [], []
    for i, row in enumerate(rows):
        assert all(
            value.is_finite() for value in [candidates[i], previous[i], bests[i]]
        )
        assert row["accepted"] in ["0", "1"]
        take = row["accepted"] == "1"
        assert take == (candidates[i] >= previous[i])
        assert bests[i] == (candidates[i] if take else previous[i])
        if i:
            assert previous[i] == bests[i - 1]
        assert int(row["best_round"]) <= rounds[i]
        if take:
            assert int(row["best_round"]) == rounds[i]
            accepted.append(i)
            strict = candidates[i] > previous[i]
            assert row["decision"] == ("accept_improvement" if strict else "accept_tie")
            (gains if strict else ties).append(i)
        else:
            assert row["decision"] == "restore_best_model_and_optimizer"
            if i:
                assert row["best_round"] == rows[i - 1]["best_round"]
            rejected.append(i)
    # The older displayed window is an exact prefix, not an independent sample.
    with (FIGURES / "data/live_admissions.csv").open() as stream:
        reference = list(csv.DictReader(stream))
    assert len(reference) <= len(rows)
    for old, new in zip(reference, rows):
        assert old["round"] == new["round"] and old["accepted"] == new["accepted"]
        assert old["best_round"] == new["best_round"]
        assert all(
            Decimal(old[k]) == Decimal(new[k]) for k in ["candidate_val", "best_val"]
        )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 2.55))
    ax.scatter(
        [rounds[i] for i in rejected],
        [float(candidates[i]) for i in rejected],
        s=4,
        color="#7c858e",
        alpha=0.45,
        linewidths=0,
        label=f"Rejected candidates ({len(rejected):,})",
    )
    ax.step(
        rounds,
        [float(value) for value in bests],
        where="post",
        color="#0b5394",
        lw=1.25,
        label="Retained best (post-decision)",
    )
    ax.scatter(
        [rounds[i] for i in gains],
        [float(candidates[i]) for i in gains],
        s=38,
        marker="*",
        color="#c92a2a",
        zorder=5,
        label=f"Accepted strict gains ({len(gains)})",
    )
    ax.scatter(
        [rounds[i] for i in ties],
        [float(candidates[i]) for i in ties],
        s=27,
        marker="D",
        facecolors="none",
        edgecolors="#d9480f",
        lw=1,
        zorder=6,
        label=f"Accepted exact ties ({len(ties)})",
    )
    if gains:
        i = gains[-1]
        ax.annotate(
            f"Round {rounds[i]}: strict gain\n+{candidates[i] - previous[i]:.10f}",
            xy=(rounds[i], float(bests[i])),
            xytext=(0.64, 0.52),
            textcoords="axes fraction",
            fontsize=7,
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#c92a2a", lw=0.7),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
        )
    if ties:
        i = ties[-1]
        ax.annotate(
            f"Round {rounds[i]}: exact tie",
            xy=(rounds[i], float(bests[i])),
            xytext=(0.84, 0.25),
            textcoords="axes fraction",
            fontsize=7,
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#d9480f", lw=0.7),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
        )
    ax.set(
        xlabel="Round",
        ylabel="Development pool utility",
        xlim=(rounds[0] - 12, rounds[-1] + 12),
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.grid(axis="y", alpha=0.22, lw=0.4)
    ax.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="none",
        ncol=2,
    )
    fig.tight_layout(pad=0.5)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    provenance = dict(
        schema="whitepaper.latest-admissions-figure/v1",
        source_receipt_sha256=sha(SOURCE),
        csv_sha256=sha(DATA),
        generator_sha256=sha(Path(__file__)),
        figure_sha256=sha(OUTPUT),
        round_window=[rounds[0], rounds[-1]],
        rows=len(rows),
        older_displayed_prefix_rows=len(reference),
        additional_logged_rows=len(rows) - len(reference),
        accepted_rounds=[rounds[i] for i in accepted],
        strict_gain_rounds=[rounds[i] for i in gains],
        accepted_tie_rounds=[rounds[i] for i in ties],
        rejected_rows=len(rejected),
        final_recorded_best=str(bests[-1]),
        final_recorded_best_round=int(rows[-1]["best_round"]),
        classification="Exact Decimal comparison of logged candidate with previous best; no epsilon or rounded-value tie classification",
        default_acceptance="candidate >= previous best, without a margin; every captured flag agrees",
        transformations="All rows retained; no smoothing, clipping, downsampling or added cumulative maximum",
        source_binding=source["source_binding"],
        scope=source["scope"],
        matplotlib_version=matplotlib.__version__,
        reproduce="python3 figures/scripts/make_latest_admissions_figure.py",
    )
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: provenance[k]
                for k in [
                    "rows",
                    "round_window",
                    "accepted_rounds",
                    "strict_gain_rounds",
                    "accepted_tie_rounds",
                    "final_recorded_best",
                ]
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if bool(args.log) != bool(args.snapshot):
        parser.error("Supply both captured log and snapshot receipt")
    if args.log:
        extract(args.log, args.snapshot)
    plot()


if __name__ == "__main__":
    main()
