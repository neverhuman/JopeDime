#!/usr/bin/env python3
"""Account for every committed round duration by the recorded adapter decision.

No journal extraction, model loading, dataset access or scientific run. The two input CSVs
are pinned below and carry their original capture receipts. Legacy missing
completion flags are retained; a clock without a scored decision stays unscored.
"""
from pathlib import Path
from decimal import Decimal, localcontext
import csv
import hashlib
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

OUT = Path(__file__).resolve().parents[1]
DATA = OUT / "data"
SOURCE_COMMIT = "167a527e2889b6b4e4be69736e03a0507bd43f4f"
INPUTS = {
    "full_round_wall_time.csv": "46daa1916799f347f1af01311c07a6044d9bc2c9e9d6f758ed6e7e15aafd6488",
    "full_retained_trajectory.csv": "7bb0dd88e1a6ac1a57075d1b72c87db26d7bbe95a953bf90b8ccf526297fdd58",
}
CLASSES = ["rollback", "strict_gain", "exact_tie", "unscored"]
ACTIONS = {"restore_best_model_and_optimizer": "rollback", "accept_improvement": "strict_gain", "accept_tie": "exact_tie"}
sha = lambda b: hashlib.sha256(b).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")

def main():
    tables = {}
    receipts = {}
    for name, expected in INPUTS.items():
        raw = (DATA / name).read_bytes()
        assert sha(raw) == expected, name
        tables[name] = list(csv.DictReader(io.StringIO(raw.decode())))
        receipt_path = DATA / name.replace(".csv", ".source.json")
        receipt = json.loads(receipt_path.read_text())
        assert receipt["csv_sha256"] == expected
        receipts[receipt_path.name] = sha(receipt_path.read_bytes())
    clocks = tables["full_round_wall_time.csv"]
    decisions = tables["full_retained_trajectory.csv"]
    assert len(clocks) == 4064 and len(decisions) == 4063
    by_round = {int(r["round"]): r for r in decisions}
    assert len(by_round) == 4063
    assert [int(r["round"]) for r in clocks] == list(range(775, 4839))
    assert set(range(775, 4839)) - set(by_round) == {4331}
    groups = {key: [] for key in CLASSES}
    for row in clocks:
        duration = Decimal(row["wall_seconds"])
        assert duration.is_finite() and duration >= 0
        decision = by_round.get(int(row["round"]))
        key = ACTIONS[decision["action"]] if decision else "unscored"
        if decision:
            assert (decision["accepted"] == "1") == (key != "rollback")
            prior = Decimal(decision["previous_best_utility"])
            retained = Decimal(decision["best_utility"])
            assert (retained > prior) if key == "strict_gain" else (retained == prior)
        identity = {"round": int(row["round"]), "clock_record_sha256": row["source_record_sha256"], "decision_record_sha256": decision["source_record_sha256"] if decision else None}
        groups[key].append((duration, identity))
    assert [len(groups[k]) for k in CLASSES] == [4055, 4, 4, 1]
    aggregates = []
    with localcontext() as context:
        context.prec = 50
        total = sum((Decimal(r["wall_seconds"]) for r in clocks), Decimal(0))
        for key in CLASSES:
            members = [identity for _, identity in groups[key]]
            seconds = sum((duration for duration, _ in groups[key]), Decimal(0))
            aggregates.append(dict(decision_class=key, rounds=len(members), wall_seconds=format(seconds, "f"), wall_hours=format(seconds / 3600, ".12f"), time_share_percent=format(100 * seconds / total, ".9f"), first_round=members[0]["round"], last_round=members[-1]["round"], member_identity_sha256=sha(json.dumps(members, sort_keys=True, separators=(",", ":")).encode())))
    csv_path = DATA / "adapter_outcome_clock.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregates)
    source_path = DATA / "adapter_outcome_clock.source.json"
    source = dict(schema="whitepaper.adapter-outcome-clock-source/v1", source_commit=SOURCE_COMMIT, input_csv_sha256=INPUTS, input_receipt_sha256=receipts, round_window=[775, 4838], clock_records=4064, scored_decisions=4063, unscored_rounds=[4331], legacy_completion_flag_unrecorded=3556, explicit_completion_flag_records=508, all_rounds_and_durations_retained=True, outcome_counts={k: len(groups[k]) for k in CLASSES}, total_recorded_seconds=format(total, "f"), aggregate_rows=4, csv_sha256=sha(csv_path.read_bytes()), derivation="Join committed clocks to decisions by round; exact Decimal sums of every recorded duration. Hours divide by3600; hours CSV rounded to12decimal places, shares to9, arithmetic precision50. No duration or outcome filtering; the one unscored round stays separate.", scope="Summed recorded in-round wall time, including work and waits; excludes inter-round gaps. Legacy records need not mark completion. Adapter rollback does not mean program search made no progress. Not CPU/GPU time, financial cost, independent first-proposal quality or learning.", source_binding="Logged aggregate field interpretation only. Exact installed source, adapter, candidate pools, normalization and evaluation-population closure remain unqualified.", new_observations=0, new_scientific_runs=0, imputed_values=False)
    write_json(source_path, source)
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9.0, "axes.labelsize": 9.0, "xtick.labelsize": 8.5, "ytick.labelsize": 9.0, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.375, 2.35))
    fig.subplots_adjust(left=.40, right=.97, bottom=.25, top=.95)
    labels = ["Rollback\n(n = 4,055)", "Strict gain\n(n = 4)", "Exact tie\n(n = 4)", "No scored decision\n(n = 1)"]
    colors = ["#62666b", "#0072b2", "#d18f00", "#8a4d76"]
    markers = ["o", "^", "s", "D"]
    for i, row in enumerate(aggregates):
        hours = float(Decimal(row["wall_seconds"]) / 3600)
        y = 3 - i
        ax.scatter([hours], [y], color=colors[i], marker=markers[i], s=28, zorder=3)
        ax.annotate(f"{hours:.3f} h" if hours < 1 else f"{hours:.2f} h", (hours, y), xytext=(-7 if i == 0 else 7, 0), textcoords="offset points", ha="right" if i == 0 else "left", va="center", fontsize=9.0)
    ax.set_xscale("log")
    ax.set_xlim(.01, 200)
    ax.xaxis.set_major_locator(FixedLocator([.01, .1, 1, 10, 100]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.01", "0.1", "1", "10", "100"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-.5, 3.5)
    ax.set_yticks([3, 2, 1, 0], labels)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.grid(axis="x", color=".88", linewidth=.5, zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.text(.5, .055, "sum of recorded round time (hours; log scale)", ha="center", fontsize=9.0)
    pdf = OUT / "fig_adapter_outcome_clock.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    font = Path(font_manager.findfont("Liberation Serif"))
    write_json(DATA / "adapter_outcome_clock.provenance.json", dict(schema="whitepaper.adapter-outcome-clock-figure/v1", input_csv_sha256=INPUTS, aggregate_csv_sha256=sha(csv_path.read_bytes()), source_receipt_sha256=sha(source_path.read_bytes()), generator_sha256=sha(Path(__file__).read_bytes()), figure_sha256=sha(pdf.read_bytes()), source_clock_records=4064, plotted_aggregate_points=4, memberships_total=4064, round_window=[775, 4838], x_unit="sum of recorded in-round wall hours", x_scale="logarithmic", unknown_outcome_kept_separate=True, filtering=False, smoothing=False, interpolation=False, jitter=False, imputation=False, estimated_uncertainty=False, minimum_text_size_pt=8.5, figure_inches=[3.375, 2.35], font_family="Liberation Serif", font_file_sha256=sha(font.read_bytes()), matplotlib_version=matplotlib.__version__, fixed_canvas_no_tight_crop=True, reproduce="python3 figures/scripts/make_adapter_outcome_clock.py"))
    print(json.dumps(dict(status="PASS", records=4064, points=4, total_seconds=str(total), rollback_share_percent=aggregates[0]["time_share_percent"])))

if __name__ == "__main__":
    main()
