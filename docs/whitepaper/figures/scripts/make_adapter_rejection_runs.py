#!/usr/bin/env python3
"""Plot every observed rejection run in a pinned, committed admission record.

Admissions delimit runs; missing round identifiers split them. Capture/gap
boundaries remain explicit. No journal extraction, dataset/model access or run.
"""
from pathlib import Path
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
SOURCE_COMMIT = "6f4d1636d0010c4181f1a4be05c693f2ea5b90c3"
INPUT = "stopped_run_admissions.csv"
INPUT_SHA256 = "197c519554675d398cb77c81e652788b51462ca372a2ea6c7b41811958b4238a"
sha = lambda b: hashlib.sha256(b).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")

def main():
    raw = (DATA / INPUT).read_bytes()
    assert sha(raw) == INPUT_SHA256
    receipt_path = DATA / "stopped_run_record.source.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["csv_sha256"][INPUT] == INPUT_SHA256
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    ids = [int(row["round"]) for row in rows]
    assert ids == [n for n in range(775, 5050) if n != 4331]
    assert len(rows) == 4274 and receipt["missing_admission_rounds"] == [4331]
    allowed = {"restore_best_model_and_optimizer", "accept_improvement", "accept_tie"}
    for row in rows:
        assert row["action"] in allowed
        assert (row["accepted"] == "1") == (row["action"] != "restore_best_model_and_optimizer")
    admitted = {int(row["round"]): row for row in rows if row["accepted"] == "1"}
    assert len(admitted) == 8
    groups = []
    current = []
    previous = None
    for index, row in enumerate(rows, 1):
        n = int(row["round"])
        if previous is not None and n != int(previous["round"]) + 1 and current:
            groups.append(current)
            current = []
        if row["accepted"] == "1":
            if current:
                groups.append(current)
                current = []
        else:
            current.append((index, row))
        previous = row
    if current:
        groups.append(current)
    output = []
    for index, group in enumerate(groups, 1):
        first, last = int(group[0][1]["round"]), int(group[-1][1]["round"])
        assert len(group) == last - first + 1
        left = "admission" if first - 1 in admitted else "capture_start" if first == 775 else "missing_decision"
        right = "admission" if last + 1 in admitted else "capture_end" if last == 5049 else "missing_decision"
        members = [dict(csv_row=i, source_line=int(row["source_line"]), round=int(row["round"]), source_record_sha256=row["source_record_sha256"]) for i, row in group]
        output.append(dict(run_index=index, first_round=first, last_round=last, recorded_rejections=len(group), left_boundary=left, right_boundary=right, left_admitted_round=first-1 if left == "admission" else "", right_admitted_round=last+1 if right == "admission" else "", between_admissions=int(left == right == "admission"), first_source_line=members[0]["source_line"], last_source_line=members[-1]["source_line"], members_sha256=sha(json.dumps(members, sort_keys=True, separators=(",", ":")).encode())))
    assert len(output) == 10 and sum(row["recorded_rejections"] for row in output) == 4266
    assert sum(row["between_admissions"] for row in output) == 7
    csv_path = DATA / "adapter_rejection_runs.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    source_path = DATA / "adapter_rejection_runs.source.json"
    dump(source_path, dict(schema="whitepaper.adapter-rejection-runs-source/v1", source_commit=SOURCE_COMMIT, input_csv=INPUT, input_csv_sha256=INPUT_SHA256, input_receipt_sha256=sha(receipt_path.read_bytes()), source_records=4274, rejected_records=4266, admitted_records=8, round_window=[775,5049], missing_decision_rounds=[4331], derived_runs=10, between_admissions=7, capture_or_gap_boundary_runs=3, rejected_records_omitted=0, csv_sha256=sha(csv_path.read_bytes()), definition="Maximal consecutive round identifiers with logged restore_best_model_and_optimizer. All scored records delimit runs; all eight admissions serve as boundaries. Missing4331splits the observed run and is never filled. Capture/gap boundaries are labelled, not treated as observed admissions.", selection="Every rejected decision belongs to exactly one plotted run, in source order; no selection by utility, duration or gain. Admitted decisions are retained as boundary events rather than plotted rejections.", scope="Counts of recorded adaptive adapter decisions, not independent trials, elapsed training time, neural-writer/ranking-map learning or program-search progress. Exact installed source/adapter/pool/input-closure/normalization binding remains unqualified.", imputed_values=False, new_observations=0, new_scientific_runs=0))
    plt.rcParams.update({"font.family":"Liberation Serif", "font.size":9.0, "axes.labelsize":9.5, "xtick.labelsize":9.0, "ytick.labelsize":9.0, "legend.fontsize":9.0, "pdf.fonttype":42, "ps.fonttype":42})
    fig, ax = plt.subplots(figsize=(7.0, 3.25))
    fig.subplots_adjust(left=.18, right=.97, bottom=.18, top=.82)
    for complete, label, color in [(1,"Between recorded admissions (7)","#0072b2"),(0,"Capture or missing-decision boundary (3)","#b5651d")]:
        chosen = [row for row in output if row["between_admissions"] == complete]
        ax.scatter([row["recorded_rejections"] for row in chosen], [10-row["run_index"] for row in chosen], s=26, facecolors=color if complete else "none", edgecolors=color, linewidths=1.0, label=label, zorder=3)
    for row in output:
        ax.annotate(f"{row['recorded_rejections']:,}", (row["recorded_rejections"],10-row["run_index"]), xytext=(6,0), textcoords="offset points", va="center", fontsize=9.0)
    ax.set_xscale("log")
    ax.set_xlim(1, 5000)
    ax.xaxis.set_major_locator(FixedLocator([1,10,100,1000]))
    ax.xaxis.set_major_formatter(FixedFormatter(["1","10","100","1,000"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([10-row["run_index"] for row in output], [f"{row['first_round']}–{row['last_round']}" for row in output])
    ax.set_ylim(-.7,9.7)
    ax.set_xlabel("consecutive recorded rejections (log scale)")
    ax.set_ylabel("Logged round interval")
    ax.tick_params(axis="y", length=0, pad=6)
    ax.grid(axis="x", color=".88", linewidth=.5, zorder=0)
    for spine in ["top","right","left"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(.5,1.07), ncol=2, frameon=False, handletextpad=.5, columnspacing=1.0)
    pdf = OUT / "fig_adapter_rejection_runs.pdf"
    fig.savefig(pdf, metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)
    font = Path(font_manager.findfont("Liberation Serif"))
    dump(DATA / "adapter_rejection_runs.provenance.json", dict(schema="whitepaper.adapter-rejection-runs-figure/v1", input_csv_sha256=INPUT_SHA256, aggregate_csv_sha256=sha(csv_path.read_bytes()), source_receipt_sha256=sha(source_path.read_bytes()), generator_sha256=sha(Path(__file__).read_bytes()), figure_sha256=sha(pdf.read_bytes()), source_records=4274, rejection_memberships=4266, points=10, filled_points=7, open_points=3, x_unit="recorded rejected decisions", x_scale="logarithmic", y_unit="logged round interval", missing_decisions_imputed=False, smoothing=False, jitter=False, interpolation=False, estimated_uncertainty=False, figure_inches=[7.0,3.25], minimum_text_size_pt=9.0, font_family="Liberation Serif", font_file_sha256=sha(font.read_bytes()), matplotlib_version=matplotlib.__version__, fixed_canvas_no_tight_crop=True, reproduce="python3 figures/scripts/make_adapter_rejection_runs.py"))
    print(json.dumps(dict(status="PASS", source_records=4274, rejections=4266, admitted_boundaries=8, runs=10, bounded_runs=7, capture_or_gap_runs=3)))

if __name__ == "__main__":
    main()
