#!/usr/bin/env python3
"""Align two complete committed journal captures by logged round identifier."""
from pathlib import Path
import collections
import csv
import hashlib
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.patches import Rectangle
import numpy as np

OUT = Path(__file__).resolve().parents[1]
DATA = OUT / "data"
SOURCE_COMMIT = "84a5338bd73469681d3e8ba8a7c2316046c87152"
PINS = {
    "stopped_run_admissions.csv": "197c519554675d398cb77c81e652788b51462ca372a2ea6c7b41811958b4238a",
    "stopped_run_promotions.csv": "913841307f33c827b60174288c8b09404c92be1f186ff74bc83f9be6103813cc",
    "stopped_run_record.source.json": "26337f004c1ed4c1825a5907030e75b0f3944f2001f0d03e747896801f296873",
}
CLASSES = ["rejected_update", "strict_gain", "exact_tie", "decision_unrecorded"]
ACTIONS = {"restore_best_model_and_optimizer": "rejected_update", "accept_improvement": "strict_gain", "accept_tie": "exact_tie"}
sha = lambda b: hashlib.sha256(b).hexdigest()
dump = lambda p, v: p.write_text(json.dumps(v, indent=2) + "\n")

def main():
    raw = {n: (DATA / n).read_bytes() for n in PINS}
    assert all(sha(raw[n]) == h for n, h in PINS.items())
    receipt = json.loads(raw["stopped_run_record.source.json"])
    assert receipt["csv_sha256"] == {n: h for n, h in PINS.items() if n.endswith(".csv")}
    admissions = list(csv.DictReader(io.StringIO(raw["stopped_run_admissions.csv"].decode())))
    programs = list(csv.DictReader(io.StringIO(raw["stopped_run_promotions.csv"].decode())))
    assert len(admissions) == 4274 and len(programs) == 3529
    by_round = {int(r["round"]): r for r in admissions}
    assert len(by_round) == 4274 and set(range(775, 5050)) - set(by_round) == {4331}
    assert receipt["missing_admission_rounds"] == [4331]
    events = collections.defaultdict(list)
    for index, row in enumerate(programs, 1):
        n = int(row["round"])
        assert 775 <= n <= 5049
        events[n].append(dict(csv_row=index, source_line=int(row["source_line"]), source_record_sha256=row["source_record_sha256"]))
    assert len({r["source_record_sha256"] for r in programs}) == 3529
    groups = collections.defaultdict(list)
    for n in range(775, 5050):
        row = by_round.get(n)
        if row is None:
            kind = "decision_unrecorded"
            assert n == 4331
        else:
            kind = ACTIONS[row["action"]]
            assert (row["accepted"] == "1") == (kind != "rejected_update")
        member = dict(round=n, admission_source_line=int(row["source_line"]) if row else None, admission_source_record_sha256=row["source_record_sha256"] if row else None, program_records=events[n])
        groups[(kind, bool(events[n]))].append(member)
    output = []
    for kind in CLASSES:
        denominator = sum(len(groups[(kind, present)]) for present in [True, False])
        for present in [True, False]:
            members = groups[(kind, present)]
            output.append(dict(adapter_class=kind, program_record_present=int(present), logged_rounds=len(members), class_rounds=denominator, program_records=sum(len(m["program_records"]) for m in members), members_sha256=sha(json.dumps(members, sort_keys=True, separators=(",", ":")).encode())))
    assert sum(r["logged_rounds"] for r in output) == 4275
    assert sum(r["program_records"] for r in output) == 3529
    assert [r["logged_rounds"] for r in output] == [2142, 2124, 2, 2, 3, 1, 1, 0]
    aggregate = DATA / "program_event_concurrence.csv"
    with aggregate.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(output)
    source = DATA / "program_event_concurrence.source.json"
    dump(source, dict(schema="whitepaper.program-event-concurrence-source/v1", source_commit=SOURCE_COMMIT, input_sha256=PINS, scored_decisions=4274, unrecorded_decision_rounds=[4331], logged_round_span=[775,5049], round_identifiers_counted=4275, program_records=3529, program_record_rounds=2148, derived_cells=8, csv_sha256=sha(aggregate.read_bytes()), selection="Every round identifier775..5049 in the committed capture. No admission decision is inferred at4331. Every3529 program record appears once, attached to its logged round; all4274 scored admission records appear once.", zero_definition="No program record at a round in this complete bounded journal capture; not zero proposals, zero work or a failed search.", scope="Descriptive coexistence of recorded events with different admission rules, not proposal success, independence, causal effect, elapsed time or writer/map learning.", new_observations=0, imputed_decisions=False))
    plt.rcParams.update({"font.family":"Liberation Serif", "font.size":9.0, "axes.labelsize":9.5, "xtick.labelsize":9.0, "ytick.labelsize":9.0, "pdf.fonttype":42, "ps.fonttype":42})
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    fig.subplots_adjust(left=.31, right=.85, top=.96, bottom=.19)
    counts = np.array([r["logged_rounds"] for r in output]).reshape(4,2)
    image = ax.imshow(np.ma.masked_equal(counts,0), cmap="Blues", norm=colors.LogNorm(vmin=1,vmax=2142), aspect="auto", interpolation="nearest")
    for i, row in enumerate(output):
        y, x = divmod(i, 2)
        if row["logged_rounds"] == 0:
            ax.add_patch(Rectangle((x-.5,y-.5),1,1,facecolor=".95",edgecolor=".7",hatch="///",linewidth=.5))
        text = f"{row['logged_rounds']:,} {'round' if row['logged_rounds'] == 1 else 'rounds'}\n{row['program_records']:,} {'record' if row['program_records'] == 1 else 'records'}"
        ax.text(x,y,text,ha="center",va="center",fontsize=9.0,color="white" if row["logged_rounds"]>=100 else "black",linespacing=1.2,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.8) if row["logged_rounds"] == 0 else None)
    ax.set_xticks([0,1],["Program record present","No program record"])
    ax.set_yticks(range(4),["Rejected update (n = 4,266)","Strict gain (n = 4)","Exact tie (n = 4)","Decision unrecorded (n = 1)"])
    ax.tick_params(length=0,pad=5)
    for spine in ax.spines.values():spine.set_visible(False)
    cax = fig.add_axes([.89,.19,.014,.77])
    bar = fig.colorbar(image,cax=cax,ticks=[1,10,100,1000])
    bar.ax.set_yticklabels(["1","10","100","1,000"]);bar.ax.minorticks_off();bar.set_label("Round count (log color)",labelpad=8)
    pdf = OUT / "fig_program_event_concurrence.pdf"
    fig.savefig(pdf,metadata={"CreationDate":None,"ModDate":None});plt.close(fig)
    font = Path(font_manager.findfont("Liberation Serif"))
    dump(DATA / "program_event_concurrence.provenance.json",dict(schema="whitepaper.program-event-concurrence-figure/v1", input_sha256=PINS, csv_sha256=sha(aggregate.read_bytes()), source_receipt_sha256=sha(source.read_bytes()), generator_sha256=sha(Path(__file__).read_bytes()), figure_sha256=sha(pdf.read_bytes()), round_count_total=4275, scored_decisions=4274, missing_decisions=1, program_records=3529, figure_inches=[7.0,2.7], minimum_text_size_pt=9.0, font_family="Liberation Serif", font_file_sha256=sha(font.read_bytes()), matplotlib_version=matplotlib.__version__, color_scale="logarithmic round count; hatched cell denotes zero", cells=8, smoothing=False, jitter=False, interpolation=False, imputed_decisions=False, reproduce="python3 figures/scripts/make_program_event_concurrence.py"))
    print(json.dumps(dict(status="PASS",rounds=4275,scored_decisions=4274,program_records=3529,cells=8)))

if __name__ == "__main__":
    main()
