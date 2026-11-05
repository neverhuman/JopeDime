#!/usr/bin/env python3
"""Count every logged round and program record in a pinned bounded capture."""
from pathlib import Path
import collections
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
SOURCE_COMMIT = "e6606d964699b46b0d4380e32f1f653196039069"
PINS = {
    "stopped_run_admissions.csv": "197c519554675d398cb77c81e652788b51462ca372a2ea6c7b41811958b4238a",
    "stopped_run_promotions.csv": "913841307f33c827b60174288c8b09404c92be1f186ff74bc83f9be6103813cc",
    "stopped_run_record.source.json": "26337f004c1ed4c1825a5907030e75b0f3944f2001f0d03e747896801f296873",
}
sha = lambda b: hashlib.sha256(b).hexdigest()
dump = lambda p, v: p.write_text(json.dumps(v, indent=2) + "\n")
member_hash = lambda members: sha(json.dumps(members, sort_keys=True, separators=(",", ":")).encode())

def main():
    raw = {n: (DATA / n).read_bytes() for n in PINS}
    assert all(sha(raw[n]) == h for n, h in PINS.items())
    capture = json.loads(raw["stopped_run_record.source.json"])
    assert capture["round_window"] == [775,5049] and capture["missing_admission_rounds"] == [4331]
    assert capture["csv_sha256"] == {n:h for n,h in PINS.items() if n.endswith(".csv")}
    decisions = list(csv.DictReader(io.StringIO(raw["stopped_run_admissions.csv"].decode())))
    programs = list(csv.DictReader(io.StringIO(raw["stopped_run_promotions.csv"].decode())))
    decision_map = {int(r["round"]):r for r in decisions}
    assert len(decisions) == len(decision_map) == 4274
    span = set(range(775,5050))
    assert span - set(decision_map) == {4331}
    events = collections.defaultdict(list)
    for index, row in enumerate(programs,1):
        n = int(row["round"]); assert n in span
        events[n].append(dict(csv_row=index,source_line=int(row["source_line"]),source_record_sha256=row["source_record_sha256"]))
    assert len(programs) == len({r["source_record_sha256"] for r in programs}) == 3529
    assert set(decision_map) | set(events) == span
    maximum = max(map(len,events.values()))
    groups = {k:[] for k in range(maximum+1)}
    for n in sorted(span):
        d = decision_map.get(n)
        groups[len(events[n])].append(dict(round=n,admission_source_line=int(d["source_line"]) if d else None,admission_source_record_sha256=d["source_record_sha256"] if d else None,program_records=events[n]))
    output = []
    for k, members in groups.items():
        tail_ids = sorted(m["round"] for j,g in groups.items() if j>=k for m in g)
        output.append(dict(records_per_round=k,rounds_exactly_k=len(members),program_records_exact=sum(len(m["program_records"]) for m in members),rounds_with_admission_record=sum(m["admission_source_line"] is not None for m in members),rounds_without_admission_record=sum(m["admission_source_line"] is None for m in members),rounds_at_least_k=len(tail_ids),exact_members_sha256=member_hash(members),tail_round_ids_sha256=member_hash(tail_ids)))
    assert maximum == 10 and len(output) == 11
    assert sum(r["rounds_exactly_k"] for r in output) == 4275
    assert sum(r["program_records_exact"] for r in output) == 3529
    assert [r["rounds_exactly_k"] for r in output] == [2127,1282,541,214,61,32,11,5,1,0,1]
    assert sum(r["rounds_without_admission_record"] for r in output) == 1
    aggregate = DATA / "program_record_multiplicity.csv"
    with aggregate.open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(output[0]),lineterminator="\n");writer.writeheader();writer.writerows(output)
    source = DATA / "program_record_multiplicity.source.json"
    dump(source,dict(schema="whitepaper.program-record-multiplicity-source/v1",source_commit=SOURCE_COMMIT,input_sha256=PINS,round_window=[775,5049],round_identifiers=4275,scored_admission_records=4274,program_records=3529,program_record_rounds=2148,zero_program_record_rounds=2127,maximum_records_per_round=maximum,derived_thresholds=len(output),missing_admission_rounds=[4331],missing_admission_round_program_records=len(events[4331]),CSV_sha256=sha(aggregate.read_bytes()),selection="Union of both complete committed journals across every logged round775..5049; all4274 admission records and3529 program records retained. Historical/earlier warm-up records were already excluded by the owning capture receipt. No outcome, duration or count filtering.",transformation="Each round belongs to exactly one exact-count group. Each plotted thresholdk counts the round identifiers in groupsk..10. Threshold counts overlap by definition; no round or program record is imputed.",scope="Adaptive training-side journal counts, not proposal attempts, success rates, independent tasks, elapsed time or neural-writer/ranking-map learning.",new_observations=0,new_scientific_runs=0))
    plt.rcParams.update({"font.family":"Liberation Serif","font.size":9.0,"axes.labelsize":9.5,"xtick.labelsize":9.0,"ytick.labelsize":9.0,"pdf.fonttype":42,"ps.fonttype":42})
    fig, ax = plt.subplots(figsize=(7.0,2.8))
    fig.subplots_adjust(left=.13,right=.98,bottom=.22,top=.92)
    xs = [r["records_per_round"] for r in output]; ys = [r["rounds_at_least_k"] for r in output]
    ax.scatter(xs,ys,s=25,color="#0072b2",zorder=3)
    for x,y in zip(xs,ys):ax.annotate(f"{y:,}",(x,y),xytext=(0,6),textcoords="offset points",ha="center",va="bottom",fontsize=9.0)
    ax.set_yscale("log");ax.set_ylim(.7,8500);ax.set_xlim(-.35,10.35)
    ax.set_xticks(xs,[str(x) for x in xs])
    ax.yaxis.set_major_locator(FixedLocator([1,10,100,1000,5000]));ax.yaxis.set_major_formatter(FixedFormatter(["1","10","100","1,000","5,000"]));ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Minimum program records per logged round (k)")
    ax.set_ylabel("Rounds with at least k records",labelpad=8)
    ax.grid(axis="y",color=".88",linewidth=.5,zorder=0)
    for spine in ["top","right"]:ax.spines[spine].set_visible(False)
    pdf = OUT / "fig_program_record_multiplicity.pdf"
    fig.savefig(pdf,metadata={"CreationDate":None,"ModDate":None});plt.close(fig)
    font = Path(font_manager.findfont("Liberation Serif"))
    dump(DATA / "program_record_multiplicity.provenance.json",dict(schema="whitepaper.program-record-multiplicity-figure/v1",input_sha256=PINS,CSV_sha256=sha(aggregate.read_bytes()),source_receipt_sha256=sha(source.read_bytes()),generator_sha256=sha(Path(__file__).read_bytes()),figure_sha256=sha(pdf.read_bytes()),figure_inches=[7.0,2.8],plotted_thresholds=11,round_identifiers=4275,program_records=3529,x_unit="minimum program records at the logged round",y_unit="round identifiers with at least k records",y_scale="logarithmic",all_integer_thresholds_from_zero_to_observed_maximum=True,minimum_text_size_pt=9.0,font_family="Liberation Serif",font_file_sha256=sha(font.read_bytes()),matplotlib_version=matplotlib.__version__,smoothing=False,jitter=False,interpolation=False,estimated_uncertainty=False,missing_admission_decisions_imputed=False,reproduce="python3 figures/scripts/make_program_record_multiplicity.py"))
    print(json.dumps(dict(status="PASS",rounds=4275,program_records=3529,thresholds=11,maximum=maximum)))

if __name__ == "__main__":
    main()
