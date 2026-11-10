#!/usr/bin/env python3
"""Plot every captured program-journal record in a fixed24-hour window."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
from itertools import accumulate
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MaxNLocator

def sha(b):
    return hashlib.sha256(b).hexdigest()

def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=root/"data/program_24h.csv")
    parser.add_argument("--output",type=Path,default=root/"fig_program_24h.pdf")
    parser.add_argument("--provenance",type=Path,default=root/"data/program_24h.provenance.json")
    args=parser.parse_args();source_path=args.data.with_suffix(".source.json");events_path=args.data.with_suffix(".events.csv")
    source=json.loads(source_path.read_text());raw=args.data.read_bytes();eraw=events_path.read_bytes()
    assert sha(raw)==source["csv_sha256"] and sha(eraw)==source["events_csv_sha256"]
    rows=list(csv.DictReader(io.StringIO(raw.decode())));events=list(csv.DictReader(io.StringIO(eraw.decode())))
    assert len(rows)==source["bins"]==24 and len(events)==source["records"]==301
    start=datetime.fromisoformat(source["window_start_utc"]);end=datetime.fromisoformat(source["window_end_utc_exclusive"])
    assert (end-start).total_seconds()==86400
    counts=[0]*24
    for i,r in enumerate(events,1):
        t=datetime.fromisoformat(r["timestamp"]);assert start<=t<end
        hour=int((t-start).total_seconds()//3600);assert hour==int(r["hour_index"]);counts[hour]+=1
        assert int(r["event_index"])==i and int(r["source_line"])==source["source_line_numbers"][i-1]
    assert dict(Counter(r["kind"] for r in events))==source["kind_counts"]
    assert dict(Counter(r["protocol"] for r in events))==source["protocol_counts"]
    cumulative=[0]+list(accumulate(counts));assert counts==source["hourly_record_counts"] and sum(counts)==301
    for i,(r,n) in enumerate(zip(rows,counts)):
        assert int(r["hour_index"])==i and int(r["exposure_seconds"])==3600
        assert datetime.fromisoformat(r["start_utc"])==start+timedelta(hours=i)
        assert datetime.fromisoformat(r["end_utc"])==start+timedelta(hours=i+1)
        assert int(r["records"])==float(r["records_per_hour"])==n and int(r["cumulative_records"])==cumulative[i+1]
    font=Path(font_manager.findfont(font_manager.FontProperties(family="Liberation Serif"),fallback_to_default=False))
    plt.rcParams.update({"font.family":"Liberation Serif","font.size":9,"axes.labelsize":9,"xtick.labelsize":8.5,"ytick.labelsize":8.5,
                         "pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,(ax,total)=plt.subplots(1,2,figsize=(6.75,1.25),gridspec_kw={"width_ratios":[2,1]})
    fig.subplots_adjust(left=0.065,right=0.985,bottom=0.31,top=0.79,wspace=0.28)
    ax.bar([i-23.5 for i in range(24)],counts,width=0.82,color="#17649a",linewidth=0,label="Hourly records")
    ax.axhline(sum(counts)/24,color="#38945a",linestyle="--",linewidth=0.8,label=f"24h mean ({sum(counts)/24:.2f}/hr)")
    ax.set_xlim(-24,0);ax.set_ylim(0,max(counts)*1.2);ax.set_xticks([-24,-18,-12,-6,0])
    ax.set_xlabel("Hours before capture");ax.set_ylabel("Records / hour");ax.yaxis.set_major_locator(MaxNLocator(3,integer=True))
    ax.legend(loc="lower left",bbox_to_anchor=(0,1.01),borderaxespad=0.1,ncol=2,frameon=False,fontsize=8.5,handlelength=1.2,columnspacing=0.7)
    total.step(list(range(-24,1)),cumulative,where="post",color="#17649a",linewidth=1)
    total.set_xlim(-24,0);total.set_ylim(0,sum(counts)*1.08);total.set_xticks([-24,-12,0])
    total.set_xlabel("Hours before capture");total.set_ylabel("Records");total.yaxis.set_major_locator(MaxNLocator(3,integer=True))
    for a in [ax,total]:a.grid(axis="y",color="0.9",linewidth=0.4);a.set_axisbelow(True)
    fixed=datetime(2026,11,3,tzinfo=timezone.utc);args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,metadata={"Title":"Program-journal records over24hours","Author":"","Subject":"Captured records per elapsed calendar hour; no learning claim","CreationDate":fixed,"ModDate":fixed})
    plt.close(fig)
    provenance={"schema":"whitepaper.program-24h-figure/v1","csv_sha256":sha(raw),"events_csv_sha256":sha(eraw),"source_receipt_sha256":sha(source_path.read_bytes()),
                "generator_sha256":sha(Path(__file__).read_bytes()),"figure_sha256":sha(args.output.read_bytes()),"source_prefix_sha256":source["source_prefix_sha256"],
                "projection_sha256":source["projection_sha256"],"records":301,"hourly_record_counts":counts,"zero_count_hours":counts.count(0),"mean_records_per_hour":sum(counts)/24,
                "window_start_utc":source["window_start_utc"],"window_end_utc_exclusive":source["window_end_utc_exclusive"],"total_exposure_seconds":86400,"bin_exposure_seconds":3600,
                "scope":source["scope"],"matplotlib_version":matplotlib.__version__,"font_family":"Liberation Serif","font_file_sha256":sha(font.read_bytes()),
                "figure_inches":[6.75,1.25],"minimum_text_size_pt":8.5,"fixed_canvas_no_tight_crop":True,"reproduce":"python3 figures/scripts/make_program_24h.py"}
    args.provenance.parent.mkdir(parents=True,exist_ok=True);args.provenance.write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps({"records":301,"bins":24,"mean_per_hour":sum(counts)/24,"figure_sha256":provenance["figure_sha256"]}))

if __name__=="__main__":
    main()
