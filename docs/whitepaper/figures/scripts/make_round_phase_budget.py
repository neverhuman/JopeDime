#!/usr/bin/env python3
"""Plot recorded round-clock phase accounting for a frozen operational window."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
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
    parser.add_argument("--data",type=Path,default=root/"data/round_phase_budget.csv")
    parser.add_argument("--output",type=Path,default=root/"fig_round_phase_budget.pdf")
    parser.add_argument("--provenance",type=Path,default=root/"data/round_phase_budget.provenance.json")
    args=parser.parse_args();source_path=args.data.with_suffix(".source.json");source=json.loads(source_path.read_text())
    raw=args.data.read_bytes();assert sha(raw)==source["csv_sha256"]
    rows=list(csv.DictReader(io.StringIO(raw.decode())));assert len(rows)==source["records"]==20
    rounds=[int(r["round"]) for r in rows];assert rounds==source["round_ids"]==sorted(set(rounds))
    assert [int(r["record_index"]) for r in rows]==list(range(1,21))
    assert [int(r["source_line"]) for r in rows]==source["source_line_numbers"]
    wall=[float(r["wall_seconds"]) for r in rows];phases=source["phase_fields"]
    for r in rows:
        assert all(math.isfinite(float(r[k])) and float(r[k])>=0 for k in ["wall_seconds",*phases,"remainder_seconds"])
        assert float(r["remainder_seconds"])==float(r["wall_seconds"])-math.fsum(float(r[k]) for k in phases)
    names=["GP interval","Gate interval","Update phases","Headroom","Remainder"]
    colors=["#17649a","#7c8796","#38945a","#dda329","#c4c4c4"]
    components=[[float(r["gp"]) for r in rows],[float(r["dev_gate"]) for r in rows],
                [math.fsum(float(r[k]) for k in source["update_phases"]) for r in rows],
                [float(r["headroom_wait"]) for r in rows],[float(r["remainder_seconds"]) for r in rows]]
    assert all(math.isclose(math.fsum(v[i] for v in components),wall[i],abs_tol=1e-9) for i in range(len(rows)))
    total=math.fsum(wall);shares=[100*math.fsum(v)/total for v in components]
    assert math.isclose(total,source["wall_seconds_total"],abs_tol=1e-9) and math.isclose(math.fsum(shares),100,abs_tol=1e-10)
    font=Path(font_manager.findfont(font_manager.FontProperties(family="Liberation Serif"),fallback_to_default=False))
    plt.rcParams.update({"font.family":"Liberation Serif","font.size":9,"axes.labelsize":9,"xtick.labelsize":8.5,"ytick.labelsize":8.5,
                         "pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,(ax,summary)=plt.subplots(1,2,figsize=(6.75,1.60),gridspec_kw={"width_ratios":[2,1]})
    fig.subplots_adjust(left=0.065,right=0.98,bottom=0.28,top=0.79,wspace=0.67)
    bottoms=[0.0]*len(rows)
    for label,color,values in zip(names,colors,components):
        ax.bar(rounds,values,bottom=bottoms,width=0.78,color=color,linewidth=0,label=label)
        bottoms=[a+b for a,b in zip(bottoms,values)]
    ax.set_xlim(rounds[0]-0.7,rounds[-1]+0.7);ax.set_ylim(0,max(wall)*1.12)
    ax.set_xlabel("Logged round");ax.set_ylabel("Seconds")
    ax.xaxis.set_major_locator(MaxNLocator(5,integer=True));ax.yaxis.set_major_locator(MaxNLocator(3))
    summary.barh(range(5),shares,color=colors,height=0.65,linewidth=0)
    summary.set_yticks(range(5),names);summary.invert_yaxis();summary.set_xlim(0,100)
    summary.set_xlabel("Window wall time (%)");summary.set_xticks([0,50,100])
    for i,v in enumerate(shares):summary.annotate(f"{v:.1f}",(v,i),xytext=(3,0),textcoords="offset points",va="center",fontsize=8.5)
    ax.grid(axis="y",color="0.9",linewidth=0.4);ax.set_axisbelow(True)
    summary.grid(axis="x",color="0.9",linewidth=0.4);summary.set_axisbelow(True)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=5,frameon=False,fontsize=8.5,handlelength=1.0,columnspacing=0.8,borderaxespad=0.1)
    fixed=datetime(2026,11,3,tzinfo=timezone.utc)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,metadata={"Title":"Recorded round-clock phase accounting","Author":"","Subject":"Operational timing; no quality or learning claim","CreationDate":fixed,"ModDate":fixed})
    plt.close(fig)
    provenance={"schema":"whitepaper.round-phase-budget-figure/v1","csv_sha256":sha(raw),"source_receipt_sha256":sha(source_path.read_bytes()),
                "generator_sha256":sha(Path(__file__).read_bytes()),"figure_sha256":sha(args.output.read_bytes()),"source_prefix_sha256":source["source_prefix_sha256"],
                "timing_projection_sha256":source["timing_projection_sha256"],"records":20,"round_ids":rounds,"wall_seconds_total":total,
                "phase_labels":names,"phase_share_percent":dict(zip(names,shares)),"scope":source["scope"],"matplotlib_version":matplotlib.__version__,
                "font_family":"Liberation Serif","font_file_sha256":sha(font.read_bytes()),"figure_inches":[6.75,1.60],"minimum_text_size_pt":8.5,
                "fixed_canvas_no_tight_crop":True,"reproduce":"python3 figures/scripts/make_round_phase_budget.py"}
    args.provenance.parent.mkdir(parents=True,exist_ok=True);args.provenance.write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps({"records":len(rows),"wall_seconds_total":total,"phase_share_percent":dict(zip(names,shares)),"figure_sha256":provenance["figure_sha256"]}))

if __name__=="__main__":
    main()
