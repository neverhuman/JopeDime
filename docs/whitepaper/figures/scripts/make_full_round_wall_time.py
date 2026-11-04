#!/usr/bin/env python3
"""Replay full recorded round wall times and a prespecified trailing median."""
from pathlib import Path
from decimal import Decimal
import csv,hashlib,io,json,statistics
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

FIGURES=Path(__file__).resolve().parents[1]
DATA=FIGURES/'data/full_round_wall_time.csv'
SOURCE=DATA.with_suffix('.source.json')
OUTPUT=FIGURES/'fig_fw_latest_round_clock.pdf'
PROVENANCE=DATA.with_suffix('.provenance.json')
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    raw=DATA.read_bytes();source=json.loads(SOURCE.read_text());assert sha(raw)==source['csv_sha256']
    rows=list(csv.DictReader(io.StringIO(raw.decode())));ids=[int(r['round']) for r in rows]
    assert len(rows)==source['records'] and [ids[0],ids[-1]]==source['round_window']
    assert ids==list(range(ids[0],ids[-1]+1)) and ids[0]==775
    vals=[Decimal(r['wall_seconds']) for r in rows];assert all(v.is_finite() and v>0 for v in vals)
    window=source['trailing_median_rounds'];assert window==51
    medians=[statistics.median(vals[i-window+1:i+1]) for i in range(window-1,len(vals))]
    font=Path(font_manager.findfont('Liberation Serif',fallback_to_default=False))
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
    fig,ax=plt.subplots(figsize=(3.25,2.0));fig.subplots_adjust(left=.21,right=.98,bottom=.25,top=.82)
    ax.scatter(ids,[float(v) for v in vals],s=2.1,alpha=.28,color='#17649a',edgecolors='none',label='recorded round')
    ax.plot(ids[window-1:],[float(v) for v in medians],color='#a65b00',lw=1.1,label='51-round median')
    ax.set(xlim=(ids[0]-45,ids[-1]+45),yscale='log',ylim=(30,1600),xlabel='logged round',ylabel='round wall time (s)')
    ax.set_xticks([775,2000,3500,ids[-1]])
    ax.get_xticklabels()[-1].set_ha('right');ax.set_yticks([30,100,300,1000],labels=['30','100','300','1000'])
    ax.tick_params(labelsize=8.5);ax.minorticks_off();ax.grid(axis='y',color='.88',lw=.4);ax.set_axisbelow(True)
    fig.legend(loc='upper center',bbox_to_anchor=(.53,1.01),ncol=2,frameon=False,fontsize=8.5,
               handlelength=1.2,columnspacing=.8,handletextpad=.35)
    fig.savefig(OUTPUT,metadata={'Title':'Logged round wall times','Author':'',
                                'Subject':'Operating clocks include work and waits, not single-proposal runtime',
                                'CreationDate':None,'ModDate':None});plt.close(fig)
    facts=dict(schema='whitepaper.full-round-wall-time-figure/v1',csv_sha256=sha(raw),
               source_receipt_sha256=sha(SOURCE.read_bytes()),generator_sha256=sha(Path(__file__).read_bytes()),
               figure_sha256=sha(OUTPUT.read_bytes()),records=len(rows),round_window=[ids[0],ids[-1]],
               points=len(rows),trailing_median_rounds=window,median_points=len(medians),first_median_round=ids[window-1],
               minimum_seconds=str(min(vals)),maximum_seconds=str(max(vals)),raw_points_filtered=0,
               vertical_scale='logarithmic',scope=source['scope'],source_binding=source['source_binding'],
               font_family='Liberation Serif',font_file_sha256=sha(font.read_bytes()),minimum_text_size_pt=8.5,
               figure_inches=[3.25,2.0],matplotlib_version=matplotlib.__version__,fixed_canvas_no_tight_crop=True,
               reproduce='python3 figures/scripts/make_full_round_wall_time.py')
    PROVENANCE.write_text(json.dumps(facts,indent=2)+'\n');print(json.dumps(facts))

if __name__=='__main__':main()
