#!/usr/bin/env python3
"""Plot recorded candidate and retained utility differences by decision round."""
from datetime import datetime
from decimal import Decimal
import csv
import hashlib
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/recorded_decision_lift.csv'
SOURCE = DATA.with_suffix('.source.json')
OUTPUT = ROOT / 'fig_recorded_decision_lift.pdf'
PROVENANCE = DATA.with_suffix('.provenance.json')
sha = lambda b: hashlib.sha256(b).hexdigest()

def main():
    raw = DATA.read_bytes()
    source = json.loads(SOURCE.read_text())
    assert sha(raw) == source['csv_sha256']
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    rounds = [int(r['round']) for r in rows]
    assert len(rows) == source['records'] and rounds == sorted(set(rounds))
    assert [rounds[0], rounds[-1]] == source['round_window']
    assert sorted(set(range(rounds[0], rounds[-1] + 1)) - set(rounds)) == source['missing_rounds']
    candidate = [Decimal(r['candidate_lift']) for r in rows]
    retained = [Decimal(r['retained_lift']) for r in rows]
    assert all(v.is_finite() for v in candidate + retained) and min(retained) >= 0
    gains = [i for i,v in enumerate(retained) if v > 0]
    assert [rounds[i] for i in gains] == source['strict_gain_rounds']
    assert len({r['evaluation_identity_sha256'] for r in rows}) == 1
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,
        'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes = plt.subplots(2,1,sharex=True,figsize=(3.375,2.5))
    fig.subplots_adjust(left=.24,right=.97,bottom=.18,top=.85,hspace=.8)
    for ax,values,title,label in zip(axes,[candidate,retained],
            ['Before the decision','After the decision'],['Candidate lift','Retained lift']):
        ax.scatter(rounds,[float(v) for v in values],s=2.4,color='#92979c',linewidths=0,label='Recorded lift',zorder=3)
        ax.axhline(0,color='#4a4a4a',linewidth=.5,zorder=2)
        ax.set_yscale('symlog',linthresh=1e-4,linscale=1)
        ax.set(ylabel=label,xlim=(rounds[0]-30,rounds[-1]+30))
        ax.set_title(title,loc='left',fontsize=9,pad=3)
        ax.grid(axis='y',color='.9',linewidth=.4)
    axes[0].set(ylim=(-.018,.055),yticks=[-.01,0,.01])
    axes[1].set(ylim=(-.00005,.055),yticks=[0,.001,.01],xlabel='Recorded decision round',xticks=[775,2000,3500,4838])
    axes[0].set_yticklabels(['−0.01','0','0.01'])
    axes[1].set_yticklabels(['0','0.001','0.01'])
    gx=[rounds[i] for i in gains];gy=[float(retained[i]) for i in gains]
    axes[1].vlines(gx,0,gy,color='#17649a',linewidth=.7,zorder=4)
    axes[1].scatter(gx,gy,marker='*',s=38,color='#17649a',label='Strict gains (4)',zorder=5)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.57,1.0),ncol=2,
        fontsize=8.5,frameon=False,handletextpad=.2,columnspacing=.8)
    fixed=datetime.fromisoformat(source['captured_utc'])
    fig.savefig(OUTPUT,metadata={'Title':'Recorded utility lift per decision','Author':'',
        'Subject':'Reused development contexts; no independent learning claim','CreationDate':fixed,'ModDate':fixed})
    plt.close(fig)
    font=Path(font_manager.findfont('Liberation Serif',fallback_to_default=False))
    p=dict(schema='whitepaper.recorded-decision-lift-figure/v1',csv_sha256=sha(raw),
        source_receipt_sha256=sha(SOURCE.read_bytes()),generator_sha256=sha(Path(__file__).read_bytes()),
        figure_sha256=sha(OUTPUT.read_bytes()),records=len(rows),round_window=source['round_window'],
        missing_rounds=source['missing_rounds'],strict_gain_rounds=source['strict_gain_rounds'],
        accepted_tie_rounds=source['accepted_tie_rounds'],lift_values=2*len(rows),strict_gain_overlays=len(gains),
        symmetric_log_linear_threshold='0.0001',smoothing=False,imputation=False,connected_data_lines=False,
        scope=source['scope'],font_family='Liberation Serif',font_file_sha256=sha(font.read_bytes()),
        matplotlib_version=matplotlib.__version__,figure_inches=[3.375,2.5],minimum_text_size_pt=8.5,
        fixed_canvas_no_tight_crop=True,reproduce='python3 figures/scripts/make_recorded_decision_lift.py')
    PROVENANCE.write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(records=len(rows),lift_values=2*len(rows),strict_gains=len(gains),figure_sha256=p['figure_sha256'])))

if __name__ == '__main__':
    main()
