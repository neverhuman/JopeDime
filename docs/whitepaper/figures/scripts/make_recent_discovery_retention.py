#!/usr/bin/env python3
"""Plot a fixed, source-bound pair of distinct admission records."""
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
from matplotlib.ticker import MaxNLocator
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/recent_discovery_retention.csv'
SOURCE = DATA.with_suffix('.source.json')
OUTPUT = ROOT / 'fig_recent_discovery_retention.pdf'
PROVENANCE = DATA.with_suffix('.provenance.json')
sha = lambda raw: hashlib.sha256(raw).hexdigest()

def main():
    raw = DATA.read_bytes()
    source = json.loads(SOURCE.read_text())
    assert sha(raw) == source['csv_sha256']
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    assert len(rows) == source['records']
    rounds = [int(r['round']) for r in rows]
    assert rounds == list(range(rounds[0], rounds[-1] + 1))
    assert [rounds[0], rounds[-1]] == source['round_window']
    counts = [int(r['program_promotions']) for r in rows]
    assert min(counts) >= 0 and sum(counts) == source['program_promotions']
    assert all(len(r['program_source_lines'].split(';')) == v if v else r['program_source_lines'] == '' for r,v in zip(rows,counts))
    assert all(len(r['summary_record_sha256']) == 64 for r in rows)
    assert len({r['evaluation_identity_sha256'] for r in rows}) == 1
    candidate = [Decimal(r['candidate_utility']) for r in rows]
    retained = [Decimal(r['retained_utility']) for r in rows]
    assert all(v.is_finite() for v in candidate + retained)
    assert sum(int(r['adapter_accepted']) for r in rows) == source['adapter_admissions']
    font = Path(font_manager.findfont('Liberation Serif', fallback_to_default=False))
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,
        'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes = plt.subplots(2,1,sharex=True,figsize=(3.375,2.3))
    fig.subplots_adjust(left=.22,right=.97,bottom=.18,top=.82,hspace=.65)
    axes[0].bar(rounds,counts,width=.72,color='#438669',zorder=3)
    axes[0].set_title('Training-program promotions',loc='left',fontsize=9,pad=4)
    axes[0].set_ylabel('Events / round')
    axes[0].set_ylim(0,max(1,max(counts))*1.18)
    axes[0].yaxis.set_major_locator(MaxNLocator(3,integer=True))
    axes[1].scatter(rounds,[float(v) for v in candidate],s=13,color='#8c9095',label='Candidate',zorder=3)
    axes[1].plot(rounds,[float(v) for v in retained],color='#17649a',linewidth=1.2,label='Retained',zorder=4)
    axes[1].set_title('Development-pool utility',loc='left',fontsize=9,pad=4)
    lo,hi=min(candidate+retained),max(candidate+retained)
    pad=max(float(hi-lo)*.12,.0002)
    axes[1].set(ylim=(float(lo)-pad,float(hi)+pad),ylabel='Utility',xlabel='Completed round')
    axes[1].yaxis.set_major_locator(MaxNLocator(3))
    axes[1].xaxis.set_major_locator(MaxNLocator(4,integer=True))
    for ax in axes:
        ax.set_xlim(rounds[0]-.7,rounds[-1]+.7)
        ax.ticklabel_format(style='plain',useOffset=False)
        ax.grid(axis='y',color='.9',linewidth=.4)
        ax.set_axisbelow(True)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.56,1.005),
        ncol=2,fontsize=8.5,frameon=False,handletextpad=.3,columnspacing=1)
    fixed=datetime.fromisoformat(source['captured_utc'])
    fig.savefig(OUTPUT,metadata={'Title':'Program promotion and adapter retention',
        'Author':'','Subject':'Distinct admission records; no independent learning claim',
        'CreationDate':fixed,'ModDate':fixed})
    plt.close(fig)
    p=dict(schema='whitepaper.recent-discovery-retention-figure/v1',csv_sha256=sha(raw),
        source_receipt_sha256=sha(SOURCE.read_bytes()),generator_sha256=sha(Path(__file__).read_bytes()),
        figure_sha256=sha(OUTPUT.read_bytes()),records=len(rows),program_promotions=sum(counts),
        adapter_admissions=source['adapter_admissions'],utility_observations=2*len(rows),
        round_window=source['round_window'],summary_prefix_sha256=source['summary_prefix_sha256'],
        program_prefix_sha256=source['program_prefix_sha256'],transformation=source['transformation'],
        scope=source['scope'],font_family='Liberation Serif',font_file_sha256=sha(font.read_bytes()),
        matplotlib_version=matplotlib.__version__,figure_inches=[3.375,2.3],
        minimum_text_size_pt=8.5,fixed_canvas_no_tight_crop=True,
        reproduce='python3 figures/scripts/make_recent_discovery_retention.py')
    PROVENANCE.write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(records=len(rows),program_promotions=sum(counts),figure_sha256=p['figure_sha256'])))

if __name__ == '__main__':
    main()
