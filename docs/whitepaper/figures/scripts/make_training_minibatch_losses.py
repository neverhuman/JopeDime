#!/usr/bin/env python3
"""Plot the first and last candidate training-minibatch losses in a fixed snapshot."""
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
DATA = ROOT / 'data/training_minibatch_losses.csv'
SOURCE = DATA.with_suffix('.source.json')
OUTPUT = ROOT / 'fig_training_minibatch_losses.pdf'
PROVENANCE = DATA.with_suffix('.provenance.json')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def main():
    raw = DATA.read_bytes()
    source = json.loads(SOURCE.read_text())
    assert sha(raw) == source['csv_sha256']
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    assert rows and len(rows) == source['records']
    rounds = [int(r['round']) for r in rows]
    assert rounds == list(range(rounds[0], rounds[-1] + 1))
    assert [rounds[0], rounds[-1]] == source['round_window']
    a,b = source['source_line_window']
    assert [int(r['source_line']) for r in rows] == list(range(a,b+1))
    assert all(int(r['optimizer_steps']) == 2 and len(r['source_record_sha256']) == 64 for r in rows)
    accepted = sum(int(r['adapter_accepted']) for r in rows)
    assert accepted == source['accepted']
    assert all(r['first_batch_order_sha256'] != r['last_batch_order_sha256'] for r in rows)
    values = [[Decimal(r[k]) for r in rows] for k in ['loss_first','loss_last']]
    assert all(v.is_finite() and v >= 0 for vv in values for v in vv)
    font = Path(font_manager.findfont('Liberation Serif', fallback_to_default=False))
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,
        'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax = plt.subplots(figsize=(3.375,1.75))
    fig.subplots_adjust(left=.20,right=.97,bottom=.30,top=.78)
    for ys,color,marker,label in zip(values,['#17649a','#c47719'],['o','x'],['First minibatch','Last minibatch']):
        ax.scatter(rounds,[float(v) for v in ys],s=17,color=color,marker=marker,label=label,zorder=3)
    lo,hi = min(v for vv in values for v in vv),max(v for vv in values for v in vv)
    pad = max(float(hi-lo)*.2,.0005)
    ax.set(xlim=(rounds[0]-.6,rounds[-1]+.6),ylim=(max(0,float(lo)-pad),float(hi)+pad),
           xlabel='Completed round',ylabel='Training-minibatch loss')
    ax.xaxis.set_major_locator(MaxNLocator(4,integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(style='plain',useOffset=False)
    ax.grid(axis='y',color='.9',linewidth=.4)
    ax.set_axisbelow(True)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=8.5,frameon=False,handletextpad=.2,borderpad=.1,columnspacing=.8)
    fixed = datetime.fromisoformat(source['captured_utc'])
    fig.savefig(OUTPUT,metadata={'Title':'Recorded candidate training-minibatch losses','Author':'',
        'Subject':'Different training minibatches; no matched loss reduction or retained learning claim',
        'CreationDate':fixed,'ModDate':fixed})
    plt.close(fig)
    prov = dict(schema='whitepaper.training-minibatch-losses-figure/v1',csv_sha256=sha(raw),
        source_receipt_sha256=sha(SOURCE.read_bytes()),source_prefix_sha256=source['source_prefix_sha256'],
        generator_sha256=sha(Path(__file__).read_bytes()),figure_sha256=sha(OUTPUT.read_bytes()),
        records=len(rows),points=2*len(rows),round_window=source['round_window'],accepted=accepted,
        minimum_loss=str(lo),maximum_loss=str(hi),
        transformation=source['transformation'],scope=source['scope'],font_family='Liberation Serif',
        font_file_sha256=sha(font.read_bytes()),matplotlib_version=matplotlib.__version__,
        figure_inches=[3.375,1.75],minimum_text_size_pt=8.5,fixed_canvas_no_tight_crop=True,
        reproduce='python3 figures/scripts/make_training_minibatch_losses.py')
    PROVENANCE.write_text(json.dumps(prov,indent=2)+'\n')
    print(json.dumps(dict(records=len(rows),points=2*len(rows),accepted=accepted,figure_sha256=prov['figure_sha256'])))
if __name__ == '__main__': main()
