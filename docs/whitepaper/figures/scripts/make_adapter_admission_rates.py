#!/usr/bin/env python3
"""Plot adapter admissions in all captured current-run journal decisions."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MaxNLocator

def sha(b):
    return hashlib.sha256(b).hexdigest()

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=root/'data/adapter_admission_rates.csv')
    parser.add_argument('--output', type=Path, default=root/'fig_adapter_admission_rates.pdf')
    parser.add_argument('--provenance', type=Path, default=root/'data/adapter_admission_rates.provenance.json')
    args = parser.parse_args()
    source_path = args.data.with_suffix('.source.json')
    events_path = args.data.with_suffix('.events.csv')
    source = json.loads(source_path.read_text())
    raw, eraw = args.data.read_bytes(), events_path.read_bytes()
    assert sha(raw) == source['csv_sha256'] and sha(eraw) == source['events_csv_sha256']
    bins = list(csv.DictReader(io.StringIO(raw.decode())))
    events = list(csv.DictReader(io.StringIO(eraw.decode())))
    assert len(events) == source['source_prefix_records'] and len(bins) == source['bins']
    rounds = [int(r['round']) for r in events]
    assert all(b > a for a,b in zip(rounds, rounds[1:]))
    assert sorted(set(range(rounds[0],rounds[-1]+1))-set(rounds)) == source['missing_rounds']
    for i,r in enumerate(events,1):
        assert int(r['source_line']) == i and len(r['source_record_sha256']) == 64
        assert int(r['accepted']) == int(r['action'].startswith('accept_'))
    assert dict(Counter(r['action'] for r in events)) == source['action_counts']
    for i,b in enumerate(bins):
        group = events[200*i:200*(i+1)]
        c = Counter(r['action'] for r in group)
        n = sum(int(r['accepted']) for r in group)
        expected = dict(bin_index=i, first_source_line=200*i+1, last_source_line=200*i+len(group),
                        first_round=int(group[0]['round']), last_round=int(group[-1]['round']),
                        decisions=len(group), improvements=c['accept_improvement'], ties=c['accept_tie'],
                        rejections=c['restore_best_model_and_optimizer'], admissions=n)
        assert all(int(b[k]) == v for k,v in expected.items())
        assert float(b['admission_percent']) == 100*n/len(group)
    font = Path(font_manager.findfont(font_manager.FontProperties(family='Liberation Serif'), fallback_to_default=False))
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,'axes.labelsize':9,
                         'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(6.75,1.35))
    fig.subplots_adjust(left=.075,right=.985,bottom=.31,top=.78)
    centers = [(int(b['first_round'])+int(b['last_round']))/2 for b in bins]
    widths = [.78*(int(b['last_round'])-int(b['first_round'])+1) for b in bins]
    gains = [100*int(b['improvements'])/int(b['decisions']) for b in bins]
    ties = [100*int(b['ties'])/int(b['decisions']) for b in bins]
    ax.bar(centers,gains,width=widths,color='#17649a',label='Strict improvement',linewidth=0)
    ax.bar(centers,ties,bottom=gains,width=widths,color='#c48a25',label='Tie',linewidth=0)
    zeros = [x for x,b in zip(centers,bins) if int(b['admissions']) == 0]
    ax.scatter(zeros,[0]*len(zeros),s=13,facecolors='white',edgecolors='#17649a',linewidths=.7,zorder=4,clip_on=False,label='Zero admissions')
    for x,g,t,b in zip(centers,gains,ties,bins):
        if g+t: ax.text(x,g+t+.08,str(b['admissions']),ha='center',va='bottom',fontsize=8.5)
    ax.set_ylim(0,max(a+b for a,b in zip(gains,ties))*1.35)
    ax.set_xlim(rounds[0]-50,rounds[-1]+75)
    ax.set_xticks([775,1500,2250,3000,3750,4529])
    ax.set_xlabel('Logged round (200 decisions per bin; final bin: 154)')
    ax.set_ylabel('Admissions (%)')
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.grid(axis='y',color='.9',linewidth=.4);ax.set_axisbelow(True)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.015),borderaxespad=.1,ncol=3,frameon=False,fontsize=8.5,handlelength=1.2,columnspacing=1.1)
    fixed = datetime(2026,11,3,tzinfo=timezone.utc)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,metadata={'Title':'Adapter admission frequency across captured rounds','Author':'',
                                     'Subject':'Recorded decisions, including ties; not a learning curve',
                                     'CreationDate':fixed,'ModDate':fixed})
    plt.close(fig)
    provenance = {'schema':'whitepaper.adapter-admission-rates-figure/v1','csv_sha256':sha(raw),
                  'events_csv_sha256':sha(eraw),'source_receipt_sha256':sha(source_path.read_bytes()),
                  'generator_sha256':sha(Path(__file__).read_bytes()),'figure_sha256':sha(args.output.read_bytes()),
                  'source_prefix_sha256':source['source_prefix_sha256'],'rows':len(events),'bins':len(bins),
                  'bin_denominators':[int(b['decisions']) for b in bins], 'missing_rounds':source['missing_rounds'],
                  'action_counts':source['action_counts'],'scope':source['scope'],'matplotlib_version':matplotlib.__version__,
                  'font_family':'Liberation Serif','font_file_sha256':sha(font.read_bytes()),'figure_inches':[6.75,1.35],
                  'minimum_text_size_pt':8.5,'fixed_canvas_no_tight_crop':True,
                  'reproduce':'python3 figures/scripts/make_adapter_admission_rates.py'}
    args.provenance.parent.mkdir(parents=True,exist_ok=True)
    args.provenance.write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'rows':len(events),'bins':len(bins),'action_counts':source['action_counts'],
                      'figure_sha256':provenance['figure_sha256']}))

if __name__ == '__main__':
    main()
