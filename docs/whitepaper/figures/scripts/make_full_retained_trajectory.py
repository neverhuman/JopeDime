#!/usr/bin/env python3
"""Replay every captured admission's retained utility, with missing rounds broken."""
from pathlib import Path
from decimal import Decimal
import csv,hashlib,io,json,math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

FIGURES=Path(__file__).resolve().parents[1]
DATA=FIGURES/'data/full_retained_trajectory.csv'
SOURCE=DATA.with_suffix('.source.json')
OUTPUT=FIGURES/'fig_fw_retained_best_plateau.pdf'
PROVENANCE=DATA.with_suffix('.provenance.json')
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    raw=DATA.read_bytes();source=json.loads(SOURCE.read_text());assert sha(raw)==source['csv_sha256']
    rows=list(csv.DictReader(io.StringIO(raw.decode())));ids=[int(r['round']) for r in rows]
    assert len(rows)==source['records'] and [ids[0],ids[-1]]==source['round_window']
    assert ids[0]==775 and all(b>a for a,b in zip(ids,ids[1:]))
    missing=sorted(set(range(ids[0],ids[-1]+1))-set(ids));assert missing==source['missing_rounds']
    xs=[];ys=[];gains=[];ties=[]
    for i,r in enumerate(rows):
        candidate,previous,best=[Decimal(r[k]) for k in ['candidate_utility','previous_best_utility','best_utility']]
        assert all(v.is_finite() for v in [candidate,previous,best])
        accepted=int(r['accepted']);assert accepted in [0,1]
        assert accepted==int(r['action'].startswith('accept_'))
        assert best==(candidate if accepted else previous)
        if i:assert previous==Decimal(rows[i-1]['best_utility'])
        if i and ids[i]!=ids[i-1]+1:xs.append(float('nan'));ys.append(float('nan'))
        xs.append(ids[i]);ys.append(float(best))
        if accepted:
            assert candidate>=previous
            (gains if candidate>previous else ties).append((ids[i],float(best)))
    assert [r for r,_ in gains]==source['strict_gain_rounds']
    assert [r for r,_ in ties]==source['tie_rounds']
    font=Path(font_manager.findfont('Liberation Serif',fallback_to_default=False))
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
    fig,ax=plt.subplots(figsize=(3.25,2.0));fig.subplots_adjust(left=.20,right=.98,bottom=.25,top=.75)
    line=ax.step(xs,ys,where='post',color='#17649a',lw=1.15,label='retained utility')[0]
    stars=ax.scatter(*zip(*gains),marker='*',s=37,color='#b55b00',zorder=4,label=f'strict gain ({len(gains)})')
    diamonds=ax.scatter(*zip(*ties),marker='D',s=23,facecolors='none',edgecolors='#6d4ab1',linewidths=.8,zorder=5,label=f'accepted tie ({len(ties)})')
    gap=None
    for n in missing:gap=ax.axvline(n,color='.50',lw=.7,ls=':',label='missing decision')
    handles=[line,stars,diamonds]+([gap] if gap is not None else [])
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,1.01),ncol=2,frameon=False,
               fontsize=8.5,handlelength=1.3,columnspacing=.9,handletextpad=.4,labelspacing=.22)
    values=[float(r['best_utility']) for r in rows];span=max(values)-min(values);assert span>0
    ax.set(xlim=(ids[0]-45,ids[-1]+45),ylim=(min(values)-span*.12,max(values)+span*.15),
           xlabel='logged round',ylabel='retained utility')
    ax.set_xticks([775,2000,3500,ids[-1]])
    ax.get_xticklabels()[-1].set_ha('right')
    ax.set_yticks([1.11,1.12,1.13,1.14]);ax.tick_params(labelsize=8.5)
    ax.grid(axis='y',color='.88',lw=.4);ax.set_axisbelow(True)
    fig.savefig(OUTPUT,metadata={'Title':'Full logged retained adapter utility','Author':'',
                                'Subject':'Adaptive admission record; independent learning remains unqualified',
                                'CreationDate':None,'ModDate':None});plt.close(fig)
    facts=dict(schema='whitepaper.full-retained-trajectory-figure/v1',csv_sha256=sha(raw),
               source_receipt_sha256=sha(SOURCE.read_bytes()),generator_sha256=sha(Path(__file__).read_bytes()),
               figure_sha256=sha(OUTPUT.read_bytes()),records=len(rows),round_window=[ids[0],ids[-1]],
               missing_rounds=missing,strict_gain_rounds=[x for x,_ in gains],tie_rounds=[x for x,_ in ties],
               plotted_states=len(rows),line_gap_breaks=len(missing),cumulative_maximum=False,smoothing=False,
               first_retained_utility=rows[0]['best_utility'],final_retained_utility=rows[-1]['best_utility'],
               scope=source['scope'],source_binding=source['source_binding'],font_family='Liberation Serif',
               font_file_sha256=sha(font.read_bytes()),minimum_text_size_pt=8.5,figure_inches=[3.25,2.0],
               matplotlib_version=matplotlib.__version__,fixed_canvas_no_tight_crop=True,
               reproduce='python3 figures/scripts/make_full_retained_trajectory.py')
    PROVENANCE.write_text(json.dumps(facts,indent=2)+'\n');print(json.dumps(facts))

if __name__=='__main__':main()
