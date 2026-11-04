#!/usr/bin/env python3
"""Replay a fixed aggregate admission/program-journal capture; no runtime inputs."""
from pathlib import Path
from decimal import Decimal
import csv,hashlib,io,json,math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.ticker import MaxNLocator

FIGURES=Path(__file__).resolve().parents[1]
SOURCE=FIGURES/'data/stopped_run_record.source.json'
ADMISSIONS=FIGURES/'data/stopped_run_admissions.csv'
PROMOTIONS=FIGURES/'data/stopped_run_promotions.csv'
OUTPUT=FIGURES/'fig_stopped_run_record.pdf'
PROVENANCE=FIGURES/'data/stopped_run_record.provenance.json'
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    source=json.loads(SOURCE.read_text())
    for p in [ADMISSIONS,PROMOTIONS]:
        assert sha(p.read_bytes())==source['csv_sha256'][p.name]
    rows=list(csv.DictReader(io.StringIO(ADMISSIONS.read_text())))
    events=list(csv.DictReader(io.StringIO(PROMOTIONS.read_text())))
    ids=[int(r['round']) for r in rows];lo,hi=source['round_window']
    assert len(rows)==source['admission_records']==4274 and [ids[0],ids[-1]]==[lo,hi]==[775,5049]
    assert all(b>a for a,b in zip(ids,ids[1:]))
    missing=sorted(set(range(lo,hi+1))-set(ids));assert missing==source['missing_admission_rounds']==[4331]
    assert len(events)==source['program_records']==3529
    erounds=[int(r['round']) for r in events]
    assert erounds==sorted(erounds) and [erounds[0],erounds[-1]]==source['program_record_round_window']
    xs=[];ys=[];gains=[];ties=[]
    for i,r in enumerate(rows):
        candidate,previous,best=[Decimal(r[k]) for k in ['candidate_utility','previous_best_utility','best_utility']]
        assert all(x.is_finite() for x in [candidate,previous,best])
        accepted=int(r['accepted']);assert accepted in [0,1] and accepted==int(r['action'].startswith('accept_'))
        assert best==(candidate if accepted else previous)
        if i:
            assert previous==Decimal(rows[i-1]['best_utility'])
            if ids[i]!=ids[i-1]+1:xs.append(float('nan'));ys.append(float('nan'))
        xs.append(ids[i]);ys.append(float(best))
        if accepted:
            assert candidate>=previous
            (gains if candidate>previous else ties).append((ids[i],float(best)))
    assert [r for r,_ in gains]==source['logged_increase_rounds']
    assert [r for r,_ in ties]==source['accepted_tie_rounds']
    bins=source['bins'];assert sum(b['program_records'] for b in bins)==len(events)
    for b in bins:
        assert b['program_records']==sum(b['first_round']<=r<=b['last_round'] for r in erounds)
        assert b['round_span']==b['last_round']-b['first_round']+1
    assert all(b['round_span']==200 for b in bins[:-1]) and bins[-1]['round_span']==75
    font=Path(font_manager.findfont('Liberation Serif',fallback_to_default=False))
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,'axes.labelsize':9,
        'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'path.simplify':False})
    fig,axes=plt.subplots(2,1,sharex=True,figsize=(7.0,3.4))
    fig.subplots_adjust(left=.105,right=.985,bottom=.14,top=.81,hspace=.45)
    ax=axes[0]
    candidate=[float(r['candidate_utility']) for r in rows]
    cand=ax.scatter(ids,candidate,s=3.5,color='#777f87',alpha=.28,linewidths=0,label='Candidate',zorder=2)
    retained=ax.step(xs,ys,where='post',color='#17649a',lw=1.1,label='Retained best',zorder=3)[0]
    stars=ax.scatter(*zip(*gains),marker='*',s=43,color='#b55b00',label='Logged increase (4)',zorder=5)
    diamonds=ax.scatter(*zip(*ties),marker='D',s=23,facecolors='none',edgecolors='#6d4ab1',lw=.8,label='Admitted tie (4)',zorder=6)
    gap=ax.axvline(missing[0],color='.50',lw=.7,ls=':',label='Missing decision (4331)')
    low=min(candidate+[v for v in ys if math.isfinite(v)]);high=max(candidate+[v for v in ys if math.isfinite(v)]);pad=(high-low)*.12
    ax.set(ylim=(low-pad,high+pad),ylabel='Validation-pool utility')
    ax.set_title('Logged adapter utility',loc='left',fontsize=9,pad=4)
    ax.yaxis.set_major_locator(MaxNLocator(4));ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax=axes[1]
    bars=ax.bar([(b['first_round']+b['last_round'])/2 for b in bins],
        [b['program_records'] for b in bins],width=[b['round_span']*.82 for b in bins],
        color='#216da0',edgecolor='white',lw=.4,zorder=3)
    bars[-1].set_hatch('///');bars[-1].set_edgecolor('#555555');bars[-1].set_facecolor('#e6eef5')
    ax.annotate('75 rounds',xy=((bins[-1]['first_round']+bins[-1]['last_round'])/2,bins[-1]['program_records']),
        xytext=(-5,12),textcoords='offset points',ha='right',fontsize=8.5)
    ax.set_title('Ordinary program-promotion records (200-round bins; final bin shorter)',loc='left',fontsize=9,pad=4)
    ax.set(ylabel='Records / bin',xlabel='Logged round',ylim=(0,max(b['program_records'] for b in bins)*1.15))
    ax.yaxis.set_major_locator(MaxNLocator(4,integer=True))
    for ax in axes:
        ax.set_xlim(lo-30,hi+30);ax.grid(axis='y',color='.88',lw=.4);ax.set_axisbelow(True)
    axes[1].set_xticks([775,1500,2250,3000,3750,4500,5049]);axes[1].get_xticklabels()[-1].set_ha('right')
    fig.legend(handles=[cand,retained,stars,diamonds,gap],loc='upper center',bbox_to_anchor=(.54,1.01),
        ncol=3,fontsize=8.5,frameon=False,handlelength=1.4,handletextpad=.4,columnspacing=1.1,labelspacing=.35)
    fig.savefig(OUTPUT,metadata={'Title':'Logged adapter utility and program-pool activity through round 5049',
        'Author':'','Subject':'Adaptive operating records; no independent learning claim',
        'CreationDate':None,'ModDate':None});plt.close(fig)
    facts=dict(schema='whitepaper.stopped-operating-record-figure/v1',
        input_csv_sha256=source['csv_sha256'],source_receipt_sha256=sha(SOURCE.read_bytes()),
        generator_sha256=sha(Path(__file__).read_bytes()),figure_sha256=sha(OUTPUT.read_bytes()),
        admission_points=len(rows),program_event_records=len(events),program_count_bins=len(bins),
        round_window=[lo,hi],missing_admission_rounds=missing,logged_increase_rounds=[r for r,_ in gains],
        accepted_tie_rounds=[r for r,_ in ties],last_bin_rounds=75,bin_counts=[b['program_records'] for b in bins],
        smoothing=False,cumulative_maximum=False,scope=source['scope'],source_binding=source['source_binding'],
        font_family='Liberation Serif',font_file_sha256=sha(font.read_bytes()),minimum_text_size_pt=8.5,
        figure_inches=[7.0,3.4],fixed_canvas_no_tight_crop=True,matplotlib_version=matplotlib.__version__,
        reproduce='python3 figures/scripts/make_stopped_run_record.py')
    PROVENANCE.write_text(json.dumps(facts,indent=2)+'\n');print(json.dumps(facts))

if __name__=='__main__':main()
