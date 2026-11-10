#!/usr/bin/env python3
"""Replay counted program-form changes from a frozen promotion-journal capture."""
from pathlib import Path
import csv,hashlib,io,json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager

FIGURES=Path(__file__).resolve().parents[1]
DATA=FIGURES/'data/program_complexity_changes.csv'
SOURCE=DATA.with_suffix('.source.json')
OUTPUT=FIGURES/'fig_program_complexity_changes.pdf'
PROVENANCE=DATA.with_suffix('.provenance.json')
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    source=json.loads(SOURCE.read_text());raw=DATA.read_bytes()
    assert sha(raw)==source['csv_sha256']
    rows=list(csv.DictReader(io.StringIO(raw.decode())))
    assert len(rows)==source['records']==3529
    assert [int(r['event_index']) for r in rows]==list(range(1,len(rows)+1))
    assert [int(r['source_line']) for r in rows]==list(range(183,3712))
    rounds=[int(r['round']) for r in rows]
    assert rounds==sorted(rounds) and [rounds[0],rounds[-1]]==source['event_round_window']==[775,5047]
    xs=[];ys=[];counts=dict(fewer=0,same=0,more=0,unmatched=0)
    for r in rows:
        assert len(r['source_record_sha256'])==64 and int(r['candidate_form_count'])>=0
        matched=int(r['matched_prior']);assert matched in [0,1]
        if matched:
            prior,candidate=int(r['prior_form_count']),int(r['candidate_form_count'])
            delta=int(r['form_count_decrease']);assert delta==prior-candidate
            assert int(r['prior_source_line'])<int(r['source_line']) and len(r['prior_source_record_sha256'])==64
            xs.append(int(r['round']));ys.append(delta)
            counts['fewer' if delta>0 else 'same' if delta==0 else 'more']+=1
        else:
            assert all(r[k]=='' for k in ['prior_source_line','prior_source_record_sha256','prior_form_count','form_count_decrease'])
            counts['unmatched']+=1
    assert counts=={k:source['counts'][k] for k in counts}
    assert len(xs)==source['counts']['matched']==2554 and sum(counts.values())==3529
    font=Path(font_manager.findfont('Liberation Serif',fallback_to_default=False))
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,'axes.labelsize':9,
        'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,
        'mathtext.fontset':'custom','mathtext.rm':'Liberation Serif','mathtext.it':'Liberation Serif:italic',
        'mathtext.bf':'Liberation Serif:bold','mathtext.cal':'Liberation Serif:italic',
        'mathtext.sf':'Liberation Serif','mathtext.tt':'Liberation Serif','mathtext.fallback':None,
        'axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.45),gridspec_kw={'width_ratios':[1.85,1]})
    ax=axes[0];ax.scatter(xs,ys,s=3,alpha=.30,color='#17649a',linewidths=0)
    ax.axhline(0,color='.55',lw=.6,zorder=0)
    ax.set(xlim=(725,5099),ylim=(-4.5,3.5),xlabel='Logged round',ylabel='Prior − new form count')
    ax.set_xticks([775,2000,3500,5049]);ax.get_xticklabels()[-1].set_ha('right')
    ax.set_yticks([-4,-2,0,2,3]);ax.set_title('Matched journal pairs (2,554)',fontsize=9)
    ax=axes[1];names=['Fewer\nforms','Same\ncount','More\nforms','Unmatched'];values=[counts[k] for k in counts]
    bars=ax.bar(range(4),values,color=['#17649a']*3+['#969fa5'],width=.68)
    ax.bar_label(bars,padding=3,fontsize=8.5)
    ax.set(xticks=range(4),xticklabels=names,ylabel='Journal records',ylim=(0,max(values)*1.18))
    ax.set_yticks([0,400,800,1200]);ax.set_title('All ordinary promotions (3,529)',fontsize=9)
    for ax in axes:ax.grid(axis='y',color='.88',lw=.4);ax.set_axisbelow(True)
    fig.tight_layout(pad=.6,w_pad=1.2)
    fig.savefig(OUTPUT,metadata={'Title':'Counted program-form changes in recorded promotions','Author':'',
        'Subject':'Conditional training-program representation counts; no learning claim',
        'CreationDate':None,'ModDate':None});plt.close(fig)
    facts=dict(schema='whitepaper.program-complexity-changes-figure/v1',input_csv_sha256=sha(raw),
        source_receipt_sha256=sha(SOURCE.read_bytes()),generator_sha256=sha(Path(__file__).read_bytes()),
        figure_sha256=sha(OUTPUT.read_bytes()),records=len(rows),plotted_pairs=len(xs),counts=source['counts'],
        event_round_window=source['event_round_window'],selection_round_window=source['selection_round_window'],
        unknowns_imputed=False,jitter=False,smoothing=False,scope=source['scope'],source_binding=source['source_binding'],
        font_family='Liberation Serif',font_file_sha256=sha(font.read_bytes()),minimum_text_size_pt=8.5,
        figure_inches=[7.0,2.45],fixed_canvas_no_tight_crop=True,matplotlib_version=matplotlib.__version__,
        reproduce='python3 figures/scripts/make_program_complexity_changes.py')
    PROVENANCE.write_text(json.dumps(facts,indent=2)+'\n');print(json.dumps(facts))

if __name__=='__main__':main()
