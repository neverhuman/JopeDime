#!/usr/bin/env python3
"""Plot all captured full-cohort program records in fixed calendar-time bins."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal
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
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=root/'data/full_cohort_program_rates.csv')
    parser.add_argument('--output',type=Path,default=root/'fig_full_cohort_program_rates.pdf')
    parser.add_argument('--provenance',type=Path,default=root/'data/full_cohort_program_rates.provenance.json')
    args=parser.parse_args();source_path=args.data.with_suffix('.source.json');events_path=args.data.with_suffix('.events.csv')
    source=json.loads(source_path.read_text());raw=args.data.read_bytes();eraw=events_path.read_bytes()
    assert sha(raw)==source['csv_sha256'] and sha(eraw)==source['events_csv_sha256']
    rows=list(csv.DictReader(io.StringIO(raw.decode())));events=list(csv.DictReader(io.StringIO(eraw.decode())))
    assert len(rows)==source['bins'] and len(events)==source['selected_records']
    start=datetime.fromisoformat(source['observation_start_utc']);end=datetime.fromisoformat(source['observation_end_utc'])
    def us(dt):
        return (dt.days*86400+dt.seconds)*1000000+dt.microseconds
    duration=us(end-start);assert duration==source['observation_us']
    counts=[0]*len(rows)
    for i,r in enumerate(events,1):
        assert int(r['event_index'])==i and int(r['round'])>=775 and len(r['source_record_sha256'])==64
        t=datetime.fromisoformat(r['timestamp']);assert start<=t<=end
        index=min(us(t-start)//source['bin_duration_us'],len(rows)-1)
        assert int(r['bin_index'])==index;counts[index]+=1
    assert counts==source['bin_record_counts'] and dict(Counter(r['kind'] for r in events))==source['kind_counts']
    assert dict(Counter(r['protocol'] for r in events))==source['protocol_counts']
    assert source['source_prefix_records']==len(events)+source['warmup_records_excluded']+source['post_cutoff_records_excluded']
    for i,(r,n) in enumerate(zip(rows,counts)):
        a=i*source['bin_duration_us'];b=min((i+1)*source['bin_duration_us'],duration)
        assert int(r['bin_index'])==i and int(r['start_offset_us'])==a and int(r['end_offset_us'])==b
        assert int(r['exposure_us'])==b-a>0 and int(r['records'])==n
        assert us(datetime.fromisoformat(r['start_utc'])-start)==a and us(datetime.fromisoformat(r['end_utc'])-start)==b
        assert Decimal(r['records_per_hour'])==Decimal(n)*Decimal(3600000000)/Decimal(b-a)
        assert int(r['partial'])==int(b-a<source['bin_duration_us'])
    font=Path(font_manager.findfont(font_manager.FontProperties(family='Liberation Serif'),fallback_to_default=False))
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,'axes.labelsize':9,'xtick.labelsize':8.5,
                         'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(3.25,1.40));fig.subplots_adjust(left=.16,right=.98,bottom=.29,top=.89)
    centers=[(int(r['start_offset_us'])+int(r['end_offset_us']))/7200000000 for r in rows]
    widths=[.85*int(r['exposure_us'])/3600000000 for r in rows]
    rates=[float(r['records_per_hour']) for r in rows]
    ax.bar(centers,rates,width=widths,color='#17649a',linewidth=0)
    for x,y,w,r in zip(centers,rates,widths,rows):
        if int(r['partial']):
            ax.bar(x,y,width=w,facecolor='none',edgecolor='0.25',hatch='///',linewidth=.5)
            ax.annotate('Partial',(x,y),xytext=(-4,10),textcoords='offset points',fontsize=8.5,ha='right',va='bottom')
    mean=float(source['mean_records_per_hour'])
    ax.axhline(mean,color='#38945a',linestyle='--',linewidth=.8,label='Window mean')
    ax.set_xlim(0,duration/3600000000);ax.set_ylim(0,max(rates)*1.22);ax.set_xticks([0,24,48,72,96])
    ax.set_xlabel('Hours since first full-cohort record');ax.set_ylabel('Records / hour')
    ax.yaxis.set_major_locator(MaxNLocator(3));ax.grid(axis='y',color='.9',linewidth=.4);ax.set_axisbelow(True)
    ax.legend(loc='upper right',frameon=False,fontsize=8.5,handlelength=1.2,borderaxespad=.1)
    fixed=datetime(2026,11,3,tzinfo=timezone.utc);args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,metadata={'Title':'Full-cohort program-journal rates','Author':'',
                                     'Subject':'Captured records per calendar hour, not learning or throughput',
                                     'CreationDate':fixed,'ModDate':fixed});plt.close(fig)
    provenance={'schema':'whitepaper.full-cohort-program-rates-figure/v1','csv_sha256':sha(raw),'events_csv_sha256':sha(eraw),
                'source_receipt_sha256':sha(source_path.read_bytes()),'generator_sha256':sha(Path(__file__).read_bytes()),
                'figure_sha256':sha(args.output.read_bytes()),'source_prefix_sha256':source['source_prefix_sha256'],
                'source_records':source['source_prefix_records'],'plotted_records':len(events),'round_window':source['selected_round_window'],
                'bins':len(rows),'bin_record_counts':counts,'bin_duration_us':source['bin_duration_us'],
                'final_bin_exposure_us':source['final_bin_exposure_us'],'total_exposure_us':duration,
                'observation_start_utc':source['observation_start_utc'],'observation_end_utc':source['observation_end_utc'],
                'warmup_records_excluded':source['warmup_records_excluded'],'scope':source['scope'],
                'matplotlib_version':matplotlib.__version__,'font_family':'Liberation Serif','font_file_sha256':sha(font.read_bytes()),
                'figure_inches':[3.25,1.40],'minimum_text_size_pt':8.5,'fixed_canvas_no_tight_crop':True,
                'reproduce':'python3 figures/scripts/make_full_cohort_program_rates.py'}
    args.provenance.parent.mkdir(parents=True,exist_ok=True);args.provenance.write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'records':len(events),'bins':len(rows),'figure_sha256':provenance['figure_sha256']}))

if __name__=='__main__':
    main()
