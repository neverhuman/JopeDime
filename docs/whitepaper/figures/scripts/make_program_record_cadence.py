#!/usr/bin/env python3
"""Plot all adjacent timestamp intervals of a pinned program-journal projection."""
from datetime import datetime
from decimal import Decimal
import csv,hashlib,io,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager,pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/program_record_cadence.csv';SOURCE=DATA.with_suffix('.source.json')
OUTPUT=ROOT/'fig_program_record_cadence.pdf';PROVENANCE=DATA.with_suffix('.provenance.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    raw=DATA.read_bytes();source=json.loads(SOURCE.read_text());assert sha(raw)==source['csv_sha256']
    rows=list(csv.DictReader(io.StringIO(raw.decode())));assert len(rows)==source['intervals']
    rounds=[int(r['ending_round']) for r in rows];gaps=[Decimal(r['interval_seconds']) for r in rows]
    assert min(gaps)>0 and all(v.is_finite() for v in gaps)
    assert all(Decimal(r['interval_us'])/Decimal(1000000)==v for r,v in zip(rows,gaps))
    assert [int(r['current_event_index']) for r in rows]==list(range(2,source['input_records']+1))
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,'axes.labelsize':9,
        'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.375,1.35));fig.subplots_adjust(left=.23,right=.96,bottom=.34,top=.94)
    ax.scatter(rounds,[float(v) for v in gaps],s=2,color='#397ba3',alpha=.65,linewidths=0,zorder=3)
    ax.set_yscale('log');ax.set(xlim=(min(rounds)-50,max(rounds)+50),
        ylim=(min(.02,float(min(gaps))*.7),max(10000,float(max(gaps))*2)),
        xticks=[775,2000,3500,4561],yticks=[.1,10,1000],
        xlabel='Ending recorded round',ylabel='Record gap (s)')
    ax.set_yticklabels(['0.1','10','1000']);ax.grid(axis='y',color='.9',linewidth=.4)
    fixed=datetime.fromisoformat(source['captured_utc'])
    fig.savefig(OUTPUT,metadata={'Title':'Program-journal recording cadence','Author':'',
        'Subject':'Timestamp intervals include work, waits and batching; no speed verdict',
        'CreationDate':fixed,'ModDate':fixed});plt.close(fig)
    font=Path(font_manager.findfont('Liberation Serif',fallback_to_default=False))
    p=dict(schema='whitepaper.program-record-cadence-figure/v1',csv_sha256=sha(raw),
        source_receipt_sha256=sha(SOURCE.read_bytes()),generator_sha256=sha(Path(__file__).read_bytes()),
        figure_sha256=sha(OUTPUT.read_bytes()),input_records=source['input_records'],
        plotted_intervals=len(rows),round_window=source['round_window'],
        minimum_interval_seconds=str(min(gaps)),maximum_interval_seconds=str(max(gaps)),
        logarithmic_axis=True,connected_data_lines=False,smoothing=False,filtering=False,
        scope=source['scope'],font_family='Liberation Serif',font_file_sha256=sha(font.read_bytes()),
        matplotlib_version=matplotlib.__version__,figure_inches=[3.375,1.35],minimum_text_size_pt=8.5,
        fixed_canvas_no_tight_crop=True,reproduce='python3 figures/scripts/make_program_record_cadence.py')
    PROVENANCE.write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(intervals=len(rows),figure_sha256=p['figure_sha256'])))
if __name__=='__main__':main()
