#!/usr/bin/env python3
"""Plot all committed program-journal records in fixed round-identifier bins."""
from datetime import datetime
import csv,hashlib,io,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager,pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/program_record_windows.csv';SOURCE=DATA.with_suffix('.source.json')
OUTPUT=ROOT/'fig_program_record_windows.pdf';PROVENANCE=DATA.with_suffix('.provenance.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    raw=DATA.read_bytes();source=json.loads(SOURCE.read_text());assert sha(raw)==source['csv_sha256']
    rows=list(csv.DictReader(io.StringIO(raw.decode())));assert len(rows)==source['windows']==38
    starts=[int(r['window_start']) for r in rows];counts=[int(r['program_records']) for r in rows]
    assert starts==list(range(775,4562,100)) and sum(counts)==source['input_records']==3364
    assert [int(r['round_identifier_span']) for r in rows]==[100]*37+[87]
    assert [int(r['partial']) for r in rows]==[0]*37+[1] and min(counts)>=0
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,'axes.labelsize':9,
        'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.375,1.5));fig.subplots_adjust(left=.20,right=.97,bottom=.31,top=.96)
    bars=ax.bar(starts,counts,width=82,color='#397ba3',linewidth=0,zorder=3)
    bars[-1].set(hatch='////',edgecolor='#173c53',linewidth=.5)
    ax.set(xlim=(700,4630),ylim=(0,350),xticks=[775,1775,2775,3775,4475],
        yticks=[0,100,200,300],xlabel='First round identifier in bin',ylabel='Program records')
    ax.grid(axis='y',color='.9',linewidth=.4);ax.set_axisbelow(True)
    fixed=datetime.fromisoformat(source['captured_utc'])
    fig.savefig(OUTPUT,metadata={'Title':'Program-journal records by round window','Author':'',
        'Subject':'Unnormalized counts including a partial final identifier bin; no speed or learning verdict',
        'CreationDate':fixed,'ModDate':fixed});plt.close(fig)
    font=Path(font_manager.findfont('Liberation Serif',fallback_to_default=False))
    p=dict(schema='whitepaper.program-record-windows-figure/v1',csv_sha256=sha(raw),
        source_receipt_sha256=sha(SOURCE.read_bytes()),generator_sha256=sha(Path(__file__).read_bytes()),
        figure_sha256=sha(OUTPUT.read_bytes()),input_records=source['input_records'],
        plotted_windows=len(rows),round_window=source['round_window'],record_sum=sum(counts),
        full_windows=37,partial_final_identifier_span=87,partial_final_record_count=counts[-1],
        counts_not_normalized=True,filtering=False,scope=source['scope'],font_family='Liberation Serif',
        font_file_sha256=sha(font.read_bytes()),matplotlib_version=matplotlib.__version__,
        figure_inches=[3.375,1.5],minimum_text_size_pt=8.5,fixed_canvas_no_tight_crop=True,
        reproduce='python3 figures/scripts/make_program_record_windows.py')
    PROVENANCE.write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(windows=len(rows),records=sum(counts),figure_sha256=p['figure_sha256'])))
if __name__=='__main__':main()
