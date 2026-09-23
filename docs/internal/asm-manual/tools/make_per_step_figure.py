#!/usr/bin/env python3
"""Plot audited log observations, with explicit aggregation and changing corpora."""
import argparse, csv, gzip, hashlib, json, math, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 root=Path(__file__).resolve().parents[1]
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=root/'data/per-step-history');ap.add_argument('--history',type=Path,default=root/'data/training-history');ap.add_argument('--output',type=Path,default=root/'figures');ap.add_argument('--preview',action='store_true');a=ap.parse_args()
 paths=[a.data/'a100-step-metrics.csv.gz',a.data/'a100-step-bins.csv',a.data/'source-manifest.json',a.history/'a100-checkpoint-validation.csv']
 pins={str(p):sha(p) for p in paths}
 with gzip.open(paths[0],'rt') as f:rows=list(csv.DictReader(f))
 assert [int(r['step']) for r in rows]==list(range(6001,69001))
 bins=list(csv.DictReader(paths[1].open())); selected=[r for r in bins if r['metric']=='loss']
 assert len(selected)==630
 for b in selected:
  start=int(b['first_step'])-6001;part=rows[start:start+100]
  assert len(part)==100 and int(part[-1]['step'])==int(b['last_step']) and len({r['cycle'] for r in part})==1
  assert float(b['mean'])==statistics.fmean(float(r['loss']) for r in part)
 validation=[r for r in csv.DictReader(paths[3].open()) if r['metric']=='sequence_nll']
 assert len(validation)==78 and all(r['independent_evaluation']=='False' for r in validation)
 assert len({r['corpus_sha256'] for r in validation})==22 and len({r['panel_sha256'] for r in validation})==21
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'jope-per-step-v1','axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,1,figsize=(9.5,6.5),sharex=True)
 blue,orange='#246b8e','#cb7743'
 for cycle in range(1,22):
  b=[r for r in selected if int(r['cycle'])==cycle]
  axes[0].plot([(int(r['first_step'])+int(r['last_step']))/2000 for r in b],[float(r['mean']) for r in b],color=blue,linewidth=1.1,label='Non-overlapping 100-step means' if cycle==1 else None)
  if cycle>1:axes[0].axvline((3000*(cycle+1)+.5)/1000,color='#bbbbbb',alpha=.4,linewidth=.6)
 accepted=rows[60000-6001]
 axes[0].scatter([60],[float(accepted['loss'])],s=95,marker='*',color=orange,zorder=5,label='Actual accepted-checkpoint endpoint')
 axes[0].set_title('Training: all 63,000 observations retained; 630 plotted bin means',loc='left',fontweight='bold')
 axes[0].set_ylabel('Recorded composite loss');axes[0].legend(fontsize=8,loc='upper right')
 axes[0].text(.01,.035,'21 separate 3,000-step blocks / corpora; lines stop at each boundary.',transform=axes[0].transAxes,fontsize=8)
 axes[1].scatter([int(r['step'])/1000 for r in validation],[float(r['value']) for r in validation],s=18,color=blue,alpha=.8,label='Adaptive checkpoint NLL (78 observations)')
 v=next(r for r in validation if r['is_accepted_serving60000']=='True')
 axes[1].scatter([60],[float(v['value'])],s=95,marker='*',color=orange,zorder=5,label='Accepted step 60,000')
 axes[1].set_title('Validation: changing corpus and monitor; no connected learning trend',loc='left',fontweight='bold')
 axes[1].set_ylabel('Adaptive sequence NLL');axes[1].set_xlabel('Recorded optimizer step (thousands)')
 axes[1].legend(fontsize=8,loc='upper right')
 for ax in axes:
  ax.axvline(60,color=orange,linestyle='--',linewidth=1)
  ax.axvspan(60,69.6,color=orange,alpha=.07)
  ax.grid(alpha=.14);ax.set_axisbelow(True);ax.set_xlim(3,69.6)
 axes[1].set_xticks([4,6,15,30,45,60,69])
 fig.suptitle('Retained A100 recovery history — accepted 60,000 is not the latest retained 69,000',fontsize=11,fontweight='bold')
 fig.text(.055,.02,'Upper: exact non-overlapping means within each corpus, not raw individual points or interpolation.\nLower: 22 corpus identities / 21 adaptive panels; not independent generalization. Shading marks unaccepted later steps.',fontsize=8.5)
 fig.tight_layout(rect=(0,.085,1,.95))
 a.output.mkdir(exist_ok=True)
 output=a.output/'training-a100-per-step.svg';fig.savefig(output,bbox_inches='tight',metadata={'Date':None})
 if a.preview:fig.savefig(a.output/'visual-review.png',dpi=150,bbox_inches='tight')
 plt.close(fig)
 assert pins=={str(p):sha(p) for p in paths}
 receipt={'schema':'jope.manual-per-step-figure/v1','inputs':pins,'figure':{'path':str(output),'sha256':sha(output),'bytes':output.stat().st_size},'source_rows':63000,'verified_bins':630,'aggregation':'Non-overlapping100-step arithmetic mean; no line crosses corpus boundary.','validation_observations':78,'models_loaded':False,'training_or_fits':False}
 (a.output/'per-step-figure-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
