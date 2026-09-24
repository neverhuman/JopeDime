#!/usr/bin/env python3
"""Pin regenerated historical figures to retained extraction rows and original sources."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT/'data/training-history'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 specs=[
  ('training-historical-endpoints.png','block-endpoint-metrics.csv','108 blocks; two endpoints per block; four displayed objectives','recorded composite / NLL / BT / Huber loss','Changing corpora and fresh AdamW per block; not a continuous optimizer history.'),
  ('training-adaptive-validation.png','block-adaptive-validation.csv','108 block validation summaries; two displayed objectives','sequence NLL and pairwise ranking loss','Changing corpus/monitor and parent regimes; not untouched evaluation.'),
  ('training-fixed-monitor.png','fixed-monitor-metrics.csv','2048 examples per evaluation; recovery under two executables plus two challengers','sequence NLL, ranking BT, listwise cross-entropy, runtime Huber','Adaptive teaching monitor; identical recovery rows visually combined and retained in raw CSV.'),
  ('training-gen31-loss.png','gen31-epoch-metrics.csv','14 retained epochs; train and validation; four objectives','recorded component and composite losses','Separate Python lineage, changing learned weights; no embedded event date.'),
  ('training-gen31-quality.png','gen31-epoch-metrics.csv','14 retained epochs; four diagnostics','accuracy fraction, retained-target regret, accuracy difference, correlation','Retained teacher labels; not a measured new GP campaign or calibrated uncertainty.'),
  ('training-a100-checkpoints.png','a100-checkpoint-validation.csv','78 checkpoints; 22 corpus identities; two displayed objectives','optimizer step, sequence NLL, ranking BT','Changing adaptive monitors; step69000 is not accepted serving60000.'),
  ('training-rtx60100-steps.png','rtx60100-step-metrics.csv','100 actual steps 60001–60100; four displayed objectives','carried optimizer step and recorded minibatch loss','Separate RTX continuation; unchanged corpus, zero fresh programs; interruption boundary retained.'),
 ]
 records=[]
 for figure,data,coverage,units,limits in specs:
  rows=list(csv.DictReader((HISTORY/data).open()))
  sources={row['source_path']:row['source_sha256'] for row in rows}
  records.append(dict(figure=figure,figure_sha256=sha(ROOT/'figures'/figure),extracted_data=data,extracted_data_sha256=sha(HISTORY/data),raw_sources=sources,extraction_verification_command='python3 docs/internal/asm-manual/data/training-history/verify_history.py',extraction_command='python3 docs/internal/asm-manual/tools/reextract_training_metrics.py --output NEW_CUSTODY_DIRECTORY',extraction_scope='Re-extract metrics from original records using frozen CSV row-selection metadata; original one-off invocation was not retained.',plot_command='python3 docs/internal/asm-manual/tools/make_training_figures.py',retained_csv_rows=len(rows),coverage=coverage,units=units,interpretation_limits=limits))
 manifest=json.loads((ROOT/'data/per-step-history/source-manifest.json').read_text())
 records.append(dict(figure='training-a100-per-step.svg',figure_sha256=sha(ROOT/'figures/training-a100-per-step.svg'),raw_sources={s['path']:s['sha256'] for s in manifest['sources']},extraction_command='python3 docs/internal/asm-manual/tools/extract_per_step.py --source /home/ubuntu/jope-migration-evidence/claude/node-high-value/files/home/ubuntu/jope-dime/parallel/runs/eight-a100-recovery-20260912/learning --history docs/internal/asm-manual/data/training-history --output NEW_CUSTODY_DIRECTORY',plot_command='python3 docs/internal/asm-manual/tools/make_per_step_figure.py',coverage='63000 actual steps 6001–69000; 630 exact non-overlapping within-corpus means; 78 separate checkpoint validation observations',units='optimizer steps in thousands; recorded composite training loss; adaptive sequence NLL',interpretation_limits='21 training corpora; separate validation populations; no invented missing steps or cross-boundary interpolation; step60000 accepted, later steps unaccepted.'))
 result=dict(schema='jope.manual-historical-plot-provenance/v1',plots=records,fits=0,training_runs=0,synthetic_fixture_results_used=False)
 (ROOT/'data/historical-plot-provenance.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(plots=len(records),fits=0)))
if __name__=='__main__':main()
