#!/usr/bin/env python3
"""Read original training telemetry only. Write a new, immutable audit directory."""
import argparse, csv, gzip, hashlib, io, json, math, statistics
from pathlib import Path
from datetime import datetime, timezone

METRICS = ('loss', 'sequence_nll', 'ranking_bt', 'ranking_listwise', 'runtime_huber', 'gradient_norm')
FIELDS = ('cycle', 'source_line', 'step', 'local_step', 'timestamp') + METRICS

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def utc(n): return datetime.fromtimestamp(int(n),timezone.utc).isoformat()
def ols(x,y):
 mx=statistics.fmean(x);my=statistics.fmean(y)
 xx=math.fsum((v-mx)**2 for v in x); yy=math.fsum((v-my)**2 for v in y)
 slope=math.fsum((a-mx)*(b-my) for a,b in zip(x,y))/xx
 residual=math.fsum((b-my-slope*(a-mx))**2 for a,b in zip(x,y))
 return {'slope_per_step':slope,'r_squared':1-residual/yy,'fitted_span_delta':slope*(x[-1]-x[0])}
def write_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--history',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 out=a.output;out.mkdir(exist_ok=False)
 lineage_path=a.history/'a100-checkpoint-lineage.csv'; validation_path=a.history/'a100-checkpoint-validation.csv'
 lineage=list(csv.DictReader(lineage_path.open())); validation=list(csv.DictReader(validation_path.open()))
 states={int(r['step']):r for r in lineage};vals={(int(r['step']),r['metric']):r for r in validation}
 assert len(states)==78
 sources=[]; summaries=[];bins=[];allrows=[];matched=0;boundaries=[];numeric_count=0
 compressed=out/'a100-step-metrics.csv.gz'
 with compressed.open('xb') as raw, gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=9) as gz, io.TextIOWrapper(gz,encoding='utf-8',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator='\n');writer.writeheader()
  previous=None
  for cycle in range(1,22):
   path=a.source/f'cycle-{cycle:06d}/train-2/metrics.jsonl';before=sha(path)
   rows=[];identities=set();parents=set();lr_fields=set();contextfree=0
   with path.open() as f:
    for line_number,line in enumerate(f,1):
     # Keep each selected numeric token exactly as written, not rounded or reserialized.
     d=json.loads(line,parse_float=str,parse_int=str)
     step=int(d['step']);local=int(d['local_step'])
     assert local==line_number and step==3000*(cycle+1)+local
     assert (d['rank'],d['world'],d['global_batch'],d['microbatch'],d['accumulation'])==('0','8','2048','256','1')
     assert d['initialization']['optimizer']=='carried FP32 AdamW with bias-correction step'
     assert int(d['initialization']['initial_optimizer_step'])==3000*(cycle+1)
     assert d['initialization']['state_sha256']==states[3000*(cycle+1)]['state_sha256']
     identities.add(d['corpus_sha256']); parents.add(d['initialization']['state_sha256'])
     lr_fields.update(k for k in d if 'learning_rate' in k or k=='lr')
     contextfree += 'contextfree_bt' in d
     r={'cycle':cycle,'source_line':line_number,**{k:d[k] for k in FIELDS[2:]}}
     assert all(math.isfinite(float(r[k])) for k in METRICS)
     writer.writerow(r);rows.append(r);allrows.append(r);numeric_count+=len(METRICS)
     if d['validation'] is not None:
      v=d['validation']; assert v['independent_evaluation'] is False
      if step in states:
       assert d['corpus_sha256']==states[step]['corpus_sha256']
       for m in ('sequence_nll','ranking_bt','ranking_listwise','runtime_huber'):
        assert float(v[m])==float(vals[(step,m)]['value']);matched+=1
       assert v['panel_sha256']==vals[(step,'sequence_nll')]['panel_sha256']
   assert len(rows)==3000 and len(identities)==len(parents)==1 and not lr_fields
   assert before==sha(path)
   corpus=identities.pop();parent=parents.pop()
   assert corpus==states[int(rows[-1]['step'])]['corpus_sha256']
   source={'cycle':cycle,'path':str(path),'sha256':before,'bytes':path.stat().st_size,'rows':len(rows),'first_step':int(rows[0]['step']),'last_step':int(rows[-1]['step']),'first_timestamp_utc':utc(rows[0]['timestamp']),'last_timestamp_utc':utc(rows[-1]['timestamp']),'corpus_sha256':corpus,'parent_state_sha256':parent,'final_state_sha256':states[int(rows[-1]['step'])]['state_sha256'],'world':8,'rank':0,'global_batch':2048,'microbatch':256,'accumulation':1,'optimizer_policy':'carried FP32 AdamW with bias-correction step','contextfree_bt_rows':contextfree,'per_step_learning_rate_fields':sorted(lr_fields)}
   sources.append(source)
   if previous: boundaries.append({'next_cycle':cycle,'previous_last_step':int(previous[-1]['step']),'next_first_step':int(rows[0]['step']),'loss_point_jump':float(rows[0]['loss'])-float(previous[-1]['loss']),'loss_100_step_mean_jump':statistics.fmean(float(r['loss']) for r in rows[:100])-statistics.fmean(float(r['loss']) for r in previous[-100:])})
   for metric in METRICS:
    values=[float(r[metric]) for r in rows];steps=[int(r['step']) for r in rows]
    summaries.append({'cycle':cycle,'metric':metric,'count':len(rows),'first_step':steps[0],'last_step':steps[-1],'first_value':rows[0][metric],'last_value':rows[-1][metric],'mean':statistics.fmean(values),'first_100_mean':statistics.fmean(values[:100]),'last_100_mean':statistics.fmean(values[-100:]),'last100_minus_first100':statistics.fmean(values[-100:])-statistics.fmean(values[:100]),**ols(steps,values),'corpus_sha256':corpus,'source_sha256':before})
    for start in range(0,3000,100):
     subset=rows[start:start+100]
     bins.append({'cycle':cycle,'metric':metric,'first_step':subset[0]['step'],'last_step':subset[-1]['step'],'count':100,'mean':statistics.fmean(float(r[metric]) for r in subset),'corpus_sha256':corpus})
   previous=rows
 assert [int(r['step']) for r in allrows]==list(range(6001,69001))
 assert len({s['corpus_sha256'] for s in sources})==21
 # Independent read-back, comparing every selected token against the originals again.
 checked=0
 with gzip.open(compressed,'rt',newline='') as f:
  reader=csv.DictReader(f)
  for src in sources:
   with Path(src['path']).open() as original:
    for n,line in enumerate(original,1):
     d=json.loads(line,parse_float=str,parse_int=str);r=next(reader)
     assert int(r['cycle'])==src['cycle'] and int(r['source_line'])==n
     assert all(r[k]==d[k] for k in FIELDS[2:]);checked+=1
   assert sha(Path(src['path']))==src['sha256']
  assert next(reader,None) is None
 write_csv(out/'a100-block-statistics.csv',summaries);write_csv(out/'a100-step-bins.csv',bins);write_csv(out/'a100-corpus-boundaries.csv',boundaries)
 selected=[r for r in allrows if int(r['step'])<=60000]
 loss_summaries=[r for r in summaries if r['metric']=='loss']
 analysis={'schema':'jope.manual-per-step-analysis/v1','rows':len(allrows),'selected_numeric_metric_cells':numeric_count,'exact_token_roundtrip_rows':checked,'checkpoint_validation_cells_crossmatched':matched,'first_step':6001,'last_step':69000,'accepted_step':60000,'unaccepted_retained_last_step':69000,'corpus_count':21,'block_count':21,'rows_per_block':3000,'loss_first':allrows[0]['loss'],'loss_accepted60000':selected[-1]['loss'],'loss_last':allrows[-1]['loss'],'descriptive_ols_6001_60000':ols([int(r['step']) for r in selected],[float(r['loss']) for r in selected]),'block_loss_last100_minus_first100':{'min':min(r['last100_minus_first100'] for r in loss_summaries),'median':statistics.median(r['last100_minus_first100'] for r in loss_summaries),'max':max(r['last100_minus_first100'] for r in loss_summaries)},'block_ols_negative_count':sum(r['slope_per_step']<0 for r in loss_summaries),'block_endpoint_decrease_count':sum(float(r['last_value'])<float(r['first_value']) for r in loss_summaries),'boundary_point_increase_count':sum(r['loss_point_jump']>0 for r in boundaries),'boundary_100mean_increase_count':sum(r['loss_100_step_mean_jump']>0 for r in boundaries),'first_timestamp_utc':min(s['first_timestamp_utc'] for s in sources),'last_timestamp_utc':max(s['last_timestamp_utc'] for s in sources),'uncertainty':'Descriptive OLS only. Serially dependent samples and changing corpora; no inferential confidence, independent evaluation or downstream quality claim.'}
 (out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
 manifest={'schema':'jope.manual-per-step-sources/v1','sources':sources,'column_json_pointers':{k:'/'+k for k in FIELDS if k not in ('cycle','source_line')},'row_binding':'cycle resolves source file; source_line is one-based original JSONL line. Exact selected numeric tokens are retained. This is not a complete copy of all telemetry fields.','history_inputs':[{ 'path':str(p),'sha256':sha(p)} for p in (lineage_path,validation_path)],'bound_constant_fields':'Per-file identities and world/rank/batch/optimizer checked on every row; parent hash matched previously audited checkpoint lineage; every retained validation cell matches prior normalized checkpoint evidence.','missing_scope':'No metrics.jsonl exists at the inspected cycle-000000/train-2 path. Steps3001–6000 have checkpoint metadata but no per-step log here; antecedent steps1–3000 are outside this inspected per-step source set. No global absence assertion.','omitted_fields':'Other telemetry timing/counters, nested validation and initialization strings remain in pinned original logs. contextfree_bt and per-step learning rate are absent as top-level fields in these63000 rows; no zero values are imputed.'}
 (out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(analysis,indent=2))
 print('gzip bytes',compressed.stat().st_size,'sha256',sha(compressed))
if __name__=='__main__':main()
