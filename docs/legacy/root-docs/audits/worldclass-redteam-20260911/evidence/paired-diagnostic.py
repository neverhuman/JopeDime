from pathlib import Path
import json,hashlib,collections,statistics,math,time
root=Path('/home/ubuntu/jope-dime/parallel');f=root/'runs/continuous-learning-v2-20260909/generation-000066/quality-report.json';raw=f.read_bytes();d=json.loads(raw);rows=d['rows'];groups={};keys=['dataset','family','partition','cut','rng_seed','prepared_sha256','training_sha256','validation_sha256','evaluator_sha256','vocabulary_sha256']
for row in rows:
 k=tuple(row.get(f) for f in keys);groups.setdefault(k,{})[row['arm']]=row
x={'schema':'jope.readonly-paired-diagnostic.v1','observed_at':time.time(),'selection_rule':'generation000066: latest complete quality report at capture; no candidate selection','source':str(f),'source_sha256':hashlib.sha256(raw).hexdigest(),'rows':len(rows),'identity_fields':keys,'groups':len(groups),'baseline_available':sum(r.get('baseline_available') is True for r in rows),'pure_prediction_statuses':dict(collections.Counter((r.get('pure_predicted_best') or {}).get('status','absent') for r in rows)),'partitions':{},'comparisons':[],'independent_evaluation':False,'promotion_evidence':False,'source_receipts_reverified':False}
for part in ['meta_train','meta_validation']:
 gs=[(k,g) for k,g in groups.items() if k[2]==part]
 p={'groups':len(gs),'by_opponent':{}}
 for opponent in ['generic','shuffled']:
  result={'win':0,'tie':0,'loss':0,'missing':0,'unequal_evaluations':0,'pairs':0};family={}
  for k,g in gs:
   a,b=g.get('neural'),g.get(opponent)
   if a is None or b is None or not all(type(z.get('best_final_gp')) in (int,float) and math.isfinite(z['best_final_gp']) and z['best_final_gp']>=0 for z in [a,b]):result['missing']+=1;continue
   av,bv=a['best_final_gp'],b['best_final_gp'];outcome='win' if av<bv else 'loss' if av>bv else 'tie';result[outcome]+=1;result['pairs']+=1
   result['unequal_evaluations']+=a['completed_evaluations']!=b['completed_evaluations']
   skill=1-av/bv if bv>0 else None
   if skill is not None:family.setdefault(k[1],[]).append(skill)
   x['comparisons'].append({'partition':part,'family':k[1],'cut':k[3],'rng_seed':k[4],'opponent':opponent,'neural_rmse':av,'opponent_rmse':bv,'outcome':outcome,'relative_rmse_gain':skill,'neural_evaluations':a['completed_evaluations'],'opponent_evaluations':b['completed_evaluations'],'neural_source':a['output'],'opponent_source':b['output']})
  result['family_macro_relative_rmse_gain']=statistics.fmean(statistics.fmean(v) for v in family.values()) if family else None;result['families']=len(family);p['by_opponent'][opponent]=result
 x['partitions'][part]=p
print(json.dumps(x))
