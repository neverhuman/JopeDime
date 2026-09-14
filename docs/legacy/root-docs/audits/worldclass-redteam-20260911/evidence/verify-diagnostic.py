from pathlib import Path
import json,hashlib,math,collections,statistics,time
root=Path('/home/ubuntu/jope-dime/parallel');f=root/'runs/continuous-learning-v2-20260909/generation-000066/quality-report.json';raw=f.read_bytes();d=json.loads(raw);out={'observed_at':time.time(),'source':str(f),'source_sha256':hashlib.sha256(raw).hexdigest(),'rows':len(d['rows']),'summary_receipts_matched':0,'mismatches':[],'origin_counts':{},'later_vs_teacher':{},'search_cost':{},'source_observation_ledgers_read':False,'native_admission_verified':False,'head_profile_unavailable':True};samples=[]
for row in d['rows']:
 p=root/row['output']/'search.json';raw=p.read_bytes();s=json.loads(raw)
 if hashlib.sha256(raw).hexdigest()!=row['search_sha256'] or s.get('best_validation_rmse')!=row['best_final_gp'] or s.get('rng_seed')!=row['rng_seed']:out['mismatches'].append(str(p));continue
 out['summary_receipts_matched']+=1
 samples.append({'arm':row['arm'],'family':row['family'],'partition':row['partition'],'evaluations':s['completed_evaluations'],'wall_seconds':s.get('wall_seconds'),'best_origin':row.get('best_final_origin')})
for arm in ['neural','generic','shuffled']:
 rows=[r for r in d['rows'] if r['arm']==arm];out['origin_counts'][arm]=dict(collections.Counter(r['best_final_origin'] for r in rows));pairs=[(r['best_final_gp'],r['best_retained_teacher']) for r in rows if type(r.get('best_retained_teacher')) in (int,float) and type(r.get('best_final_gp')) in (int,float)]
 out['later_vs_teacher'][arm]={'pairs':len(pairs),'final_better':sum(a<b for a,b in pairs),'final_equal':sum(a==b for a,b in pairs),'final_worse':sum(a>b for a,b in pairs)}
 ss=[s for s in samples if s['arm']==arm];sec=[s['wall_seconds'] for s in ss if type(s['wall_seconds']) in (int,float)];ev=[s['evaluations'] for s in ss]
 out['search_cost'][arm]={'searches':len(ss),'completed_evaluations':sum(ev),'evaluations_median':statistics.median(ev),'at_1000_limit':sum(v==1000 for v in ev),'sum_search_wall_seconds':sum(sec),'wall_median':statistics.median(sec),'wall_max':max(sec),'timing_scope':'search process active elapsed; not instrumented numerical CPU'}
out['samples']=samples
print(json.dumps(out))
