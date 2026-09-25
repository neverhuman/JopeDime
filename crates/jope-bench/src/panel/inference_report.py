"""Read-only v2 evidence reporting. Missing, corrupt and unsupported are explicit."""
import csv, hashlib, json, math, statistics, sys, struct
from pathlib import Path

SCHEMA='jope.inference-benchmark/v2'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verified(pin):
    try:return sha(pin['path'])==pin['sha256']
    except (OSError,KeyError,TypeError):return False

def qualified(plan,q):
    keys={'hyperion','foundation','target','context','policy_logits','reward_and_runtime','cache_full_prefix'}
    cases=q.get('cases',[])
    return (q.get('schema')==SCHEMA and q.get('passed') is True
        and q.get('plan_identity')==plan['experiment']['methods_sha256']
        and q.get('binary_sha256')==plan['binary']['sha256']
        and q.get('source_sha')==plan['experiment']['source_sha']
        and len(cases)==8 and len({c.get('case') for c in cases})==8
        and len(q.get('devices',[]))==3 and all('Cuda' in d for d in q['devices']) and 'Cuda' in q.get('policy_device','')
        and all(c.get('pass') is True and set(c.get('errors',{}))==keys and all(isinstance(x,(float,int)) and math.isfinite(x) and 0<=x<=3e-4 for x in c['errors'].values()) for c in cases))
def parity_verified(b):
    if not verified(b.get('qualification_plan',{})) or not verified(b.get('prediction',{})):return False
    q=read(Path(b['qualification_plan']['path']));actual=read(Path(b['prediction']['path']))
    expected=q.get('expected',{})
    if not verified(b.get('bundle',{})):return False
    bundle=read(Path(b['bundle']['path']))
    identity=hashlib.sha256(json.dumps(bundle,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if q.get('bundle_sha256')!=identity or q.get('backend')!=bundle.get('manifest',{}).get('backend'):return False
    try:return actual['kind']==expected['kind']=='values' and len(actual['data'])==len(expected['data']) and len(actual['data'])>0 and all(struct.pack('>d',a)==struct.pack('>d',e) for a,e in zip(actual['data'],expected['data']))
    except (KeyError,TypeError):return False
def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}
def distribution(values):
    values=sorted(values)
    return {'n':len(values),'median_seconds':statistics.median(values) if values else None,'p95_seconds':values[math.ceil(.95*len(values))-1] if values else None,'mean_seconds':statistics.mean(values) if values else None}
def summarize(plan,journal):
    q=read(journal/'qualification.json');qualification=qualified(plan,q)
    records=[];missing=[];backend_times={};failures=[]
    for d in plan['inputs']:
        for mode,count in [('warm',10),('cold',1),('warmup',2),('profile',1)]:
            for seed in range(count):
                p=journal/f"{mode}-{d['id']}-{seed}"/'receipt.json'
                if not p.exists():
                    missing.append({'dataset':d['id'],'mode':mode,'seed':seed,'status':'unexecuted_or_interrupted' if p.parent.exists() else 'unexecuted'});continue
                r=read(p)
                valid=(r.get('schema')==SCHEMA and r.get('plan_identity')==plan['experiment']['methods_sha256'] and r.get('binary_sha256')==plan['binary']['sha256'] and r.get('source_sha')==plan['experiment']['source_sha'] and r.get('dataset')==d['id'] and r.get('seed')==seed and r.get('mode')==mode and r.get('input_sha256')==d['raw']['sha256'] and r.get('selected_origin')=='neural_policy' and r.get('ranking_backend')=='candle_cuda_float32' and len(r.get('devices',[]))==3 and all('Cuda' in x for x in r['devices']) and 'Cuda' in r.get('policy_device','') and r.get('export',{}).get('validation_in_timer') is True)
                elapsed=r.get('elapsed_seconds')
                valid=valid and isinstance(elapsed,(int,float)) and math.isfinite(elapsed) and elapsed>=0
                export=r.get('export',{}); recipe=export.get('recipe',{})
                valid=valid and recipe.get('original_proposal')==r.get('selected_program') and verified(export.get('fitted_output',{}))
                completions=[]
                for b in export.get('backends',[]):
                    t=b.get('first_prediction_seconds')
                    good=(qualification and valid and b.get('status')=='completed' and isinstance(t,(int,float)) and math.isfinite(t) and t>0 and parity_verified(b) and b.get('qualification',{}).get('values',0)>0 and b.get('qualification',{}).get('maximum_absolute_error')==0 and all(verified(f) for f in b.get('files',{}).values()) and bool(b.get('files')) and verified(b.get('bundle',{})) and all(c.get('exit_code')==0 for c in b.get('commands',[])))
                    if good:
                        completions.append(b['backend'])
                        if mode=='warm':backend_times.setdefault((d['id'],b['backend']),[]).append(t)
                    elif b.get('status')!='unsupported':failures.append({'dataset':d['id'],'mode':mode,'seed':seed,'backend':b.get('backend'),'status':b.get('status'),'reason':b.get('reason','receipt or artifact validation failed')})
                r['accepted']=qualification and valid and r.get('success') is True and all(x in completions for x in ['python','rust'])
                if not r['accepted']:failures.append({'dataset':d['id'],'mode':mode,'seed':seed,'failure':r.get('failure','required path incomplete')})
                records.append(r)
    warm=[r for r in records if r['mode']=='warm' and r['accepted']]
    gp=[]
    for d in plan['inputs']:
        if d.get('gp') is None:continue
        for arm in ('neural','generic'):
            p=journal/f"gp-{d['id']}"/arm/'receipt.json'
            r=read(p) or {'dataset':d['id'],'arm':arm,'status':'unexecuted'}
            r['accepted']=qualification and r.get('source_sha')==plan['experiment']['source_sha'] and r.get('plan_identity')==plan['experiment']['methods_sha256'] and r.get('binary_sha256')==plan['binary']['sha256'] and r.get('input_sha256')==d['raw']['sha256'] and verified(r.get('outcome',{}).get('search',{})) and bool(r.get('outcome',{}).get('execution_inventory')) and all(verified(pin) for pin in r['outcome']['execution_inventory'])
            r['export_qualified']=r['accepted'] and any(parity_verified(b) for b in r.get('outcome',{}).get('export',{}).get('backends',[]))
            gp.append(r)
    prediction=[]
    for path in sorted(journal.glob('prediction-*/receipt.json')):
        r=read(path);d=next((d for d in plan['inputs'] if d['id']==r.get('dataset')),None)
        valid=qualification and d is not None and r.get('source_sha')==plan['experiment']['source_sha'] and r.get('plan_identity')==plan['experiment']['methods_sha256'] and r.get('binary_sha256')==plan['binary']['sha256'] and r.get('input_sha256')==d['raw']['sha256'] and verified(r.get('source_qualification',{})) and verified(r.get('prediction',{})) and r.get('parity')=='exact_bits'
        source=read(journal/f"warm-{r.get('dataset')}-0"/'receipt.json')
        entry=next((b for b in source.get('export',{}).get('backends',[]) if b.get('backend')==r.get('backend')), {})
        valid=valid and parity_verified(entry) and r.get('bundle_sha256')==entry.get('bundle',{}).get('sha256') and r.get('source_qualification')==entry.get('qualification_plan')
        if valid:
            prediction_plan=read(Path(r['source_qualification']['path']));a=read(Path(r['prediction']['path']))
            try:valid=len(a['data'])==r['batch_rows'] and all(struct.pack('>d',x)==struct.pack('>d',prediction_plan['expected']['data'][0]) for x in a['data'])
            except (KeyError,TypeError):valid=False
        # Timing and backend membership are part of receipt validity.
        valid=valid and r.get('backend') in ('python','rust','cpp17') and r.get('batch_rows') in (1,32,1024) and len(r.get('samples_seconds',[]))==10 and all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in r.get('samples_seconds',[]))
        r['accepted']=bool(valid and r.get('status')=='completed');prediction.append(r)
    seen={(r.get('dataset'),r.get('backend'),r.get('batch_rows')) for r in prediction}
    for d in plan['inputs']:
        source=read(journal/f"warm-{d['id']}-0"/'receipt.json')
        entries={r.get('backend'):r for r in source.get('export',{}).get('backends',[])}
        for backend in ('python','rust','cpp17'):
            for size in (1,32,1024):
                if (d['id'],backend,size) in seen:continue
                status='unsupported' if entries.get(backend,{}).get('status')=='unsupported' else 'interrupted' if (journal/f"prediction-{d['id']}-{backend}-{size}").exists() else 'unexecuted'
                prediction.append({'dataset':d['id'],'backend':backend,'batch_rows':size,'status':status,'accepted':False})
    summary={'schema':SCHEMA,'qualified':qualification,'qualification':q,'publication_headline':qualification and len(warm)==100,'completed_successfully':len(warm),'expected':100,'missing':missing,'failures':failures,'warm_cumulative_backend_milestones':[{'dataset':d,'backend':b,**distribution(v)} for (d,b),v in sorted(backend_times.items())],'cold_observations':[read(p) for p in sorted(journal.glob('cold-process-*.json'))],'full_gp':gp,'prediction':prediction,'window':read(journal/'window.json'),'warm_complete_request_distribution':distribution([r['elapsed_seconds'] for r in warm]),'boundary':'Fitting data to first prediction, including preprocessing, encoders, policy, fitting, emission and compilation. Backends emitted sequentially; later milestones include earlier backend work. No LLM speed comparison.'}
    return summary,records

def main():
    filename,expected,journal,output=sys.argv[1:];filename=Path(filename);journal=Path(journal);output=Path(output)
    if sha(filename)!=expected:raise ValueError('plan identity changed')
    plan=json.loads(filename.read_text())
    if plan['schema']!=SCHEMA:raise ValueError('receipt version unsupported')
    summary,records=summarize(plan,journal);output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (output/'requests.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','mode','seed','elapsed_seconds','accepted','failure'])
        for r in records:w.writerow([r.get(k,'') for k in ['dataset','mode','seed','elapsed_seconds','accepted','failure']])
    (output/'methodology.md').write_text('# GPU timing study\n\n'+summary['boundary']+'\n\nQualified: '+str(summary['qualified'])+'. Completed warm Python and Rust paths: '+str(summary['completed_successfully'])+'/100. Missing attempts are unexecuted, never zero latency. Cold observations are individual process wall times. Full GP uses five fitting-side OOF folds,96 candidate slots and120seconds per arm; exports occur after the winner freezes. Prediction API samples exclude process initialization, which is reported as a separate residual containing I/O and teardown.\n')
if __name__=='__main__':main()
