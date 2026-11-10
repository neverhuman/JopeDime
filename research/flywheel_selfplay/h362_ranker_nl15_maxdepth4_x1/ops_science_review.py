from pathlib import Path
import ast,json,hashlib,datetime
P=Path(__file__).parent;S=P/'source';T=P/'template'
def norm(t):return t.replace('h362-x1','h359-hwv').replace('362191','359191').replace('H362','H359').replace('h362','h359').replace("nodename.split('.')[0]=='xbabe1'","nodename.split('.')[0]=='xbabe2'").replace('2026,11,5,14,30','2026,11,5,13,15').replace('2026-11-05T14:30:00','2026-11-05T13:15:00').replace('2026-11-05T14:45:00','2026-11-05T13:30:00')
def dump(t):return ast.dump(ast.parse(t),include_attributes=False)
def funcs(t):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)}
groups=[]
for old,new in [('run_h359.py','run_h362.py'),('runtime_h359.py','runtime_h362.py'),('evaluate_h359.py','evaluate_h362.py'),('native_audit.py','native_audit.py'),('depth_patch.py','depth_patch.py'),('leaves_patch.py','leaves_patch.py'),('veto_identity.py','veto_identity.py'),('forecast_h359.py','forecast_h362.py')]:
 assert dump((T/old).read_text())==dump(norm((S/new).read_text())),new
 groups.append(dict(reference=old,candidate=new,entire_AST_exact_after_declared_normalization=True))
a=funcs((T/'archive_h359.py').read_text());b=funcs(norm((S/'archive_h362.py').read_text()))
for name in ['compress_inner','restore_inner','remove_restored_inner']:assert a[name]==b[name],name
groups.append(dict(reference='archive_h359.py',candidate='archive_h362.py',scientific_model_review_and_verified_reclamation_functions_exact=True))
rec=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference_commit='816c27db36ea9cb56d81120eb6dcfe5e0cf5daee',groups=groups,declared_deltas=['wave/path/RNG362191/window/host adaptation','qualifiedH353 credential-free host transport; corrected owner resource guard','TRAIN copy default path rather than shared live inputs on xbabe1','No scientific/selection/decision/metric/model-parameter changes relativeH359'],same_agent_review=True)
(P/'science-equivalence.json').write_text(json.dumps(rec,indent=2)+'\n');files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in S.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json')};(P/'runtime-source-manifest.json').write_text(json.dumps(files,indent=2)+'\n');print(json.dumps(dict(status='PASS',groups=len(groups),frozen_files=len(files))))
