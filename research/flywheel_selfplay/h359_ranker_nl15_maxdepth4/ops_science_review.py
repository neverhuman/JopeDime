from pathlib import Path
import ast,json,hashlib,datetime
P=Path(__file__).parent;S=P/'source';T=P/'template'
def norm(t):return t.replace('h359-hwv','h352-jrb01').replace('NL15MD4','NL15').replace('359191','352191').replace('H359','H352').replace('h359','h352')
def funcs(t):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(t).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
groups=[]
for old,new,names in [('run_h352.py','run_h359.py',['sha','durable','outer_fold','member_scope','full_packets','task','score','verify_locks']),('runtime_h352.py','runtime_h359.py',['sha','durable','fold','packets','score_models']),('evaluate_h352.py','evaluate_h359.py',['family_d','regress','drop_top','paired_drop_top','decide','jaccard'])]:
 a=funcs((T/old).read_text());b=funcs(norm((S/new).read_text()))
 for n in names:assert a[n]==b[n],(new,n)
 groups.append(dict(reference=old,candidate=new,functions=names,exact_AST=True))
for name in ['leaves_patch.py','veto_identity.py']:
 assert ast.dump(ast.parse((T/name).read_text()),include_attributes=False)==ast.dump(ast.parse(norm((S/name).read_text())),include_attributes=False),name
 groups.append(dict(reference=name,candidate=name,full_AST=True))
rec=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),groups=groups,declared_deltas=['Only existing max_depth3->4 relative to exact H352 NL15 function ASTs; two reverse proofs','Host/path/core/window adaptations and TRAIN-only input copies','Explicit read-only archived NL15 regression/comparator, dual decision reporting','Read-only native H352 full defaults comparison and actual per-tree telemetry','Verified compressed duplicate-cache reclamation after S3 whole/member proof'],source_commit='02dfee2f7e0b8543324494dd4f99d074cd6b93b6',same_agent_review=True)
(P/'science-equivalence.json').write_text(json.dumps(rec,indent=2)+'\n');files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in S.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json')};(P/'runtime-source-manifest.json').write_text(json.dumps(files,indent=2)+'\n');print(json.dumps(dict(status='PASS',groups=len(groups),frozen_files=len(files))))
