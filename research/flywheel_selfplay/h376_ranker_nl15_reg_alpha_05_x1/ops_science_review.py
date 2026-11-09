from pathlib import Path
import ast,json,hashlib,datetime
P=Path(__file__).parent;S=P/'source';T=P/'template'
def norm(t):return t.replace('h376-x1','h368-x1').replace('376191','368191').replace('H348–H376','H348–H368').replace('H348-H376','H348-H368').replace('L1A05','MB63').replace('l1a05','mb63').replace('H376','H368').replace('h376','h368').replace('2026,11,5,18,15','2026,11,5,16,30').replace('2026-11-05T18:15:00','2026-11-05T16:30:00').replace('2026-11-05T18:30:00','2026-11-05T16:45:00').replace('20261105T145600Z','20261105T133000Z')
def dump(t):return ast.dump(ast.parse(t),include_attributes=False)
def funcs(t):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(t).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
groups=[]
for n in ['runtime_h368.py','leaves_patch.py','veto_identity.py','forecast_h368.py','guard_h368.py','resource_probe.py','archive_h368.py','archive_relay.py','nl15_reference.py','input_preflight.py','controller_h368.py']:
 new=n.replace('368','376');assert dump((T/n).read_text())==dump(norm((S/new).read_text())),new;groups.append(dict(file=new,entire_AST_exact_after_wave_path_RNG_window_normalization=True))
for n,allowed in [('run_h368.py',{'ctor_audit','fit_native_audit','native_audit_summary','audit_summary'}),('evaluate_h368.py',{'evaluate'}),('native_audit.py',{'compare_native','compare_h352_defaults'}),('dataset_audit.py',{'validate_rows','audit_fit'})]:
 a=funcs((T/n).read_text());b=funcs(norm((S/n.replace('368','376')).read_text()));assert a.keys()==b.keys();unchanged=[]
 for name in a:
  if name not in allowed:assert a[name]==b[name],(n,name);unchanged.append(name)
 groups.append(dict(file=n.replace('368','376'),exact_unchanged_functions=unchanged,reviewed_delta_functions=sorted(allowed)))
rec=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference_commit='a3d72923dc0dc530904eb0e9107751e1ed8c7be5',groups=groups,declared_deltas=['wave/path/RNG376191/window','Restore H352 implicit max_bin255, add only ranker reg_alpha=.5/native lambda_l1; inverse removes reg_alpha restoring two complete H352 ASTs','Native/dataset checks require H352 max_bin255; ranker lambda_l1=.5/veto own1; archived H352 zeroL1 spec gate','Unchanged primaryNL15/four gates and secondaryBASE metrics; reporting arm/intervention text only'],qualification='Source/fixture proof only; x1resourceguard prevented trial',same_agent_review=True)
(P/'science-equivalence.json').write_text(json.dumps(rec,indent=2)+'\n');files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in S.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json')};(P/'runtime-source-manifest.json').write_text(json.dumps(files,indent=2)+'\n');print(json.dumps(dict(status='PASS',groups=len(groups),frozen_files=len(files))))
