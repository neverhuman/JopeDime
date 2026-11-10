from pathlib import Path
import ast,json,hashlib,datetime
P=Path(__file__).parent;S=P/'source';T=P/'template'
def norm(t):return t.replace('h368-x1','h362-x1').replace('368191','362191').replace('H348–H368','H348–H362').replace('MB63','NL15MD4').replace('mb63','nl15md4').replace('H368','H362').replace('h368','h362').replace('2026,11,5,16,30','2026,11,5,14,30').replace('2026-11-05T16:30:00','2026-11-05T14:30:00').replace('2026-11-05T16:45:00','2026-11-05T14:45:00').replace('20261105T133000Z','20261105T110000Z')
def dump(t):return ast.dump(ast.parse(t),include_attributes=False)
def funcs(t):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(t).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
groups=[]
for n in ['runtime_h362.py','leaves_patch.py','veto_identity.py','forecast_h362.py','guard_h362.py','resource_probe.py','archive_h362.py','archive_relay.py','nl15_reference.py','input_preflight.py']:
 new=n.replace('362','368');assert dump((T/n).read_text())==dump(norm((S/new).read_text())),new;groups.append(dict(file=new,entire_AST_exact_after_wave_path_RNG_window_normalization=True))
for n,allowed in [('run_h362.py',{'ctor_audit','fit_native_audit','native_audit_summary','audit_summary','main'}),('evaluate_h362.py',{'decide','decide_with_secondary','evaluate'}),('native_audit.py',{'compare_native','compare_h352_defaults'})]:
 a=funcs((T/n).read_text());b=funcs(norm((S/n.replace('362','368')).read_text()));assert a.keys()==b.keys();unchanged=[]
 for name in a:
  if name not in allowed:assert a[name]==b[name],(n,name);unchanged.append(name)
 groups.append(dict(file=n.replace('362','368'),exact_unchanged_functions=unchanged,reviewed_delta_functions=sorted(allowed)))
rec=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference_commit='e976935dc7a4dbdbaa6047bc659966c7c18e0480',groups=groups,declared_deltas=['wave/path/RNG368191/window','H352 NL15/depth3 with sole added ranker max_bin63 keyword; inverse removes keyword restoring two entire H352 ASTs','fresh-Dataset construction/native parameter audit in every actual fit; archived max_bin255 spec gate','PRIMARY paired NL15/four gates; secondary BASE descriptive only; comparator-distinguishing fixture','correct legacy forecast-miss target to owned h368-x1-run before fits'],same_agent_review=True)
(P/'science-equivalence.json').write_text(json.dumps(rec,indent=2)+'\n');files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in S.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json')};(P/'runtime-source-manifest.json').write_text(json.dumps(files,indent=2)+'\n');print(json.dumps(dict(status='PASS',groups=len(groups),frozen_files=len(files))))
