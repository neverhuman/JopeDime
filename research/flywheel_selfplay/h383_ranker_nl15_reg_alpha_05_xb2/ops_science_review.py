from pathlib import Path
import ast,json,hashlib,datetime
P=Path(__file__).parent;S=P/'source';T=P/'template'
def norm(t,host=False):
 if host:t=t.replace('h383-run.service','h376-x1-run.service').replace('h383-prepare.service','h376-x1-prepare.service')
 t=t.replace('h383-hwv','h376-x1').replace('hwv-xb2-20261105T170500Z','solhwv-x1-20261105T145600Z').replace('383191','376191').replace('H348-H383','H348-H376').replace('H348–H383','H348–H376').replace('H383','H376').replace('h383','h376').replace('2026,11,5,20,45','2026,11,5,18,15').replace('2026-11-05T20:45:00','2026-11-05T18:15:00').replace('2026-11-05T21:00:00','2026-11-05T18:30:00')
 return t.replace("nodename.split('.')[0]=='xbabe2'","nodename.split('.')[0]=='xbabe1'") if host else t
def dump(t):return ast.dump(ast.parse(t),include_attributes=False)
groups=[]
for n in ['run_h376.py','runtime_h376.py','evaluate_h376.py','depth_patch.py','leaves_patch.py','native_audit.py','dataset_audit.py','veto_identity.py','tree_structure.py','nl15_reference.py','input_preflight.py','archive_h376.py','archive_relay.py','check_base_native.py','check_nl15_native.py','forecast_h376.py']:
 new=n.replace('376','383');assert dump((T/n).read_text())==dump(norm((S/new).read_text(),host=n=='run_h376.py')),new;groups.append(dict(reference=n,candidate=new,entire_AST_exact_after_declared_wave_path_host_RNG_window_normalization=True))
rec=dict(status='PASS',name='FROZEN-SCIENCE-EQUIVALENCE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference_commit='48e39890be89978e2793c1240aa0d1a2625e9438',reference_prereg='8383d031176b2d4708893c2e4caa6d41f6766e4a',groups=groups,declared_operational_deltas=['xbabe2 rehost only; frozenL1A05science exact','mandatory read-only h377-run inactive gate,18:50cutoff beforeONEcapacityinterval','oneunion primary27..37 thenfallback12/13/14 +SMTs','CPU85avg60/load108three60/Mem20/scratch2GB; no x1rootguard','fixed20:45compute21:00close; RNG383191'],same_agent_review=True)
(P/'science-equivalence.json').write_text(json.dumps(rec,indent=2)+'\n');files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in S.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json')};(P/'runtime-source-manifest.json').write_text(json.dumps(files,indent=2)+'\n');print(json.dumps(dict(status='PASS',groups=len(groups),frozen_files=len(files))))
