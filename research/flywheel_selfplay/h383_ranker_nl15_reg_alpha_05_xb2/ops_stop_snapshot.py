"""Read stopped local execution; no remote host, fit, prediction or capacity sampling."""
from pathlib import Path
import subprocess,json,hashlib,datetime
P=Path(__file__).parent;D=Path('/home/ubuntu/jope-data/h383-hwv');unit=subprocess.run(['systemctl','--user','is-active','h383-run'],capture_output=True,text=True).stdout.strip();assert unit not in ('active','activating','deactivating');assert (P/'unit-exit.json').exists()
files={}
for p in sorted(D.rglob('*')):
 if p.is_file():
  assert not p.is_symlink();h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  files[str(p.relative_to(D))]=dict(sha256=h.hexdigest(),bytes=p.stat().st_size)
(P/'stopped-file-inventory.json').write_text(json.dumps(files,indent=1)+'\n')
r=dict(status='PASS',host='xbabe2',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Stopped local output bytes inventoried; no cross-host transfer',files=len(files),bytes=sum(x['bytes'] for x in files.values()),inventory_sha256=hashlib.sha256((P/'stopped-file-inventory.json').read_bytes()).hexdigest(),execution=json.loads((P/'unit-exit.json').read_text()),fits=0,new_model_predictions=0);(P/'stop-verification.json').write_text(json.dumps(r,indent=2)+'\n')
raw=subprocess.check_output(['journalctl','--user','-u','h383-run.service','--no-pager','-o','json']);(P/'actual-unit-journal.jsonl').write_bytes(raw);rows=[json.loads(x) for x in raw.splitlines()];terminal=[r for r in rows if 'CPU_USAGE_NSEC' in r]
actual=dict(wave='H383',host='xbabe2',source='Terminal actual own user-unit CPU_USAGE_NSEC journal field; last execution bound to current unit-exit/launch',cpu_usage_nsec=int(terminal[-1]['CPU_USAGE_NSEC']) if terminal else None,cpu_seconds=int(terminal[-1]['CPU_USAGE_NSEC'])/1e9 if terminal else None,current_invocation=json.loads((P/'unit-exit.json').read_text())['INVOCATION_ID'],launch_invocation=json.loads((P/'run-launch.json').read_text())['invocation_id'],startup_invocation=json.loads((P/'startup-unit-exit.json').read_text())['INVOCATION_ID'],journal_sha256=hashlib.sha256(raw).hexdigest(),all_terminal_cpu_accounting=[dict(cpu_usage_nsec=r['CPU_USAGE_NSEC'],message=r.get('MESSAGE')) for r in terminal],note='Actual accounting only, no speed verdict; failed pre-fit startup retained separately');assert actual['current_invocation']==actual['launch_invocation'];(P/'actual-resources.json').write_text(json.dumps(actual,indent=2)+'\n');print(json.dumps(dict(stopped=True,files=len(files),cpu_seconds=actual['cpu_seconds'])))
