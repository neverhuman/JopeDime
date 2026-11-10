"""Inventory only this stopped scientific execution, without new fits or scores."""
from pathlib import Path
import datetime,hashlib,json,subprocess
P=Path(__file__).parent;D=Path('/home/ubuntu/jope-data/h392-hwv')
def main():
 state=subprocess.run(['systemctl','--user','is-active','h392-run'],capture_output=True,text=True).stdout.strip();assert state not in ('active','activating','deactivating')
 execution=json.loads((P/'unit-exit.json').read_text());launch=json.loads((P/'run-launch.json').read_text());assert execution['INVOCATION_ID']==launch['invocation_id']
 files={}
 for path in sorted(D.rglob('*')):
  if not path.is_file():continue
  assert not path.is_symlink();h=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
  files[str(path.relative_to(D))]=dict(bytes=path.stat().st_size,sha256=h.hexdigest())
 (P/'stopped-file-inventory.json').write_text(json.dumps(files,indent=1)+'\n');proof=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host='xbabe2',scope='Stopped local bytes inventoried; no cross-host transfer',files=len(files),bytes=sum(v['bytes'] for v in files.values()),inventory_sha256=hashlib.sha256((P/'stopped-file-inventory.json').read_bytes()).hexdigest(),execution=execution,new_fits=0,new_predictions=0);(P/'stop-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
 raw=subprocess.check_output(['journalctl','--user','-u','h392-run.service','--no-pager','-o','json']);(P/'actual-unit-journal.jsonl').write_bytes(raw);rows=[json.loads(line) for line in raw.splitlines()];terminal=[r for r in rows if 'CPU_USAGE_NSEC' in r];cpu=int(terminal[-1]['CPU_USAGE_NSEC']) if terminal else None
 accounting=dict(wave='H392',host='xbabe2',source='Actual terminal own user-unit CPU_USAGE_NSEC; one scientific invocation, raw journal retained',cpu_usage_nsec=cpu,cpu_seconds=cpu/1e9 if cpu is not None else None,current_invocation=execution['INVOCATION_ID'],launch_invocation=launch['invocation_id'],journal_sha256=hashlib.sha256(raw).hexdigest(),all_terminal_cpu_accounting=[dict(cpu_usage_nsec=r['CPU_USAGE_NSEC'],message=r.get('MESSAGE')) for r in terminal],note='Accounting only; no speed conclusion');(P/'actual-resources.json').write_text(json.dumps(accounting,indent=2)+'\n');print(json.dumps(dict(stopped=True,files=len(files),cpu_seconds=accounting['cpu_seconds'])))
if __name__=='__main__':main()
