"""Return only owned xbabe1 STOP metadata; no TRAIN/model staging occurred there."""
from pathlib import Path
import subprocess,json,hashlib,datetime,shlex
P=Path(__file__).parent;D=Path('/home/ubuntu/jope-data/h376-x1')
code='''from pathlib import Path
import json,hashlib,subprocess,datetime
D=Path('/home/ubuntu/jope-data/h376-x1');assert {str(p.relative_to(D)) for p in D.rglob('*') if p.is_file()}=={'STOP.json'}
units={n:subprocess.run(['systemctl','--user','is-active',n],capture_output=True,text=True).stdout.strip() for n in ['h376-x1-prepare','h376-x1-run']};assert all(v not in ('active','activating','deactivating') for v in units.values())
p=D/'STOP.json';files={'STOP.json':dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)}
r=dict(status='PASS',scope='Only resource-stop metadata bytes; no TRAIN/checkpoint/code staged',host='xbabe1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,total_bytes=p.stat().st_size,own_units=units,scientific_compute_units_launched=0,capacity_probes=0,optimizer_fits=0,outer_scores=0,execution=dict(status='NOT_RUN_RESOURCE_GUARD',EXIT_CODE=None,EXIT_STATUS=None,SERVICE_RESULT=None))
q=D/'control/output-inventory.json';q.parent.mkdir();q.write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(dict(files=files,inventory_sha256=hashlib.sha256(q.read_bytes()).hexdigest())))
'''
r=json.loads(subprocess.check_output(['ssh','-T','-o','BatchMode=yes','xbabe1','nice -n 19 taskset -c 35 python3 -c '+shlex.quote(code)]))
subprocess.run(['rsync','-a','-e','ssh -T -o BatchMode=yes','xbabe1:'+str(D)+'/',str(D)+'/'],check=True)
for n,w in r['files'].items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==w['sha256'] and (D/n).stat().st_size==w['bytes']
assert hashlib.sha256((D/'control/output-inventory.json').read_bytes()).hexdigest()==r['inventory_sha256']
(P/'transfer-verification.json').write_text(json.dumps(dict(status='PASS',scope='STOP metadata copy only; TRAIN/checkpoint/source transfer to xbabe1 NOT_RUN_RESOURCE_GUARD',files=2,all_transferred_SHA256_verified=True,remote_inventory_sha256=r['inventory_sha256'],optimizer_fits=0,outer_scores=0,credential_transfer=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n');print('Only two xbabe1 metadata files copied and SHA verified; no TRAIN/model staging')
