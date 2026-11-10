"""Verify every xbabe1 owned output byte after workers stop; no fitting or scoring."""
from pathlib import Path
import subprocess,json,hashlib,datetime
P=Path(__file__).parent;D=Path('/home/ubuntu/jope-data/h368-x1');REMOTE='/home/ubuntu/jope-data/h368-x1'
assert subprocess.run(['systemctl','--user','is-active','h368-x2-relay'],capture_output=True,text=True).stdout.strip() not in ('active','activating'),'Wait for own courier to finish before snapshot'
code='''from pathlib import Path
import subprocess,json,hashlib,datetime
D=Path('/home/ubuntu/jope-data/h368-x1');P=D/'control'
units={u:subprocess.run(['systemctl','--user','is-active',u],capture_output=True,text=True).stdout.strip() for u in ['h368-x1-prepare','h368-x1-run']};assert all(x not in ('active','activating') for x in units.values())
assert (P/'unit-exit.json').exists()
files={}
for p in sorted(D.rglob('*')):
 if p.is_file() and p!=P/'output-inventory.json':
  assert not p.is_symlink();h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
  files[str(p.relative_to(D))]=dict(sha256=h.hexdigest(),bytes=p.stat().st_size)
r=dict(status='PASS',host='xbabe1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),own_units=units,execution=json.loads((P/'unit-exit.json').read_text()),files=files,total_bytes=sum(x['bytes'] for x in files.values()))
p=P/'output-inventory.json';p.write_text(json.dumps(r,indent=1)+'\\n');print(json.dumps(dict(files=len(files),bytes=r['total_bytes'],output_inventory_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
'''
r=json.loads(subprocess.check_output(['ssh','-T','-o','BatchMode=yes','xbabe1','nice -n 19 taskset -c 35 python3 -'],input=code.encode()))
subprocess.run(['rsync','-a','-e','ssh -T -o BatchMode=yes','xbabe1:'+REMOTE+'/',str(D)+'/'],check=True)
inv=D/'control/output-inventory.json';assert hashlib.sha256(inv.read_bytes()).hexdigest()==r['output_inventory_sha256'];entries=json.loads(inv.read_text())
for name,want in entries['files'].items():
 path=D/name;h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 assert h.hexdigest()==want['sha256'] and path.stat().st_size==want['bytes'],name
# Preserve xbabe2 prep proof separately; install the actual xbabe1 execution receipts for reporting.
import shutil
if (P/'input_identity_validation.json').exists():shutil.copy2(P/'input_identity_validation.json',P/'xbabe2-input-identity-validation.json')
for name in ['unit-exit.json','prepare-unit-exit.json','base-native-config.json','nl15-native-config.json','nl15-regression.json','input_identity_validation.json','base-prepared.json','nativecheck-exit.json','stage-verification.json','tests-receipt.json','tests-final.log','capacity.json','initial-resource.json','controller-affinity-binding.json','runtime-forecast-at-launch.json','first-cell-forecast.json','forecast-watch-receipt.json','run-launch.json','prepare-launch.json']:
 src=D/'control'/name
 if src.exists():shutil.copy2(src,P/name)
record=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_host='xbabe1',local_host='xbabe2',all_transferred_SHA256_verified=True,files=len(entries['files']),total_bytes=entries['total_bytes'],remote_inventory_sha256=r['output_inventory_sha256'],remote_units=entries['own_units'],execution=entries['execution'],model_fits=0,model_scores=0,credential_transfer=False)
(P/'transfer-verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
