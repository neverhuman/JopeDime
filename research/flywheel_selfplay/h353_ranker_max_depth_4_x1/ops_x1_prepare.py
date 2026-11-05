"""Verify staged copies and BASE entirely on xbabe1, in a process ending before fits."""
from pathlib import Path
import os,json,hashlib,subprocess,sys,threading,datetime
import numpy,lightgbm,sklearn
P=Path(__file__).parent;S=P/'source';D=P.parent;sys.path.insert(0,str(S));import guard_h353 as guard
assert os.uname().nodename.split('.')[0]=='xbabe1' and (numpy.__version__,lightgbm.__version__,sklearn.__version__)==('1.26.4','4.6.0','1.7.2')
threading.Thread(target=lambda:guard.monitor('h353-x1-prepare.service'),daemon=True).start()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
source=json.loads((P/'runtime-source-manifest.json').read_text());manifest=json.loads((S/'manifest.json').read_text())
for n,h in source.items():assert sha(S/n)==h,n
for n,h in manifest['sources'].items():assert sha(P/'vendor'/n)==h,n
inputs=json.loads((P/'staged-input-inventory.json').read_text())
for n,r in inputs['files'].items():assert sha(D/'inputs'/n)==r['sha256'],n
fetch=json.loads((P/'base-fetch-receipt.json').read_text());count=0
for n,r in fetch['h316']['members'].items():
 p=D/('base_native' if '/refit/' in n else 'base')/n[4:];assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],n;count+=1
assert sha(D/'base/h315_picks.npz')==fetch['h315']['sha256']
(P/'stage-verification.json').write_text(json.dumps(dict(status='PASS',host='xbabe1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),BASE_selected_archive_members=count,BASE_H315_sha_verified=True,input_files=len(inputs['files']),source_files=len(source),vendor_files=len(manifest['sources']),all_staged_SHA256_verified=True,credential_transfer=False,scratch_bytes=sum(p.stat().st_size for p in D.rglob('*') if p.is_file())),indent=2)+'\n')
with (P/'tests-final.log').open('w') as log:r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(S),'-v'],stdout=log,stderr=subprocess.STDOUT)
text=(P/'tests-final.log').read_text();assert r.returncode==0 and 'Ran 55 tests' in text and text.endswith('OK\n')
(P/'tests-receipt.json').write_text(json.dumps(dict(status='PASS',host='xbabe1',tests_collected=55,tests_executed=55,failed=0,skipped=0,exit_status=r.returncode,synthetic_model_fits=0,log_sha256=sha(P/'tests-final.log'),source_sha256=source),indent=2)+'\n')
steps=[]
for args in [['input_preflight.py'],['run_h353.py','regress'],['check_base_native.py']]:
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();r=subprocess.run([sys.executable,str(S/args[0]),*args[1:]]);steps.append(dict(command=args,exit_status=r.returncode,started_utc=start,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 if args[0]=='check_base_native.py':(P/'nativecheck-exit.json').write_text(json.dumps(dict(EXIT_STATUS=str(r.returncode),observed_subprocess_exit=True)))
 if r.returncode:raise SystemExit(r.returncode)
(P/'base-prepared.json').write_text(json.dumps(dict(status='PASS',host='xbabe1',steps=steps),indent=2)+'\n')
