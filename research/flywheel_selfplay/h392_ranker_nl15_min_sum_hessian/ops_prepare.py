"""Actual xbabe2-only full preparation service; no admission or optimizer fit."""
from pathlib import Path
import sys,json,os,hashlib,subprocess,datetime,threading
P=Path(__file__).parent;S=P/'source';D=Path('/home/ubuntu/jope-data/h392-hwv');sys.path.insert(0,str(S));import guard_h392 as guard
assert os.uname().nodename.split('.')[0]=='xbabe2' and os.getpriority(os.PRIO_PROCESS,0)==19
threading.Thread(target=lambda:guard.monitor('h392-prepare.service'),daemon=True).start()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
frozen=json.loads((P/'runtime-source-manifest.json').read_text());manifest=json.loads((S/'manifest.json').read_text())
for n,h in frozen.items():assert sha(S/n)==h,n
for n,h in manifest['sources'].items():assert sha(P/'vendor'/n)==h,n
inputs=json.loads((P/'staged-input-inventory.json').read_text())
for n,v in inputs['files'].items():assert sha(D/'inputs'/n)==v['sha256'],n
fetch=json.loads((P/'base-fetch-receipt.json').read_text())
for n,v in fetch['h316']['members'].items():
 p=D/('base_native' if '/refit/' in n else 'base')/n[4:];assert sha(p)==v['sha256'] and p.stat().st_size==v['bytes'],n
assert sha(D/'base/h315_picks.npz')==fetch['h315']['sha256']
nl=json.loads((P/'nl15-fetch-receipt.json').read_text())
for n,v in nl['saved_members'].items():assert sha(D/'nl15'/n)==v['sha256'] and (D/'nl15'/n).stat().st_size==v['bytes'],n
assert sha(D/'nl15/recovery-manifest.json')==nl['manifest']['sha256']
(P/'stage-verification.json').write_text(json.dumps(dict(status='PASS',host='xbabe2',all_staged_SHA256_verified=True,source_files=len(frozen),vendor_files=len(manifest['sources']),input_files=len(inputs['files']),BASE_selected_members=len(fetch['h316']['members']),NL15_selected_members=len(nl['saved_members']),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
with (P/'tests-final.log').open('w') as log:r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(S),'-v'],stdout=log,stderr=subprocess.STDOUT)
text=(P/'tests-final.log').read_text();assert r.returncode==0 and ('Ran '+str(manifest['tests_required'])+' tests') in text and text.endswith('OK\n')
(P/'tests-receipt.json').write_text(json.dumps(dict(status='PASS',host='xbabe2',tests_collected=manifest['tests_required'],tests_executed=manifest['tests_required'],failed=0,skipped=0,exit_status=0,synthetic_model_fits=0,log_sha256=sha(P/'tests-final.log'),source_sha256=frozen),indent=2)+'\n')
steps=[]
for args in [['input_preflight.py'],['run_h392.py','regress'],['check_base_native.py'],['check_nl15_native.py']]:
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();r=subprocess.run([sys.executable,str(S/args[0]),*args[1:]]);steps.append(dict(command=args,started_utc=start,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_status=r.returncode))
 if args[0]=='check_base_native.py':(P/'nativecheck-exit.json').write_text(json.dumps(dict(EXIT_STATUS=str(r.returncode),observed_subprocess_exit=True))+'\n')
 if r.returncode:raise SystemExit(r.returncode)
for n,h in frozen.items():assert sha(S/n)==h,n
assert json.loads((D/'nl15_regression.json').read_text())['status']=='PASS'
(P/'base-prepared.json').write_text(json.dumps(dict(status='PASS',host='xbabe2',steps=steps,optimizer_fits=0,new_comparator_predictions=0),indent=2)+'\n');print('H392 full pre-fit source/input/BASE/NL15/native/tests PASS; capacity probes0',flush=True)
