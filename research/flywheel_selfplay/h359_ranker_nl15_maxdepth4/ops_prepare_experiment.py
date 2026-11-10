from pathlib import Path
import subprocess,sys,json,datetime,hashlib,re
P=Path(__file__).parent;S=P/'source';D=Path('/home/ubuntu/jope-data/h359-hwv')
assert json.loads((P/'prepare-unit-exit.json').read_text())['EXIT_STATUS']=='0'
assert json.loads((P/'nl15-fetch-receipt.json').read_text())['status']=='EXACT_VERSION_GET_AND_ALL_MEMBERS_SHA_PASS'
assert json.loads((P/'prereg-published.json').read_text())['published']
steps=[]
for name,args in [('regress',['run_h359.py','regress']),('nativecheck',['check_base_native.py'])]:
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();cmd=[sys.executable,str(S/args[0]),*args[1:]];r=subprocess.run(cmd);record=dict(command=cmd,started_utc=start,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_status=r.returncode);steps.append(record)
 if name=='nativecheck':(P/'nativecheck-exit.json').write_text(json.dumps(dict(EXIT_CODE='exited' if r.returncode>=0 else 'killed',EXIT_STATUS=str(r.returncode),scope='Actual native-check subprocess exit, not systemd receipt',**record),indent=2)+'\n')
 if r.returncode:raise SystemExit(r.returncode)
cmd=[sys.executable,'-m','unittest','discover','-s',str(S),'-p','test_*.py','-v'];start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (P/'tests-final.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
log=(P/'tests-final.log').read_text();match=re.search(r'Ran (\d+) tests in',log);assert match and r.returncode==0 and log.rstrip().endswith('OK')
tests=dict(command=cmd,started_utc=start,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_status=r.returncode,tests_collected=int(match.group(1)),tests_executed=int(match.group(1)),failed=0,skipped=0,synthetic_optimizer_fits=0,log_sha256=hashlib.sha256(log.encode()).hexdigest())
(P/'tests-receipt.json').write_text(json.dumps(tests,indent=2)+'\n');assert tests['tests_executed']==65
assert json.loads((D/'regression.json').read_text())['status']=='PASS' and json.loads((D/'nl15_regression.json').read_text())['status']=='PASS'
(P/'base-prepared.json').write_text(json.dumps(dict(status='PASS',steps=steps,tests=tests,BASE_refits=0,NL15_refits=0,new_comparator_model_predictions=0),indent=2)+'\n')
print(json.dumps(dict(prepared='PASS',tests=65,BASE_regression='EXACT',NL15_regression='EXACT',prereg=json.loads((P/'prereg-published.json').read_text())['head'])),flush=True)
