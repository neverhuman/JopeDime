"""xbabe2-only prepared source/input proof; zero treatment fits or predictions."""
from pathlib import Path
import sys,os,json,hashlib,subprocess,datetime
P=Path(__file__).parent;S=P/'source';D=Path('/home/ubuntu/jope-data/h376-x1');sys.path.insert(0,str(S));assert os.uname().nodename.split('.')[0]=='xbabe2'
frozen=json.loads((P/'runtime-source-manifest.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in frozen.items():assert sha(S/n)==h,n
with (P/'tests-final.log').open('w') as log:r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(S),'-v'],stdout=log,stderr=subprocess.STDOUT)
text=(P/'tests-final.log').read_text();assert r.returncode==0 and 'Ran 84 tests' in text and text.endswith('OK\n')
(P/'tests-receipt.json').write_text(json.dumps(dict(status='PASS',host='xbabe2',tests_collected=84,tests_executed=84,failed=0,skipped=0,exit_status=0,synthetic_model_fits=0,Dataset_only_bin_fixture=True,log_sha256=sha(P/'tests-final.log'),source_sha256=frozen,xbabe1_tests='NOT_RUN_RESOURCE_GUARD'),indent=2)+'\n')
import run_h376 as runner
r303=runner.initialize(require_prereg=False);runner.full_packets()
import evaluate_h376 as ev
ev.regress(r303,D)
from nl15_reference import regress
regress(r303,D,runner.PARTS,runner.SEEDS,runner.outer_fold,runner.durable)
subprocess.run([sys.executable,str(S/'check_base_native.py')],check=True)
for n,h in frozen.items():assert sha(S/n)==h,n
record=dict(status='PASS',host='xbabe2',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='CPU preparation only; archived arrays/native/input proof, no new predictions/refits/treatment',BASE_panels=9,BASE_pick_lock_hashes=36,BASE_partition_medians_exact=True,NL15_panels=9,NL15_locks_scores=36,NL15_raw_family_arrays_exact=True,H352_saved_ranker_headers=972,H352_actual_ranker_fit_headers=29484,H352_native_lambda_l1=[0],H352_native_max_bin=[255],optimizer_fits=0,new_model_predictions=0,xbabe1_staging_tests_regressions='NOT_RUN_RESOURCE_GUARD',frozen_source_sha256=frozen)
(P/'cpu-prepared.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='frozen_source_sha256'}),flush=True)
