from pathlib import Path
import subprocess,json,datetime,math,hashlib,os
P=Path(__file__).parent;S=P/'source';D=Path('/home/ubuntu/jope-data/h359-hwv');unit='h359-run'
assert os.uname().nodename.split('.')[0]=='xbabe2'
assert not (P/'capacity-interval-mpstat.json').exists() and not (P/'run-launch.json').exists(),'Never duplicate launch/probe'
for u in ['h359-reference-prepare','h359-input-prepare','h359-experiment-prepare',unit]:assert subprocess.run(['systemctl','--user','is-active',u],capture_output=True,text=True).stdout.strip() not in ('active','activating','deactivating'),u
assert json.loads((P/'experiment-prepare-unit-exit.json').read_text())['EXIT_STATUS']=='0'
assert json.loads((P/'tests-receipt.json').read_text())['tests_executed']==65
assert json.loads((P/'science-equivalence.json').read_text())['status']=='PASS'
for n,h in json.loads((P/'runtime-source-manifest.json').read_text()).items():assert hashlib.sha256((S/n).read_bytes()).hexdigest()==h,n
now=datetime.datetime.now(datetime.timezone.utc);stop=datetime.datetime(2026,11,5,13,15,tzinfo=datetime.timezone.utc);remaining=math.floor((stop-now).total_seconds());assert remaining>0
cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=6G','--property=MemorySwapMax=0','--property=CPUQuota=800%','--property=RuntimeMaxSec='+str(remaining),'--property=KillMode=control-group','--property=TimeoutStopSec=15','--property=StandardOutput=append:'+str(D/'run.log'),'--property=StandardError=append:'+str(D/'run.err'),'--property=ExecStopPost=/usr/bin/nice -n 19 /usr/bin/taskset -c 62 /usr/bin/python3 '+str(S/'capture_exit.py'),'--setenv=H359_EXIT_RECEIPT='+str(P/'unit-exit.json')]
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:cmd+=['--setenv='+k+'=1']
cmd+=['/usr/bin/nice','-n','19','/usr/bin/taskset','-c','54-63','/usr/bin/python3',str(S/'controller_h359.py')];subprocess.run(cmd,check=True)
r=dict(utc=now.isoformat(),command=cmd,compute_stop=stop.isoformat(),close_stop='2026-11-05T13:30:00Z',invocation_id=subprocess.check_output(['systemctl','--user','show',unit,'-p','InvocationID','--value'],text=True).strip());(P/'run-launch.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
