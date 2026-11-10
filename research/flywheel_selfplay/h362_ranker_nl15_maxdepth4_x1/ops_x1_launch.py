from pathlib import Path
import os,subprocess,json,datetime,math,sys
P=Path(__file__).parent;S=P/'source';D=P.parent;mode=sys.argv[1];assert mode in ('prepare','run') and os.uname().nodename.split('.')[0]=='xbabe1'
unit='h362-x1-'+mode;state=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True).stdout.strip();assert state not in ('active','activating')
now=datetime.datetime.now(datetime.timezone.utc);deadline=datetime.datetime(2026,11,5,14,30,tzinfo=datetime.timezone.utc);remaining=math.floor((deadline-now).total_seconds());assert remaining>0
receipt=P/('prepare-unit-exit.json' if mode=='prepare' else 'unit-exit.json');assert not receipt.exists()
if mode=='run':
 assert json.loads((P/'prepare-unit-exit.json').read_text())['EXIT_STATUS']=='0'
 assert json.loads((P/'stage-verification.json').read_text())['status']=='PASS'
 assert json.loads((P/'prereg-published.json').read_text())['published']
cmd=['systemd-run','--user','--unit='+unit,'--property=CPUQuota=800%','--property=MemoryMax=6G','--property=MemorySwapMax=0','--property=UnsetEnvironment=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE AWS_DEFAULT_PROFILE GH_TOKEN GITHUB_TOKEN','--property=RuntimeMaxSec='+str(remaining),'--property=KillMode=control-group','--property=TimeoutStopSec=15','--property=StandardOutput=append:'+str(P/(mode+'.log')),'--property=StandardError=append:'+str(P/(mode+'.err')),'--property=ExecStopPost=/usr/bin/nice -n 19 /usr/bin/taskset -c 35 /usr/bin/python3 '+str(S/'capture_exit.py'),'--setenv=H362_EXIT_RECEIPT='+str(receipt)]
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:cmd+=['--setenv='+key+'=1']
cmd+=['/usr/bin/nice','-n','19','/usr/bin/taskset','-c','35','/usr/bin/python3',str(P/'x1_prepare.py' if mode=='prepare' else S/'controller_h362.py')]
subprocess.run(cmd,check=True);record=dict(utc=now.isoformat(),fixed_deadline=deadline.isoformat(),command=cmd,invocation_id=subprocess.check_output(['systemctl','--user','show',unit,'-p','InvocationID','--value'],text=True).strip());(P/(mode+'-launch.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
