from pathlib import Path
import sys,subprocess,json,datetime,math,os
P=Path(__file__).parent;mode=sys.argv[1];assert mode in ['prepare','run','reference'];assert os.uname().nodename.split('.')[0]=='xbabe2'
unit='h392-'+mode;receipt=P/({'prepare':'prepare-unit-exit.json','run':'unit-exit.json','reference':'reference-archive-unit-exit.json'}[mode]);assert not receipt.exists();assert subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True).stdout.strip() not in ('active','activating')
if mode=='run':
 assert json.loads((P/'prepare-unit-exit.json').read_text())['EXIT_STATUS']=='0'
 assert json.loads((P/'reference-copy-reclamation.json').read_text())['status']=='PASS'
 assert json.loads((P/'science-equivalence.json').read_text())['status']=='PASS'
 assert json.loads((P/'prereg-published.json').read_text())['published']
now=datetime.datetime.now(datetime.timezone.utc);remaining=math.floor((datetime.datetime(2026,11,6,0,45,tzinfo=datetime.timezone.utc)-now).total_seconds());assert remaining>0
script=P/('prepare.py' if mode=='prepare' else 'source/controller_h392.py' if mode=='run' else 'reference_custody.py')
cmd=['systemd-run','--user','--unit='+unit,'--property=CPUQuota='+('800%' if mode=='run' else '100%'),'--property=MemoryMax=6G','--property=MemorySwapMax=0','--property=UnsetEnvironment=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN GH_TOKEN GITHUB_TOKEN','--property=RuntimeMaxSec='+str(remaining),'--property=KillMode=control-group','--property=TimeoutStopSec=15','--property=StandardOutput=append:'+str(P/(mode+'.log')),'--property=StandardError=append:'+str(P/(mode+'.err')),'--property=ExecStopPost=/usr/bin/nice -n 19 /usr/bin/taskset -c 12 /usr/bin/python3 '+str(P/'source/capture_exit.py'),'--setenv=H392_EXIT_RECEIPT='+str(receipt)]
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:cmd+=['--setenv='+k+'=1']
cmd+=['/usr/bin/nice','-n','19','/usr/bin/taskset','-c','12','/usr/bin/python3',str(script)];subprocess.run(cmd,check=True);(P/(mode+'-launch.json')).write_text(json.dumps(dict(utc=now.isoformat(),fixed_deadline='2026-11-06T00:45:00Z',command=cmd,invocation_id=subprocess.check_output(['systemctl','--user','show',unit,'-p','InvocationID','--value'],text=True).strip()),indent=2)+'\n');print('Launched',unit)
