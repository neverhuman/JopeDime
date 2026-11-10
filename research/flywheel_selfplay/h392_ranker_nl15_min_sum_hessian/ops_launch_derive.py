from pathlib import Path
import subprocess,datetime,json,sys,math
P=Path(__file__).parent;S=P/'source';which=sys.argv[1];assert which in ('derive',);unit='h392-'+which;assert subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True).stdout.strip() not in ('active','activating')
receipt=P/(which+'-unit-exit.json');assert not receipt.exists();now=datetime.datetime.now(datetime.timezone.utc);remaining=math.floor((datetime.datetime(2026,11,6,0,45,tzinfo=datetime.timezone.utc)-now).total_seconds());script=S/'derive_minhess.py'
cmd=['systemd-run','--user','--unit='+unit,'-p','CPUQuota=100%','-p','MemoryMax=6G','-p','MemorySwapMax=0','-p','RuntimeMaxSec='+str(remaining),'-p','KillMode=control-group','-p','StandardOutput=append:'+str(P/(which+'.log')),'-p','StandardError=append:'+str(P/(which+'.err')),'-p','ExecStopPost=/usr/bin/nice -n 19 /usr/bin/taskset -c 12 /usr/bin/python3 '+str(S/'capture_exit.py'),'--setenv=H392_EXIT_RECEIPT='+str(receipt)]
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:cmd+=['--setenv='+k+'=1']
cmd+=['/usr/bin/nice','-n','19','/usr/bin/taskset','-c','12','/usr/bin/python3',str(script)];subprocess.run(cmd,check=True);(P/(which+'-launch.json')).write_text(json.dumps(dict(command=cmd,utc=now.isoformat()),indent=2)+'\n')
