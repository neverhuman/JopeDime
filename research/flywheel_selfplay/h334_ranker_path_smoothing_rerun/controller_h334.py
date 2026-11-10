"""One BASE gate, one capacity probe, full36 or incomplete; no retries."""
from pathlib import Path
import datetime,json,os,subprocess,sys,threading
import psutil
import guard_h334 as guard
from resource_probe import probe
from forecast_h334 import estimate
HERE=Path(__file__).resolve().parent;P=HERE.parent;D=Path('/home/ubuntu/jope-data/h334-gov03')

def main():
 assert os.uname().nodename.split('.')[0]=='xbabe2' and os.getpriority(os.PRIO_PROCESS,0)==19
 assert json.loads((P/'prereg-published.json').read_text())['published']
 assert not (P/'capacity-interval-mpstat.json').exists()
 initial=dict(cpu_percent=psutil.cpu_percent(interval=1),load1=os.getloadavg()[0],mem_available_bytes=psutil.virtual_memory().available)
 assert initial['cpu_percent']<=85 and initial['load1']<=85 and initial['mem_available_bytes']>=20*1024**3
 (P/'initial-resource.json').write_text(json.dumps(initial,indent=2)+'\n')
 threading.Thread(target=guard.monitor,daemon=True).start()
 for file,args in [('fetch_base_h334.py',[]),('input_preflight.py',[]),('run_h334.py',['regress'])]:
  subprocess.run([sys.executable,str(HERE/file),*args],check=True)
 assert json.loads((D/'regression.json').read_text())['status']=='PASS'
 eligible,idle,raw=probe();(P/'capacity-interval-mpstat.json').write_text(raw)
 cap=dict(eligible_cores=eligible,both_smt_idle=idle,sample_seconds=5,minimum_required=4,assigned_cores=[1,2,3,7,9,10,12,14],capacity_retries=0)
 (P/'capacity.json').write_text(json.dumps(cap,indent=2)+'\n');(D/'pre_fit_resource.json').write_text(json.dumps(cap,indent=2)+'\n')
 forecast=estimate(cap) if eligible else dict(launch_full_36=False,reason='Zero eligible cores',workers=0)
 forecast.update(capacity=cap)
 (P/'runtime-forecast-at-launch.json').write_text(json.dumps(forecast,indent=2)+'\n');(D/'runtime_forecast.json').write_text(json.dumps(forecast,indent=2)+'\n')
 print(json.dumps(dict(forecast=forecast)),flush=True)
 if not forecast['launch_full_36']:
  (D/'INCOMPLETE_RUNTIME_FORECAST.json').write_text(json.dumps(forecast,indent=2)+'\n');raise SystemExit(2)
 env=dict(os.environ,H334_WORKERS=str(len(eligible)))
 for phase in ['fit','score']:
  subprocess.run(['taskset','-c',','.join(map(str,eligible)),sys.executable,str(HERE/'run_h334.py'),phase],env=env,check=True)

if __name__=='__main__':main()
