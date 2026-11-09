"""Prepared BASE/native/prereg gates; ONE CPU admission; full36 or incomplete."""
from pathlib import Path
import datetime,json,os,subprocess,sys,threading,hashlib,shutil,time
import psutil
import guard_h383 as guard
from resource_probe import probe,ASSIGNED,admission_pass
from forecast_h383 import estimate
HERE=Path(__file__).resolve().parent;P=HERE.parent;D=Path('/home/ubuntu/jope-data/h383-hwv')
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def wait_for_h377():
 cutoff=datetime.datetime(2026,11,5,18,50,tzinfo=datetime.timezone.utc);checks=[]
 while True:
  now=datetime.datetime.now(datetime.timezone.utc)
  observed=subprocess.run(['systemctl','--user','is-active','h377-run'],capture_output=True,text=True)
  state=observed.stdout.strip();checks.append(dict(utc=now.isoformat(),state=state,exit_status=observed.returncode))
  released=state=='inactive';timed_out=not released and now>=cutoff
  save(P/'h377-release-gate.json',dict(only_command=['systemctl','--user','is-active','h377-run'],checks=checks,cutoff=cutoff.isoformat(),released=released,timed_out=timed_out,capacity_samples=0,foreign_signals=0))
  if released:return True
  if timed_out:return False
  time.sleep(min(15,max(.05,(cutoff-now).total_seconds())))

def main():
 assert os.uname().nodename.split('.')[0]=='xbabe2' and os.getpriority(os.PRIO_PROCESS,0)==19
 assert json.loads((P/'prereg-published.json').read_text())['published']
 assert json.loads((P/'prepare-unit-exit.json').read_text())['EXIT_STATUS']=='0'
 assert json.loads((P/'stage-verification.json').read_text())['all_staged_SHA256_verified']
 assert not any(k in os.environ for k in ['AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_SESSION_TOKEN','GH_TOKEN','GITHUB_TOKEN'])
 assert json.loads((P/'base-prepared.json').read_text())['status']=='PASS' and json.loads((P/'nativecheck-exit.json').read_text())['EXIT_STATUS']=='0'
 assert json.loads((D/'regression.json').read_text())['status']=='PASS' and json.loads((P/'input_identity_validation.json').read_text())['status']=='PASS'
 assert json.loads((P/'base-native-config.json').read_text())['status']=='PASS'
 assert json.loads((P/'nl15-native-config.json').read_text())['status']=='PASS'
 assert json.loads((D/'nl15_regression.json').read_text())['status']=='PASS'
 manifest=json.loads((HERE/'manifest.json').read_text())
 for n,h in manifest['source_sha256'].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h
 assert not (P/'capacity-interval-mpstat.json').exists()
 initial=dict(cpu_percent=psutil.cpu_percent(interval=1),load1=os.getloadavg()[0],mem_available_bytes=psutil.virtual_memory().available,root_free_bytes=shutil.disk_usage('/').free,scratch_bytes=sum(p.stat().st_size for p in D.rglob('*') if p.is_file()))
 save(P/'initial-resource.json',initial)
 reasons=guard.stop_reasons(initial,datetime.datetime.now(datetime.timezone.utc))
 if reasons:
  save(D/'STOP.json',dict(status='INCOMPLETE_RESOURCE_GUARD',stage='before_capacity_sample',reason=reasons,**initial));raise SystemExit(2)
 threading.Thread(target=guard.monitor,daemon=True).start()
 if not wait_for_h377():
  save(D/'INCOMPLETE_CPU_ADMISSION.json',dict(status='INCOMPLETE_CPU_ADMISSION',reason='h377-run not inactive by18:50Z',capacity_probes=0,optimizer_fits=0,outer_scores=0));raise SystemExit(2)
 eligible,idle,raw,host=probe();(P/'capacity-interval-mpstat.json').write_text(raw)
 cap=dict(eligible_cores=eligible,both_smt_idle=idle,host_at_probe=host,host_cpu_maximum=70,sample_seconds=5,minimum_required=4,assigned_cores=list(ASSIGNED),maximum_taken=8,ordering='first at most8 passing pairs over one union interval,primary27..37 thenfallback12/13/14',capacity_retries=0)
 save(P/'capacity.json',cap);save(D/'capacity.json',cap);save(D/'pre_fit_resource.json',cap)
 if not admission_pass(eligible,host['cpu_percent']):
  save(D/'INCOMPLETE_CPU_ADMISSION.json',dict(status='INCOMPLETE_CPU_ADMISSION',capacity=cap,reason='Need at least4 admitted cores and same-interval hostCPU<=70%',optimizer_fits=0,outer_scores=0));raise SystemExit(2)
 # The controller and its resource-monitor thread inherit only admitted cores.
 before={p.name:sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()}
 for tid in before:os.sched_setaffinity(int(tid),set(eligible))
 after={p.name:sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()};assert all(v==eligible for v in after.values())
 save(P/'controller-affinity-binding.json',dict(before=before,after=after,eligible_cores=eligible,maximum_workers=8))
 forecast=estimate(cap);forecast.update(capacity=cap)
 save(P/'runtime-forecast-at-launch.json',forecast);save(D/'runtime_forecast.json',forecast)
 print(json.dumps(dict(forecast=forecast)),flush=True)
 if not forecast['launch_full_36']:
  save(D/'INCOMPLETE_RUNTIME_FORECAST.json',forecast);raise SystemExit(2)
 env=dict(os.environ,H383_WORKERS=str(len(eligible)))
 for phase in ['fit','score']:subprocess.run(['taskset','-c',','.join(map(str,eligible)),sys.executable,str(HERE/'run_h383.py'),phase],env=env,check=True)
if __name__=='__main__':main()
