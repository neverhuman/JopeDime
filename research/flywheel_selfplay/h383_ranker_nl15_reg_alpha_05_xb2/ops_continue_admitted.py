"""Continue admitted H383 allocation after a pre-fit affinity order assertion; no probe."""
from pathlib import Path
import os,sys,json,hashlib,datetime,subprocess,threading
P=Path(__file__).parent;S=P/'source';D=Path('/home/ubuntu/jope-data/h383-hwv');sys.path.insert(0,str(S))
import controller_h383 as c

def bind_threads(thread_ids,eligible):
 for tid in thread_ids:os.sched_setaffinity(int(tid),set(eligible))
 after={str(tid):sorted(os.sched_getaffinity(int(tid))) for tid in thread_ids}
 assert all(set(v)==set(eligible) for v in after.values()),'Kernel affinity must equal admitted core set'
 return after

def main():
 assert os.uname().nodename.split('.')[0]=='xbabe2' and os.getpriority(os.PRIO_PROCESS,0)==19
 admission=json.loads((P/'capacity.json').read_text());eligible=admission['eligible_cores'];assert eligible==[31,12,13,14] and c.admission_pass(eligible,admission['host_at_probe']['cpu_percent'])
 gate=json.loads((P/'h377-release-gate.json').read_text());assert gate['released'] and not gate['timed_out']
 old=json.loads((P/'startup-unit-exit.json').read_text());assert old['EXIT_STATUS']=='1' and old['INVOCATION_ID']=='20c72e3931ce4f159807e274cdcc3691'
 assert not list(D.glob('P*/fold_*/root_*/constructor_audit.jsonl')) and not list(D.glob('P*/fold_*/root_*/first_fit_started.json')) and not list(D.glob('P*/fold_*/root_*/fit_lock.json')) and not list(D.glob('P*/fold_*/root_*/score_once.json'))
 assert not (P/'runtime-forecast-at-launch.json').exists()
 assert json.loads((P/'prepare-unit-exit.json').read_text())['EXIT_STATUS']=='0'
 for n,h in json.loads((P/'runtime-source-manifest.json').read_text()).items():assert hashlib.sha256((S/n).read_bytes()).hexdigest()==h,n
 review=json.loads((P/'startup-continuation-review.json').read_text());assert review['status']=='PASS' and review['entrypoint_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 sample=dict(mem_available_bytes=c.psutil.virtual_memory().available,scratch_bytes=sum(p.stat().st_size for p in D.rglob('*') if p.is_file()))
 reason=c.guard.stop_reasons(sample,datetime.datetime.now(datetime.timezone.utc))
 if reason:c.save(D/'STOP.json',dict(status='INCOMPLETE_RESOURCE_GUARD',stage='before_startup_continuation',reason=reason,**sample));raise SystemExit(2)
 threading.Thread(target=c.guard.monitor,daemon=True).start()
 before={p.name:sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()};after=bind_threads(list(before),eligible)
 c.save(P/'controller-affinity-binding.json',dict(before=before,after=after,eligible_cores=eligible,comparison='set equality; preserve priority order in admission receipt',maximum_workers=len(eligible),continuation_of_admitted_allocation=True))
 forecast=c.estimate(admission);forecast.update(capacity=admission,continuation=True,capacity_samples_total=1)
 c.save(P/'runtime-forecast-at-launch.json',forecast);c.save(D/'runtime_forecast.json',forecast);print(json.dumps(dict(forecast=forecast)),flush=True)
 if not forecast['launch_full_36']:c.save(D/'INCOMPLETE_RUNTIME_FORECAST.json',forecast);raise SystemExit(2)
 env=dict(os.environ,H383_WORKERS=str(len(eligible)))
 for phase in ['fit','score']:subprocess.run(['taskset','-c',','.join(map(str,eligible)),sys.executable,str(S/'run_h383.py'),phase],env=env,check=True)
if __name__=='__main__':main()
