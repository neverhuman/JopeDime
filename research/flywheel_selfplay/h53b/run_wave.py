#!/usr/bin/env python3
"""Resource-bounded coordinator for preregistered H53b cells."""
from __future__ import annotations
import fcntl, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
RUNS=Path(os.environ.get('H53B_RUN_ROOT',str(Path.home()/'jope-data/h53b/runs')))
STATUS=Path.home()/"jope_plan/ns/H53b_STATUS.md"
LOCK=Path.home()/"jope_plan/qualification.lock"
CELLS=[(s,x) for x in (1.0,.1) for s in (12,13,14)]
TOTAL_LIMIT=2.5*60*60  # cumulative cell wall time; lock waits are excluded
MAX_LOCK_WAIT=3*60*60
LOCK_POLL=30
MIN_GPU_FREE_MIB=6*1024
CELL_LIMIT=20*60
RSS_LIMIT_KIB=6*1024*1024

def note(msg):
 line=f"{time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())} — {msg}\n"
 STATUS.parent.mkdir(parents=True,exist_ok=True)
 with STATUS.open('a') as f: f.write(line); f.flush(); os.fsync(f.fileno())
 print(line.rstrip(),flush=True)

def gpu_free_mib():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
 return int(out.splitlines()[0])

def rss(pid):
 try:
  for line in Path(f'/proc/{pid}/status').read_text().splitlines():
   if line.startswith('VmRSS:'): return int(line.split()[1])
 except (FileNotFoundError,ProcessLookupError): return None
 return None

def lock_gpu(wait_remaining, waited_total):
 h=LOCK.open('a+'); wait_started=time.monotonic(); next_log=wait_started
 while True:
  try:
   fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
   free=gpu_free_mib()
   if free>=MIN_GPU_FREE_MIB:
    return h, time.monotonic()-wait_started, free
   fcntl.flock(h,fcntl.LOCK_UN)
   note(f"WAIT GPU memory: {free} MiB free < {MIN_GPU_FREE_MIB} MiB; lock released")
  except BlockingIOError:
   pass
  elapsed=time.monotonic()-wait_started
  if elapsed>=wait_remaining:
   h.close(); return None,elapsed,None
  if time.monotonic()>=next_log:
   note(f"WAIT qualification.lock: cumulative_wait={waited_total+elapsed:.0f}s; polling every {LOCK_POLL}s")
   next_log=time.monotonic()+LOCK_POLL
  time.sleep(min(LOCK_POLL,wait_remaining-elapsed))

def run_cell(i,seed,scale,wall_cap,wait_remaining,waited_total):
 h,waited,free=lock_gpu(wait_remaining,waited_total)
 waited_total += waited
 if h is None:
  note(f"STOP before cell {i}: exhausted 3h cumulative GPU-lock/memory wait ({waited_total:.0f}s)")
  return None,waited_total,waited
 guard={'gpu_free_mib_after_lock':free,'gpu_min_free_mib':MIN_GPU_FREE_MIB,
        'lock_wait_seconds':waited,'cumulative_lock_wait_seconds':waited_total}
 training_started=time.monotonic()
 cell_deadline=training_started+wall_cap
 out=RUNS/f"scale-{scale:g}"/f"seed{seed}"
 out.mkdir(parents=True,exist_ok=False)
 runner_cap=max(1,wall_cap-90)
 cmd=[sys.executable,str(ROOT/'research/flywheel_selfplay/h53b/runner.py'),'--seed',str(seed),'--scale',str(scale),
      '--max-rounds','300','--wall-seconds',str(runner_cap),'--device','cuda:0','--out',str(out),'--lock-held']
 env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','CARGO_INCREMENTAL':'0','CARGO_PROFILE_DEV_DEBUG':'line-tables-only'})
 start=time.monotonic(); note(f"START cell {i}/6 seed={seed} scale={scale:g} device=cuda:0 wall_cap={wall_cap}; guard={json.dumps(guard,sort_keys=True)}")
 log=out/'run.log'; code=None; fail=None; child_nice=None
 with log.open('w') as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,preexec_fn=lambda:os.nice(10))
  while proc.poll() is None:
   if time.monotonic()>=cell_deadline: proc.terminate(); fail='coordinator cell deadline'; break
   try:
    child_nice=next(int(x.split()[1]) for x in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if x.startswith('NI:'))
   except (FileNotFoundError,ProcessLookupError,StopIteration): pass
   current=rss(proc.pid)
   if current is not None and current>=RSS_LIMIT_KIB: proc.terminate(); fail='RSS cap'; break
   time.sleep(3)
  code=proc.wait()
 fcntl.flock(h,fcntl.LOCK_UN); h.close()
 wall=time.monotonic()-start
 if fail or code!=0:
  note(f"STOP after cell {i}: {fail or f'runner exit={code}'} log={log}; later cells not launched")
  return None,waited_total,waited
 summary=json.loads((out/'summary.json').read_text()); summary['coordinator_preflight']=guard; summary['process_nice']=child_nice
 (out/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
 utility=summary.get('final_family_utility')
 note(f"DONE cell {i}/6 seed={seed} scale={scale:g} status={summary.get('status')} rounds={summary['rounds_completed']} wall={wall:.1f}s utility={utility if utility is not None else 'unavailable'} device=cuda:0")
 return {'seed':seed,'scale':scale,'wall_seconds':wall,'rounds':summary['rounds_completed'],'device':'cuda:0','status':summary.get('status')},waited_total,waited

def main():
 if os.environ.get('H53B_TIMING_PILOT_DONE')!='1':
  raise RuntimeError('GPU-only equal-budget wave requires H53B_TIMING_PILOT_DONE=1')
 RUNS.mkdir(parents=True,exist_ok=False)
 from datetime import datetime,timezone
 started=time.monotonic(); started_wall=time.time(); waited_total=0.0; compute_total=0.0
 wave_start_utc=datetime.fromtimestamp(started_wall,timezone.utc).isoformat()
 note(f'H53b GPU-only wave start={wave_start_utc}; compute_budget=150m; cumulative_gpu_wait_limit=180m; six fresh cells, preregistered order, 300 units or 20m/cell')
 completed=[]; stop=None
 for i,(seed,scale) in enumerate(CELLS,1):
  if waited_total>=MAX_LOCK_WAIT:
   stop='3h cumulative GPU-lock/memory wait exhausted'; break
  r,waited_total,_=run_cell(i,seed,scale,CELL_LIMIT,MAX_LOCK_WAIT-waited_total,waited_total)
  if r is None:
   stop='GPU-only cell stopped/failed or cumulative lock wait exhausted'; break
  completed.append(r); compute_total+=r['wall_seconds']
 result={'schema':'jope.h53b.wave/v1','prereg_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'complete':len(completed)==6 and all(x['status']=='complete' for x in completed),
         'completed':completed,'planned':[{'seed':s,'scale':x} for s,x in CELLS],
         'stop_reason':stop,'wall_seconds':time.monotonic()-started,'compute_seconds':compute_total,
         'lock_wait_seconds':waited_total,'max_lock_wait_seconds':MAX_LOCK_WAIT,'total_budget_seconds':TOTAL_LIMIT,
         'wave_start_utc':wave_start_utc,'wave_deadline_utc':None,
         'timing_pilot':'~/jope-data/h53b/attempt-03/pilot-fixed/wave.json (timing-only; excluded)',
         'infra_void_attempt':'~/jope-data/h53b/go-1930-wave/scale-1/seed12 (CPU-only; no outcome)'}
 (RUNS/'wave.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 note(f"H53b wave stopped/complete {len(completed)}/6; receipt={RUNS/'wave.json'}")
if __name__=='__main__': main()
