#!/usr/bin/env python3
"""Resource-bounded coordinator for preregistered H53b cells."""
from __future__ import annotations
import fcntl, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
RUNS=Path(os.environ.get('H53B_RUN_ROOT',str(Path.home()/'jope-data/h53b/runs')))
WAVE_START_UTC="2026-10-30T18:48:50+00:00"
STATUS=Path.home()/"jope_plan/ns/H53b_STATUS.md"
LOCK=Path.home()/"jope_plan/qualification.lock"
CELLS=[(s,x) for x in (1.0,.1) for s in (12,13,14)]
TOTAL_LIMIT=2.5*60*60
CELL_LIMIT=20*60
PILOT_LIMIT=10*60
RSS_LIMIT_KIB=6*1024*1024

def note(msg):
 line=f"{time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())} — {msg}\n"
 STATUS.parent.mkdir(parents=True,exist_ok=True)
 with STATUS.open('a') as f: f.write(line); f.flush(); os.fsync(f.fileno())
 print(line.rstrip(),flush=True)

def mem_kib():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'): return int(line.split()[1])
 raise RuntimeError('MemAvailable unavailable')

def cpu_pct():
 def sample():
  a=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:])); return sum(a),a[3]+a[4]
 a=sample(); time.sleep(1); b=sample(); total=b[0]-a[0]; idle=b[1]-a[1]
 return 100*(total-idle)/total

def gpu_free_mib():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
 return int(out.splitlines()[0])

def rss(pid):
 try:
  for line in Path(f'/proc/{pid}/status').read_text().splitlines():
   if line.startswith('VmRSS:'): return int(line.split()[1])
 except (FileNotFoundError,ProcessLookupError): return None
 return None

def lock_gpu(deadline):
 h=LOCK.open('a+'); start=time.monotonic()
 while time.monotonic()<deadline:
  try: fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB); return h,False
  except BlockingIOError:
   if time.monotonic()-start>=20*60: h.close(); return None,True
   time.sleep(5)
 h.close(); return None,False

def wait_cpu_guard(deadline):
 while time.monotonic()<deadline:
  cpu=cpu_pct(); mem=mem_kib()
  if cpu<=85 and mem>=20*1024*1024: return {'cpu_utilization_percent':cpu,'mem_available_kib':mem}
  note(f"WAIT resource guard: CPU={cpu:.1f}% MemAvailable={mem} KiB")
  time.sleep(20)
 return None

def run_cell(i,seed,scale,overall_deadline,wall_cap):
 cell_started=time.monotonic()
 pilot=wall_cap<=PILOT_LIMIT
 resource_deadline=min(overall_deadline,cell_started+20*60)
 guard=wait_cpu_guard(resource_deadline)
 if guard is None: note(f"STOP before cell {i}: total/pilot deadline while waiting for CPU/memory guard"); return None
 h,fallback=lock_gpu(resource_deadline)
 if time.monotonic()>=resource_deadline and not fallback: note(f"STOP before cell {i}: deadline waiting on GPU lock"); return None
 device='cpu' if fallback else 'cuda:0'
 if device=='cuda:0':
  guard['gpu_free_mib']=gpu_free_mib()
  if guard['gpu_free_mib']<20*1024:
   fcntl.flock(h,fcntl.LOCK_UN); h.close(); h=None; fallback=True; device='cpu'
 if device=='cpu':
  guard=wait_cpu_guard(overall_deadline)
  if guard is None: note(f"STOP before cell {i}: CPU fallback guard failed"); return None
  guard['fallback_reason']='GPU qualification lock busy >20m or <20GiB free'
 training_started=time.monotonic()
 cell_deadline=min(overall_deadline,training_started+wall_cap)
 if time.monotonic()>=cell_deadline: note(f"STOP before cell {i}: cell execution budget unavailable after resource acquisition"); return None
 out=RUNS/f"scale-{scale:g}"/f"seed{seed}"
 out.mkdir(parents=True,exist_ok=False)
 runner_cap=max(1,wall_cap-90)
 cmd=[sys.executable,str(ROOT/'research/flywheel_selfplay/h53b/runner.py'),'--seed',str(seed),'--scale',str(scale),
      '--max-rounds','300','--wall-seconds',str(runner_cap),'--device',device,'--out',str(out)]
 if h is not None: cmd.append('--lock-held')
 env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','CARGO_INCREMENTAL':'0','CARGO_PROFILE_DEV_DEBUG':'line-tables-only'})
 start=time.monotonic(); note(f"START cell {i}/6 seed={seed} scale={scale:g} device={device} wall_cap={wall_cap}; guard={json.dumps(guard,sort_keys=True)}")
 log=out/'run.log'; code=None; fail=None
 with log.open('w') as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,preexec_fn=lambda:os.nice(10))
  while proc.poll() is None:
   if time.monotonic()>=cell_deadline: proc.terminate(); fail='coordinator cell deadline'; break
   current=rss(proc.pid)
   if current is not None and current>=RSS_LIMIT_KIB: proc.terminate(); fail='RSS cap'; break
   time.sleep(3)
  code=proc.wait()
 if h is not None: fcntl.flock(h,fcntl.LOCK_UN); h.close()
 wall=time.monotonic()-start
 if fail or code!=0: note(f"STOP after cell {i}: {fail or f'runner exit={code}'} log={log}; later cells not launched"); return None
 summary=json.loads((out/'summary.json').read_text()); summary['coordinator_preflight']=guard; summary['coordinator_nice']=10
 (out/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
 utility=summary.get('final_family_utility')
 note(f"DONE cell {i}/6 seed={seed} scale={scale:g} status={summary.get('status')} rounds={summary['rounds_completed']} wall={wall:.1f}s utility={utility if utility is not None else 'unavailable'} device={device}")
 return {'seed':seed,'scale':scale,'wall_seconds':wall,'rounds':summary['rounds_completed'],'device':device,'status':summary.get('status')}

def main():
 RUNS.mkdir(parents=True,exist_ok=False)
 from datetime import datetime,timezone
 started=time.monotonic(); fixed_end=datetime.fromisoformat(WAVE_START_UTC).replace(tzinfo=timezone.utc).timestamp()+TOTAL_LIMIT
 deadline=started+max(0.0,fixed_end-time.time())
 note('H53b wave start; preregistration commit precedes all cells; six cells, seeds 12/13/14 × scales 1.0/0.1; 300 update units or 20m/cell; pilot <=10m')
 completed=[]
 if os.environ.get('H53B_TIMING_PILOT_DONE')=='1':
  prior_text=os.environ.get('H53B_PRIOR_CELL_SUMMARY')
  if prior_text:
   prior=Path(prior_text); d=json.loads(prior.read_text())
   if d.get('status')!='complete' or d.get('seed')!=12 or d.get('initialization_scale')!=1.0: raise RuntimeError('prior timing cell identity/status mismatch')
   dest=RUNS/'scale-1'/'seed12'/'summary.json'; dest.parent.mkdir(parents=True,exist_ok=False); shutil.copyfile(prior,dest)
   import hashlib
   if hashlib.sha256(prior.read_bytes()).digest()!=hashlib.sha256(dest.read_bytes()).digest(): raise RuntimeError('prior summary copy verification failed')
   completed.append({'seed':12,'scale':1.0,'wall_seconds':d['wall_seconds'],'rounds':d['rounds_completed'],'device':d['device'],'status':d['status'],'prior_summary':str(prior)})
   cells=CELLS[1:]; first_index=2
   note(f'Continuing with five fresh cells after timing cell1; exact prior summary copied and verified from {prior}; fixed 2.5-hour deadline retained.')
  else:
   cells=CELLS; first_index=1
   note('Continuing six fresh cells after pre-training attempts; no adapter state carries forward; original fixed deadline retained.')
  stop=None
  for i,(seed,scale) in enumerate(cells,first_index):
   r=run_cell(i,seed,scale,deadline,CELL_LIMIT)
   if r is None: stop='cell stopped or failed'; break
   completed.append(r)
  result={'complete':len(completed)==6 and all(x['status']=='complete' for x in completed),
          'completed':completed,'planned':[{'seed':s,'scale':x} for s,x in CELLS],
          'stop_reason':stop,'wall_seconds':time.monotonic()-started,
          'pilot_attempts':['~/jope-data/h53b/runs/wave.json','~/jope-data/h53b/attempt-02/wave.json','~/jope-data/h53b/attempt-03/pilot-fixed/wave.json']}
 else:
  first=run_cell(1,12,1.0,deadline,PILOT_LIMIT)
  if first is None:
   result={'complete':False,'completed':completed,'stop_reason':'pilot did not produce a receipt'}
  else:
   completed.append(first)
   if os.environ.get('H53B_PILOT_ONLY')=='1':
    result={'complete':False,'completed':completed,'stop_reason':'pilot-only run; review timing before more cells','wall_seconds':time.monotonic()-started}
    note('Pilot-only run complete; no other cell launched pending throughput review')
   else:
    sec_per_round=first['wall_seconds']/max(1,first['rounds'])
    projected_cell=min(CELL_LIMIT,sec_per_round*300)
    projected_total=first['wall_seconds']+5*projected_cell
    note(f"PILOT estimate: {sec_per_round:.3f}s/round; projected 6-cell total={projected_total/60:.1f}m at cell caps")
    if projected_total>2.5*60*60:
     result={'complete':False,'completed':completed,'stop_reason':'projected total exceeds 2.5h','projected_total_seconds':projected_total}
     note('STOP after timing pilot: projected six-cell total exceeds 2.5h')
    else:
     stop=None
     for i,(seed,scale) in enumerate(CELLS[1:],2):
      r=run_cell(i,seed,scale,deadline,CELL_LIMIT)
      if r is None: stop='cell stopped or failed'; break
      completed.append(r)
     result={'complete':len(completed)==6 and all(x['status']=='complete' for x in completed),
             'completed':completed,'planned':[{'seed':s,'scale':x} for s,x in CELLS],
             'stop_reason':stop,'wall_seconds':time.monotonic()-started,'projected_total_seconds':projected_total}
 result['schema']='jope.h53b.wave/v1'; result['prereg_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 (RUNS/'wave.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 note(f"H53b wave stopped/complete {len(completed)}/6; receipt={RUNS/'wave.json'}")
if __name__=='__main__': main()
