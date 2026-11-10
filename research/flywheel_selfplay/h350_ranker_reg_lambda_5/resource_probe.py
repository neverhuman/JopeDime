"""Exactly one5-second candidate-core/SMT sample and same-interval hostCPU gate."""
import json,subprocess,time
from pathlib import Path
ASSIGNED=tuple(range(15,27))
def eligible_from_report(report):
 stats=report['sysstat']['hosts'][0]['statistics'];assert len(stats)==1
 values={int(x['cpu']):float(x['idle']) for x in stats[0]['cpu-load'] if x['cpu']!='all'}
 idle={c:min(values[c],values[c+64]) for c in ASSIGNED};assert all(0<=v<=100 for v in values.values())
 return [c for c,v in idle.items() if v>=85][:6],idle
def admission_pass(eligible,host_cpu):return 4<=len(eligible)<=6 and 0<=host_cpu<=70

def host_ticks():
 a=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(a[:8]),a[3]+a[4]
def probe():
 before=host_ticks();start=time.monotonic()
 raw=subprocess.check_output(['mpstat','-P',','.join(map(str,(*ASSIGNED,*(c+64 for c in ASSIGNED)))),'5','1','-o','JSON'],text=True)
 after=host_ticks();elapsed=time.monotonic()-start;dt=after[0]-before[0];assert dt>0
 host=100*(1-(after[1]-before[1])/dt);assert 0<=host<=100
 eligible,idle=eligible_from_report(json.loads(raw))
 return eligible,idle,raw,dict(cpu_percent=host,interval_seconds=elapsed,aggregate_ticks_before=before,aggregate_ticks_after=after,method='Aggregate/proc/stat deltas bracketing the SAME single mpstat5-second interval; not another core sample')
