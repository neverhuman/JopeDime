"""One five-second assigned physical/SMT probe plus same-interval host CPU."""
import json,subprocess
from pathlib import Path
ASSIGNED=tuple(range(15,27))
def cpu():
 a=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(a[:8]),a[3]+a[4]
def eligible_from_report(report):
 stats=report['sysstat']['hosts'][0]['statistics'];assert len(stats)==1
 values={int(x['cpu']):float(x['idle']) for x in stats[0]['cpu-load'] if x['cpu']!='all'}
 assert all(0<=x<=100 for x in values.values())
 idle={c:min(values[c],values[c+64]) for c in ASSIGNED}
 return [c for c,v in idle.items() if v>=85][:6],idle
def probe():
 start=cpu()
 raw=subprocess.check_output(['mpstat','-P',','.join(map(str,(*ASSIGNED,*(c+64 for c in ASSIGNED)))),'5','1','-o','JSON'],text=True)
 end=cpu();dt=end[0]-start[0];assert dt>0
 host_cpu=100*(1-(end[1]-start[1])/dt)
 eligible,idle=eligible_from_report(json.loads(raw))
 return eligible,idle,raw,host_cpu
