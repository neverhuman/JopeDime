"""Owner-corrected guard: 60s host CPU; load>108 at3 consecutive60s samples."""
import datetime,json,os,subprocess,time
from pathlib import Path
DATA=Path('/home/ubuntu/jope-data/h391-api01')
STOP=datetime.datetime(2026,11,5,23,15,tzinfo=datetime.timezone.utc)
def cpu():
 a=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(a[:8]),a[3]+a[4]
def usage(before,after):
 dt=after[0]-before[0];assert dt>0;return 100*(1-(after[1]-before[1])/dt)
def cgroup_cpu_usec():
 try:
  group=next(x.split('::',1)[1] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
  values=dict(x.split() for x in (Path('/sys/fs/cgroup')/group.lstrip('/')/'cpu.stat').read_text().splitlines());return int(values['usage_usec'])
 except (OSError,StopIteration,KeyError):return None
def host_reasons(cpu_60s,load,streak):
 streak=streak+1 if load>108 else 0
 reasons=[]
 if cpu_60s>85:reasons.append('host CPU averaged60s >85%')
 if streak>=3:reasons.append('load1 >108 on3 consecutive60s samples')
 return reasons,streak
def monitor(unit='h391-run.service'):
 prev=cpu();window_ticks=prev;window_start=time.monotonic();streak=0
 while True:
  time.sleep(5);ticks=cpu();short=usage(prev,ticks);prev=ticks
  elapsed=time.monotonic()-window_start;load=os.getloadavg()[0];reasons=[];average=None
  if elapsed>=60:
   average=usage(window_ticks,ticks);r,streak=host_reasons(average,load,streak);reasons+=r;window_ticks=ticks;window_start=time.monotonic()
  mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
  size=0
  for p in DATA.rglob('*'):
   try:
    if p.is_file():size+=p.stat().st_size
   except FileNotFoundError:pass
  utc=datetime.datetime.now(datetime.timezone.utc)
  record=dict(own_cgroup_cpu_usage_usec_observed=cgroup_cpu_usec(),utc=utc.isoformat(),cpu_percent=short,short_CPU_diagnostic_only=True,cpu_percent_60s=average,cpu_window_seconds=elapsed if average is not None else None,load1=load,load_over108_consecutive_60s_samples=streak,load_sample_is_60s=average is not None,mem_available_bytes=mem,scratch_bytes=size)
  with (DATA/'resource.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  if mem<20*1024**3:reasons.append('MemAvailable <20GiB')
  if size>2_000_000_000:reasons.append('scratch >2GB')
  if utc>=STOP:reasons.append('compute deadline')
  if reasons:
   (DATA/'STOP.json').write_text(json.dumps(dict(reason=reasons,**record)))
   subprocess.run(['systemctl','--user','stop',unit],check=False);return
