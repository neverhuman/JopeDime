"""Owner-corrected host guard, only H376's own xbabe1 cgroups."""
from pathlib import Path
import datetime,json,os,re,subprocess,time,shutil
DATA=Path('/home/ubuntu/jope-data/h376-x1')
STOP=datetime.datetime(2026,11,5,18,15,tzinfo=datetime.timezone.utc)
def cpu():
 a=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(a[:8]),a[3]+a[4]
def host_online_cpu_count(stat_text=None):
 text=Path('/proc/stat').read_text() if stat_text is None else stat_text
 count=sum(bool(re.match(r'^cpu\d+\s',line)) for line in text.splitlines());assert count>0;return count
def next_load_streak(previous,load,host_nproc):return previous+1 if load>.85*host_nproc else 0

def stop_reasons(r,utc):
 reason=[]
 if r.get('cpu_window_seconds',0)>=60 and r.get('cpu_percent_60s') is not None and r['cpu_percent_60s']>85:reason.append('host CPU >85% averaged over60s')
 if r.get('load_high_consecutive_60s',0)>=3:reason.append('host load >0.85*host_nproc on3 consecutive60s samples')
 if r['mem_available_bytes']<20*2**30:reason.append('MemAvailable <20GiB')
 if r['root_free_bytes']<80*2**30:reason.append('root free <80GiB')
 if r['scratch_bytes']>2_000_000_000:reason.append('scratch >2GB')
 if utc>=STOP:reason.append('compute deadline')
 return reason

def monitor(unit='h376-x1-run.service'):
 assert unit in ('h376-x1-run.service','h376-x1-prepare.service'),'Never signal a foreign cgroup'
 prev=cpu();window_prev=prev;window_start=time.monotonic();streak=0;count=host_online_cpu_count()
 while True:
  time.sleep(5);now=time.monotonic();total,idle=cpu();dt=total-prev[0];usage=100*(1-(idle-prev[1])/dt) if dt else 0;prev=(total,idle)
  mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
  size=0
  for p in DATA.rglob('*'):
   try:
    if p.is_file():size+=p.stat().st_size
   except FileNotFoundError:pass
  utc=datetime.datetime.now(datetime.timezone.utc);load=os.getloadavg()[0]
  r=dict(utc=utc.isoformat(),cpu_percent=usage,cpu_percent_60s=None,cpu_window_seconds=0,load1=load,host_nproc=count,host_nproc_source='Count online cpuN rows in host /proc/stat, independent of admitted affinity',load_threshold=.85*count,load_window_sample=False,load_high_consecutive_60s=streak,mem_available_bytes=mem,scratch_bytes=size,root_free_bytes=shutil.disk_usage('/').free)
  if now-window_start>=60:
   elapsed=now-window_start;delta=total-window_prev[0];assert delta>0
   average=100*(1-(idle-window_prev[1])/delta);streak=next_load_streak(streak,load,count)
   r.update(cpu_percent_60s=average,cpu_window_seconds=elapsed,cpu_window_ticks_before=list(window_prev),cpu_window_ticks_after=[total,idle],load_window_sample=True,load_high_consecutive_60s=streak)
   window_start=now;window_prev=(total,idle)
  with (DATA/'resource.jsonl').open('a') as out:out.write(json.dumps(r)+'\n')
  reason=stop_reasons(r,utc)
  if reason:
   (DATA/'STOP.json').write_text(json.dumps(dict(reason=reason,**r))+'\n');subprocess.run(['systemctl','--user','stop',unit],check=False);return
