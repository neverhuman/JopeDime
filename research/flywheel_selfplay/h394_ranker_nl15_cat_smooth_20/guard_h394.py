"""Owner H394 guard: CPU over 60s; load108 on three 60s samples."""
from pathlib import Path
import datetime,json,os,subprocess,time
DATA=Path('/home/ubuntu/jope-data/h394-gov03')
STOP=datetime.datetime(2026,11,6,1,45,tzinfo=datetime.timezone.utc)
def cpu():
 a=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(a[:8]),a[3]+a[4]
def breaches(mem,size,cpu60,load_streak,now):
 r=[]
 if mem<20*1024**3:r.append('MemAvailable <20GiB')
 if size>2_000_000_000:r.append('scratch >2GB')
 if cpu60 is not None and cpu60>85:r.append('host CPU >85% averaged over60s')
 if load_streak>=3:r.append('load1 >108 on3consecutive60s samples')
 if now>=STOP:r.append('fixed compute deadline')
 return r
def monitor(unit='h394-run.service'):
 DATA.mkdir(parents=True,exist_ok=True);previous=anchor=cpu();anchor_time=time.monotonic();streak=0;cpu60=None
 while True:
  time.sleep(5);total,idle=cpu();now_mono=time.monotonic();dt=total-previous[0];instant=100*(1-(idle-previous[1])/dt) if dt else 0;previous=(total,idle)
  interval=now_mono-anchor_time;sample60=interval>=60
  if sample60:
   ticks=total-anchor[0];cpu60=100*(1-(idle-anchor[1])/ticks) if ticks else 0;streak=streak+1 if os.getloadavg()[0]>108 else 0;anchor=(total,idle);anchor_time=now_mono
  mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
  size=0
  for p in DATA.rglob('*'):
   try:
    if p.is_file():size+=p.stat().st_size
   except FileNotFoundError:pass
  now=datetime.datetime.now(datetime.timezone.utc);r=dict(utc=now.isoformat(),cpu_percent=instant,cpu_percent_60s=cpu60,cpu60_interval_seconds=interval if sample60 else None,load1=os.getloadavg()[0],load1_over108_consecutive60s=streak,sampled60s=sample60,mem_available_bytes=mem,scratch_bytes=size,guard='CPU60>85 OR load108x3x60s OR mem20GiB OR scratch2GB OR fixedstop')
  with (DATA/'resource.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  reason=breaches(mem,size,cpu60,streak,now)
  if reason:
   (DATA/'STOP.json').write_text(json.dumps(dict(reason=reason,**r)));subprocess.run(['systemctl','--user','stop',unit],check=False);return
