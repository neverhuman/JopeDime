"""Own H345 resource guard, one admission probe and pre-fit forecast; no reprobe."""
from pathlib import Path
import datetime,json,os,subprocess,threading,time,sys
DATA=Path('/home/ubuntu/jope-data/h345-solhwv');P=Path(__file__).resolve().parent.parent
UNIT='h345r-fetch.service' if sys.argv[1]=='fetch' else 'h345r-run.service'
def cpu():
 a=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(a[:8]),a[3]+a[4]
def monitor():
 prev=cpu()
 while True:
  time.sleep(5);total,idle=cpu();usage=100*(1-(idle-prev[1])/(total-prev[0]));prev=(total,idle);mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;size=0
  for p in DATA.rglob('*'):
   try:
    if p.is_file():size+=p.stat().st_size
   except FileNotFoundError:pass
  utc=datetime.datetime.now(datetime.timezone.utc);r=dict(utc=utc.isoformat(),cpu_percent=usage,load1=os.getloadavg()[0],mem_available_bytes=mem,scratch_bytes=size)
  with (DATA/'resource.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  reason=[]
  if usage>85:reason.append('host CPU >85%')
  if r['load1']>85:reason.append('host load >85')
  if mem<20*2**30:reason.append('MemAvailable <20GiB')
  if size>2_000_000_000:reason.append('scratch >2GB')
  if utc>=datetime.datetime(2026,11,5,5,15,tzinfo=datetime.timezone.utc):reason.append('compute deadline')
  if reason:
   (DATA/'STOP.json').write_text(json.dumps(dict(reason=reason,**r)));subprocess.run(['systemctl','--user','stop',UNIT],check=False);return
threading.Thread(target=monitor,daemon=True).start()
if sys.argv[1]=='fetch':raise SystemExit(subprocess.run([sys.executable,str(P/'source/fetch_base_h345.py')]).returncode)
from run_h345 import durable
from admission_h345 import probe,forecast
for phase in ['preflight','regress','fit','score']:
 command=[sys.executable,str(P/'input_preflight.py')] if phase=='preflight' else [sys.executable,str(P/'source/run_h345.py'),phase]
 if phase=='fit':
  assert not (DATA/'cpu_admission.json').exists(),'ONE admission only';admission=probe();durable(DATA/'cpu_admission.json',admission)
  if admission['status']!='PASS':raise RuntimeError('INCOMPLETE_CPU_ADMISSION; no reprobe/treatment')
  chosen=admission['admitted'];reference=json.loads((P/'h338-runtime-reference.json').read_text());projection=forecast(reference['cpu_capacity_upper_bound_seconds'],len(chosen),datetime.datetime.now(datetime.timezone.utc));durable(DATA/'launch_forecast.json',dict(status='PASS' if projection['fits'] else 'INCOMPLETE_RUNTIME_FORECAST',reference=reference,**projection))
  if not projection['fits']:raise RuntimeError('INCOMPLETE_RUNTIME_FORECAST; no treatment')
  for tid in os.listdir('/proc/self/task'):os.sched_setaffinity(int(tid),chosen)
  durable(DATA/'pre_fit_resource.json',dict(eligible_cores=chosen,idle_percent=admission['idle_percent'],probes=1))
  command=['/usr/bin/taskset','-c',','.join(map(str,chosen)),*command];subprocess.run(command,check=True,env=dict(os.environ,H345_WORKERS=str(len(chosen))))
 else:
  if phase=='score' and datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,11,5,5,10,tzinfo=datetime.timezone.utc):raise RuntimeError('INCOMPLETE: insufficient full scoring window')
  subprocess.run(command,check=True)
