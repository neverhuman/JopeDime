"""Bounded read-only host/cgroup/GPU telemetry with append-only chained records."""
import argparse,datetime,hashlib,json,os,subprocess,time,sqlite3
from pathlib import Path

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def utc(ns):return datetime.datetime.fromtimestamp(ns//10**9,datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')+'.'+str(ns%10**9).zfill(9)+'Z'
def read(path):
 try:return Path(path).read_text()
 except(FileNotFoundError,PermissionError,ProcessLookupError):return None

def values(path):
 s=read(path)
 if s is None:return None
 return {x.split()[0]:int(x.split()[1])for x in s.splitlines()if len(x.split())==2}

def sample_unit(name,path):
 p=Path('/sys/fs/cgroup')/path.lstrip('/');out={'unit':name,'path':path}
 for name in ('cpu.stat','memory.events'):out[name]=values(p/name)
 ms=values(p/'memory.stat');out['memory.stat']=None if ms is None else{k:ms.get(k)for k in ('anon','file','pagetables','pgfault','pgmajfault','workingset_refault_file','unevictable','slab')}
 for name in ('cpu.max','memory.current','memory.peak','memory.swap.current','memory.swap.peak','pids.current','io.stat','cpu.pressure','memory.pressure','io.pressure'):out[name]=read(p/name)
 return out

def units():
 r=subprocess.run(['systemctl','--user','list-units','--type=service','--state=running','--plain','--no-legend'],text=True,capture_output=True,timeout=5)
 names=[x.split()[0]for x in r.stdout.splitlines()if x.split()and x.split()[0].startswith(('jope','grok'))]
 out={}
 for name in names[:128]:
  r=subprocess.run(['systemctl','--user','show',name,'--property=ControlGroup','--value'],text=True,capture_output=True,timeout=3)
  if r.returncode==0 and r.stdout.strip():out[name]=r.stdout.strip()
 return out

class Writer:
 def __init__(self,root,max_bytes):
  self.root=root;self.maximum=max_bytes;self.total=0;self.seq=0;self.previous='0'*64;self.part=0;self.stream=None;self.partbytes=0
 def append(self,payload):
  self.seq+=1;now=time.time_ns();row={'schema':'jope.hpc-host-event/v1','seq':self.seq,'previous':self.previous,'observed_utc':utc(now),'observed_unix_ns':now,'observed_monotonic_ns':time.monotonic_ns(),'host':os.uname().nodename,'pid':os.getpid(),'payload':payload};body=canonical(row);self.previous=sha(body);line=canonical({**row,'sha256':self.previous})+b'\n'
  if self.total+len(line)>self.maximum:raise RuntimeError('fixed telemetry byte budget exhausted')
  if self.stream is None:self.stream=(self.root/f'events-{self.part:05d}.jsonl').open('xb');self.partbytes=0
  self.stream.write(line);self.stream.flush();os.fsync(self.stream.fileno());self.partbytes+=len(line);self.total+=len(line)
  if self.partbytes>=8*1024**2:self.seal()
 def seal(self):
  if self.stream is None:return
  name=Path(self.stream.name);self.stream.close();body=name.read_bytes();receipt={'schema':'jope.hpc-host-log-segment/v1','path':str(name),'sha256':sha(body),'bytes':len(body),'last_seq':self.seq,'event_head':self.previous,'at_unix_ns':time.time_ns()}
  with name.with_suffix('.sealed.json').open('xb')as f:f.write(canonical(receipt));f.flush();os.fsync(f.fileno())
  self.stream=None;self.part+=1

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args();root=Path(a.out);assert root.is_absolute()and all(not x.is_symlink()for x in(root,*root.parents));root.mkdir(exist_ok=False)
 writer=Writer(root,2*1024**3);unitpaths={};last_units=0;lastpublication=0.;lastsequence=-1;interval=2.;start=time.monotonic();cpu_previous=None;mono_previous=None;last_seal=time.monotonic()
 clock=subprocess.run(['timedatectl','show','--property=NTPSynchronized','--property=Timezone'],text=True,capture_output=True,timeout=5)
 writer.append({'kind':'observer.start','source_sha256':sha(Path(__file__).read_bytes()),'clock_state':clock.stdout,'clock_state_returncode':clock.returncode,'clock_resolution_s':time.get_clock_info('time').resolution,'monotonic_resolution_s':time.get_clock_info('monotonic').resolution,'sample_interval_s':interval,'deadline_unix':a.deadline,'timestamp_precision_is_not_clock_accuracy':True,'raw_command_lines_collected':False})
 try:
  while time.time()<a.deadline:
   began=time.monotonic_ns()
   if time.monotonic()-last_units>=30:unitpaths=units();last_units=time.monotonic()
   cpu=[int(x)for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]];now=time.monotonic_ns();cpu_delta=None
   if cpu_previous is not None:cpu_delta={'ticks':[x-y for x,y in zip(cpu,cpu_previous)],'interval_ns':now-mono_previous,'clock_ticks_per_second':os.sysconf('SC_CLK_TCK')}
   cpu_previous,mono_previous=cpu,now
   payload={'kind':'host.sample','sample_started_unix_ns':time.time_ns(),'sample_started_monotonic_ns':began,'cpu_count':os.cpu_count(),'load':os.getloadavg(),'cpu_ticks_delta':cpu_delta,'meminfo':read('/proc/meminfo'),'pressure':{k:read('/proc/pressure/'+k)for k in('cpu','memory','io')},'network_counters':read('/proc/net/dev'),'units':[sample_unit(n,p)for n,p in unitpaths.items()]}
   gstart=time.monotonic_ns()
   r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=5)
   payload['gpu']={'sample_text':r.stdout.strip(),'returncode':r.returncode,'query_wall_ns':time.monotonic_ns()-gstart,'fields':['index','uuid','name','gpu_util_pct','memory_util_pct','used_mib','total_mib','power_watts','temperature_c','sm_mhz'],'kind':'sampled_utilization_not_GPU_execution_time'}
   payload['sample_wall_ns']=time.monotonic_ns()-began;writer.append(payload)
   if os.uname().nodename=='xbabe2':
    path=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_evergreen_20261009/data/expansion.sqlite')
    if path.exists():
     con=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.5);con.row_factory=sqlite3.Row
     try:
      rows=con.execute('SELECT v.sequence,v.identity,v.parent,v.ordinal,v.created,p.committed,p.publication_sha FROM variants v JOIN flywheel_s3_publications_v1 p ON p.sequence=v.sequence WHERE p.committed>? ORDER BY p.committed LIMIT 256',(lastpublication,)).fetchall()
      for row in rows:
       d=dict(row);lastpublication=d['committed'];d.update(kind='dataset.publication_observed',generation_to_publication_wall_seconds=d['committed']-d['created'],timestamp_source='original SQLite created/committed floating-point Unix seconds; not reconstructed CPU or encoding duration',historical_catchup=d['committed']<time.time()-interval*2);writer.append(d)
     finally:con.close()
   status={'schema':'jope.hpc-host-status/v1','host':os.uname().nodename,'at_unix_ns':time.time_ns(),'events':writer.seq,'bytes':writer.total,'head_sha256':writer.previous,'last_sample_wall_ns':payload['sample_wall_ns'],'observed_units':list(unitpaths),'state':'running','deadline_unix':a.deadline}
   tmp=root/'status.tmp';tmp.write_bytes(canonical(status));os.replace(tmp,root/'status.json')
   if time.monotonic()-last_seal>=60:writer.seal();last_seal=time.monotonic()
   time.sleep(max(.01,interval-(time.monotonic_ns()-began)/1e9))
 finally:
  try:writer.append({'kind':'observer.stop','elapsed_monotonic_seconds':time.monotonic()-start})
  finally:writer.seal()
if __name__=='__main__':main()
