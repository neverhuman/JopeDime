"""Append-only measured timing. Durations use monotonic_ns; UTC identifies events.
CUDA Event intervals are explicitly distinct from GPU utilization and transfer
profiling. Internal backend HtoD/DToH time is unavailable, never inferred.
"""
import contextlib,json,os,resource,socket,time,threading
from pathlib import Path

def counters(torch=None):
 result={'process_user_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime,'process_system_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_stime}
 try:
  rel=next(l.split(':',2)[2]for l in Path('/proc/self/cgroup').read_text().splitlines()if l.startswith('0::'));root=Path('/sys/fs/cgroup')/rel.lstrip('/')
  result['cgroup_cpu']={k:int(v)for k,v in(line.split()for line in(root/'cpu.stat').read_text().splitlines())}
  result['cgroup_memory']={n:int((root/n).read_text())for n in('memory.current','memory.peak')}
 except(OSError,ValueError,StopIteration):result['cgroup_unavailable']=True
 if torch is not None and torch.cuda.is_initialized():result['cuda_memory']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
 return result

class Logger:
 def __init__(self,path,context):
  self.path=Path(path);self.context={**context,'host':socket.gethostname(),'pid':os.getpid()};self.path.parent.mkdir(parents=True,exist_ok=True)
  if self.path.exists():raise ValueError('fresh timing journal required')
  self.file=self.path.open('xb');self.lock=threading.Lock()
 def start(self,stage,torch=None,**fields):return {'stage':stage,'start_utc_ns':time.time_ns(),'start_monotonic_ns':time.monotonic_ns(),'before':counters(torch),**fields}
 def end(self,event,torch=None,**fields):
  row={'schema':'jope.hpc-timing/v1',**self.context,**event,'end_utc_ns':time.time_ns(),'end_monotonic_ns':time.monotonic_ns(),'after':counters(torch),**fields};row['duration_ns']=row['end_monotonic_ns']-row['start_monotonic_ns'];
  with self.lock:self.file.write(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n');self.file.flush();os.fsync(self.file.fileno())
  return row
 @contextlib.contextmanager
 def span(self,stage,torch=None,**fields):
  event=self.start(stage,torch,**fields)
  try:yield
  except BaseException as e:self.end(event,torch,state='failed',error=type(e).__name__);raise
  else:self.end(event,torch,state='complete')
 def close(self):self.file.close()

class Encoder:
 def __init__(self,encoder,logger,torch):self.encoder,self.logger,self.torch=encoder,logger,torch;self.item=None;self.ordinal=0
 def __getattr__(self,key):return getattr(self.encoder,key)
 def select(self,item):self.item=item;self.ordinal=0
 def one(self,component,data):
  t=self.torch;t.cuda.synchronize();start=t.cuda.Event(enable_timing=True);end=t.cuda.Event(enable_timing=True);i=self.item;kind=['original','repeat','protected-control','target-permutation-control'][self.ordinal];self.ordinal+=1
  event=self.logger.start('kumo_encode_call',t,variant_identity=i['identity'],qualification_call=kind,rows=i['metadata']['rows'],features=i['metadata']['features'],input_bytes=len(i['raw']),host_to_device_duration_ns=None,host_to_device_status='not separately instrumented inside pinned backend')
  start.record()
  try:result=self.encoder.one(component,data);end.record();t.cuda.synchronize();self.logger.end(event,t,state='complete',cuda_event_elapsed_ms=start.elapsed_time(end),cuda_interval_scope='one supervised encoder call including backend transfers and launch intervals');return result
  except BaseException as e:t.cuda.synchronize();self.logger.end(event,t,state='failed',error=type(e).__name__);raise

class Body:
 def __init__(self,body,logger,event):self.body,self.logger,self.event=body,logger,event;self.bytes=0;self.closed=False
 def __enter__(self):return self
 def __exit__(self,*args):self.close();return False
 def read(self,n=-1):
  b=self.body.read(n);self.bytes+=len(b);return b
 def close(self):
  try:self.body.close()
  finally:
   if not self.closed:self.logger.end(self.event,state='body_closed',response_bytes=self.bytes);self.closed=True
class Client:
 def __init__(self,client,logger):self.client,self.logger=client,logger
 def __getattr__(self,name):return getattr(self.client,name)
 def get_object(self,**kwargs):
  event=self.logger.start('s3_get',bucket=kwargs['Bucket'],key=kwargs['Key'],requested_version_id=kwargs.get('VersionId'))
  try:r=self.client.get_object(**kwargs);r['Body']=Body(r['Body'],self.logger,{**event,'response_version_id':r.get('VersionId')});return r
  except BaseException as e:self.logger.end(event,state='failed',error=type(e).__name__);raise
 def put_object(self,**kwargs):
  event=self.logger.start('s3_put',bucket=kwargs['Bucket'],key=kwargs['Key'],request_bytes=len(kwargs['Body'])if isinstance(kwargs.get('Body'),bytes)else None)
  try:r=self.client.put_object(**kwargs);self.logger.end(event,state='complete',version_id=r.get('VersionId'));return r
  except BaseException as e:self.logger.end(event,state='failed',error=type(e).__name__);raise
