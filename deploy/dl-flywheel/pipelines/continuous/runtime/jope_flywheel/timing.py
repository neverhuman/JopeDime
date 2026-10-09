"""Bounded append-only measured stage timing for this owned inference campaign."""
from contextlib import contextmanager
import datetime,json,os,resource,socket,time
from pathlib import Path
ACTIVE=None


def snapshot(group):
    own=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    result={'process_cpu_seconds':own.ru_utime+own.ru_stime,'children_cpu_seconds':child.ru_utime+child.ru_stime,
        'process_user_cpu_seconds':own.ru_utime,'process_system_cpu_seconds':own.ru_stime,
        'children_user_cpu_seconds':child.ru_utime,'children_system_cpu_seconds':child.ru_stime}
    if group:
        result['cgroup_memory_current_bytes']=int((group/'memory.current').read_text())
        result['cgroup_cpu']={k:int(v)for k,v in (line.split()for line in(group/'cpu.stat').read_text().splitlines())}
        io={}
        for line in(group/'io.stat').read_text().splitlines():
            for pair in line.split()[1:]:
                k,v=pair.split('=');io[k]=io.get(k,0)+int(v)
        result['cgroup_io']=io
    return result


def delta(a,b):
    return {k:delta(a.get(k,{}),v)if isinstance(v,dict)else v-a.get(k,0)for k,v in b.items()}


class Timing:
    def __init__(self,path,group=None,**identity):
        self.path=Path(path);self.group=group;self.identity={'host':socket.gethostname(),'pid':os.getpid(),**identity}
        self.count=0;self.bytes=0;self.context={};self.path.open('xb').close()
    def event(self,kind,stage,**extra):
        ns=time.time_ns();row={'schema':'jope.flywheel-stage-timing/v1','sequence':self.count+1,'utc_time_ns':ns,
            'utc':datetime.datetime.fromtimestamp(ns//10**9,datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')+f'.{ns%10**9:09d}Z',
            'monotonic_ns':time.monotonic_ns(),'kind':kind,'stage':stage,**self.identity,**self.context,**extra}
        raw=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        if self.count>=1000000 or self.bytes+len(raw)>2*1024**3:raise ValueError('timing journal bound reached; no silent truncation')
        with self.path.open('ab')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        self.count+=1;self.bytes+=len(raw)
    @contextmanager
    def bind(self,**context):
        old=self.context;self.context={**old,**context}
        try:yield
        finally:self.context=old
    @contextmanager
    def span(self,stage,**extra):
        before=snapshot(self.group);started=time.monotonic_ns();self.event('start',stage,resources=before,**extra);error=None
        try:yield
        except BaseException as e:error=type(e).__name__+': '+str(e)[:500];raise
        finally:
            after=snapshot(self.group);self.event('end',stage,duration_monotonic_ns=time.monotonic_ns()-started,
                resource_delta=delta(before,after),resources=after,error=error,**extra)


@contextmanager
def span(stage,**extra):
    if ACTIVE is None:yield
    else:
        with ACTIVE.span(stage,**extra):yield


def native_run(command,directory,name,deadline,timeout):
    from .program_io import run
    with span('native.'+name,command=command):return run(command,directory,name,deadline,timeout)


class CUDAForwards:
    """CUDA events around each actual forward; sum excludes between-call waits."""
    def __init__(self,net):self.net=net;self.pairs=[];self.handles=[];self.enabled=next(net.parameters()).is_cuda
    def __enter__(self):
        if self.enabled:
            import torch
            def before(module,args):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();self.pairs.append((a,b))
            def after(module,args,result):self.pairs[-1][1].record()
            self.handles=[self.net.register_forward_pre_hook(before),self.net.register_forward_hook(after)]
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
    def receipt(self):
        if not self.enabled:return {'cuda_event_model_forward_ms':None,'model_forwards_timed':0}
        import torch
        torch.cuda.synchronize()
        return {'cuda_event_model_forward_ms':sum(a.elapsed_time(b)for a,b in self.pairs),'model_forwards_timed':len(self.pairs),
            'timing_scope':'Sum of CUDAEvent elapsed intervals bracketing each model forward; separate from complete decoding wall time.'}
