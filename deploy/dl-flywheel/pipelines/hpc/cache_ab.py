"""Read-only matched verification benchmark against two actually scored native GPs."""
import hashlib,json,resource,sys,time
from pathlib import Path

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 root=Path(__file__).resolve().parent;c=json.loads((root/'config.json').read_bytes())
 for ref in c['sources']:
  p=Path(ref['path']);assert not p.is_symlink() and sha(p.read_bytes())==ref['sha256']
 sys.path[:0]=[str(root),c['native_root']]
 from native_evaluate import NativeLibrary,verified_measurement
 from native_inputs import OriginalScoreContext
 from jope_flywheel.program_io import NativeTools,bind_campaign_deadline
 from jope_flywheel.s3_library import client
 from verified_cache import CachedS3Library,CachedNativeTools
 deadline=c['deadline_unix'];bind_campaign_deadline(deadline)
 class CachedLibrary(NativeLibrary,CachedS3Library):pass
 assert CachedLibrary.get_version is CachedS3Library.get_version
 trials=[];expected=None
 for mode in ['uncached','cached']:
  native=(NativeTools if mode=='uncached'else CachedNativeTools)(c['canonicalizer'],deadline)
  class CountClient:
   def __init__(self):self.inner=client();self.counts={'get_object':0,'head_object':0};self.seconds={'get_object':0.,'head_object':0.}
   def __getattr__(self,name):
    if name not in self.counts:raise ValueError('Read-only cache A/B refuses:'+name)
    def run(**kw):
     assert time.time()<deadline;start=time.monotonic();self.counts[name]+=1
     try:return getattr(self.inner,name)(**kw)
     finally:self.seconds[name]+=time.monotonic()-start
    return run
  transport=CountClient();store=(NativeLibrary if mode=='uncached'else CachedLibrary)(transport,deadline=deadline,nativecanonical_json=native.canonical_json)
  for repeat in [0,1]:
   before=resource.getrusage(resource.RUSAGE_SELF);child0=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic_ns();prior=dict(transport.counts);results=[]
   for outcome in c['outcomes']:
    raw,_=store.get_version(outcome['record_ref']);entry=json.loads(raw);context=OriginalScoreContext(**entry['context'])
    program=store.get_version(entry['artifacts']['program'])[0].decode();assert native.canonical_program(program)==program
    actual=verified_measurement(store,context,program,entry);assert actual==outcome['measurement'];results.append(actual)
   after=resource.getrusage(resource.RUSAGE_SELF);child1=resource.getrusage(resource.RUSAGE_CHILDREN)
   if expected is None:expected=results
   assert results==expected
   trials.append({'mode':mode,'repeat':repeat,'finished_utc_ns':time.time_ns(),'wall_seconds':(time.monotonic_ns()-start)/1e9,'cpu_user_seconds':after.ru_utime-before.ru_utime,'cpu_system_seconds':after.ru_stime-before.ru_stime,'child_cpu_user_seconds':child1.ru_utime-child0.ru_utime,'child_cpu_system_seconds':child1.ru_stime-child0.ru_stime,'requests':{k:transport.counts[k]-prior[k]for k in prior},'identical_verified_measurements':True,'cache_stats':store.cache_stats()if mode=='cached'else None,'native_cache_stats':native.cache_stats()if mode=='cached'else None})
  assert time.time()<deadline
 result={'schema':'jope.hpc-cache-matched-ab/v1','at':time.time(),'config_sha256':sha((root/'config.json').read_bytes()),'trials':trials,'context_count':2,'read_only':True,'scope':'Matched complete immutable result verification, not end-to-end fresh execution throughput; sequential cold/warm pairs, not randomized multi-run inference.'}
 with(root/'result.json').open('x')as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
