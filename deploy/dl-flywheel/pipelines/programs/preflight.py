"""Owned CPU-only qualification of copied native tools and112 exact inputs; no fits/PUT."""
import json,hashlib,os,socket,sys,time,subprocess
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'runtime')]
from jope_flywheel.program_io import read,bound,need,create,sha,tick,NativeTools,run
from jope_flywheel.protocol import canonical,digest
from jope_flywheel.s3_library import client
from staged_inputs import StagedInputs

def streaming(ref):
 p=Path(ref['path']);need(p.is_absolute()and all(not q.is_symlink()for q in(p,*p.parents))and p.is_file(),'native regular owned file')
 before=p.stat();need(before.st_size==ref['bytes'],'native size');h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 after=p.stat();need((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)and h.hexdigest()==ref['sha256'],'native closure hash')

def resources(m):
 p=m['resource_policy'];mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()};disk=os.statvfs(ROOT);free=disk.f_bavail*disk.f_frsize
 need(mem['MemAvailable']>=p['minimum_mem_available_bytes'] and mem['MemAvailable']-p['aggregate_memory_bytes']>=p['foreign_memory_reserve_bytes'],'leave24GiB foreign memory reserve')
 need(os.getloadavg()[0]<=p['max_host_load']and free>=p['minimum_free_disk_bytes'],'host load/disk gate')
 return {'memory_available':mem['MemAvailable'],'load1':os.getloadavg()[0],'disk_free':free}

def main():
 b=read(ROOT/'campaign.json');m=json.loads(b);msha=sha(b);deadline=min(m['deadline_unix']-300,time.time()+600)
 need(socket.gethostname().split('.')[0]==m['host']=='xbabe3'and m['owned_root']==str(ROOT),'owned host/root')
 need(m['created_unix']<=time.time()<m['deadline_unix']and m['deadline_unix']-m['created_unix']==7200,'fixed window')
 original=json.loads(bound(m['original_zero_attempt_proof']))
 need(original['closed'] is True and original['all_native_attempts']==0 and original['all_result_puts']==0 and original['original_campaign_sha256']==m['recovery_of_campaign_sha256'],'closed original zeroattempt recovery authority')
 for i,shard in original['shards'].items():
  for ref in shard['files'].values():bound(ref)
  unit='jopedime-gp112-shard'+i+'-20261009-v1';state=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState'],capture_output=True,text=True,timeout=10)
  need(state.returncode==0 and state.stdout.strip()=='ActiveState=inactive','original shard still active')
 for ref in m['sources'].values():bound(ref)
 for ref in m['runtime_pins']:bound(ref)
 closure=json.loads(bound(m['native_closure']));need(len(closure['files'])==92,'native closure92')
 for ref in closure['files']:
  tick(deadline);streaming(dict(ref,path=str(Path(m['native_root'])/ref['relative'])))
 res=resources(m);dest=ROOT/'preflight';dest.mkdir(mode=0o700);proofs=[]
 import numpy,boto3
 need({'python':sys.version.split()[0],'numpy':str(numpy.__version__),'boto3':str(boto3.__version__)}=={'python':'3.12.3','numpy':'1.26.4','boto3':'1.35.49'},'runtime exact versions')
 for shard in range(2):
  c=json.loads(read(ROOT/f'shard{shard}/config.json'));need(c['campaign_sha256']==msha and len(c['variants'])==56,'shard campaign closure')
  out=dest/f'shard{shard}';out.mkdir();(out/'data').mkdir()
  inputs=StagedInputs(dict(c,owned_root=str(out)),json.loads(bound(c['registry'])),NativeTools(c['native_canonicalizer'],deadline),client(c['aws_profile']),deadline)
  for v in c['variants']:
   ctx,raw,desc=inputs.describe(v['identity']);need(desc['rows']==v['rows']and desc['features']==v['features']and desc['partition']=='train'and ctx.target_sha256==v['target_sha256']and ctx.dataset_sha256==v['content_sha256'],'exact input replay shape/context')
   proofs.append({'identity':v['identity'],'context_id':ctx.identity,'descriptor_sha256':digest(desc),'rows':desc['rows'],'features':desc['features'],'inner_counts':desc['inner_counts'],'partition':desc['partition']})
  for program in c['programs']:need(inputs.native.canonical_program(program['dsl'])==program['dsl'],'canonical fixed panel')
  inputs.lease.check()
  # Invoke the real worker budget/claim path with an emptyownedqueue: nofit or PUT.
  from jope_flywheel.queue import Queue
  from jope_flywheel.program_worker import ProgramWorker,evaluation_limit
  need(evaluation_limit(c)==448,'real worker exactplannedcapacity')
  with Queue(out/'contract-queue.sqlite',max_jobs=896)as q:
   need(q.max_jobs>=2*len(c['variants'])*len(c['programs']),'queue capacity covers evaluations and publication')
   worker=ProgramWorker(q,inputs,None,inputs.native,c,out,deadline,sha(read(ROOT/f'shard{shard}/config.json')))
   need(worker.once()=={'state':'idle'},'actual worker budget/queue invocation contract')
 # Relocated exact compiler must find its authenticated own sysroot/libs, not a shareddifferent compiler.
 compile_dir=dest/'compile';compile_dir.mkdir();source=compile_dir/'main.rs';create(source,b'fn main(){println!("qualified-native-replica");}\n')
 run([c['rustc']['path'],str(source),'-O','-o',str(compile_dir/'proof')],compile_dir,'rustc',deadline,120)
 run([str(compile_dir/'proof')],compile_dir,'execute',deadline,10)
 need(len(proofs)==len({x['identity']for x in proofs})==112 and set(x['identity']for x in proofs)==set(v['identity']for v in m['variants']),'full112 replay support')
 tick(deadline);res=resources(m)
 result={'schema':'jope.flywheel-program112-preflight/v1','passed':True,'campaign_sha256':msha,'checked_at':time.time(),'resources':res,'replayed':112,'proofs':proofs,'native_files_verified':92,'actual_worker_contracts_passed':2,'native_gp_fits':0,'s3_puts':0,'teaching_eligible':False}
 create(ROOT/'preflight.json',canonical(result));print(json.dumps({'passed':True,'replayed':112,'campaign_sha256':msha}))
if __name__=='__main__':main()
