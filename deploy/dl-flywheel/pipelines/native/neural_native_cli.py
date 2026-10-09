"""One source-pinned full native corpus campaign; no foreign source or S3 writes."""
import hashlib,json,os,socket,sys,time
from pathlib import Path
sys.dont_write_bytecode=True

def need(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path,cap=64*1024**2):
 p=Path(path);need(p.is_absolute()and'..'not in p.parts and all(not q.is_symlink()for q in(p,*p.parents))and p.is_file(),'absolute non-symlink file')
 need(0<p.stat().st_size<=cap,'bounded file')
 with p.open('rb')as f:b=f.read(cap+1)
 need(len(b)<=cap,'file grew beyond cap');return b

def validate_permit(config,spec):
 p=config['research_optimizer_admission'];need(set(p)=={'schema','approved_for_bounded_research','production_admission','snapshot_sha256','objective_sha256','source_authority_sha256','input_roster','maximum_updates','maximum_epochs','model_initialization','generated_outcomes_used_as_reward','native_hwm_promotion'},'closed full-native research permit')
 canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
 need(p['schema']=='jope.native-research-optimizer-admission/v1'and p['approved_for_bounded_research']is True and p['production_admission']is False and p['native_hwm_promotion']is False,'explicit research-only authority')
 need(p['snapshot_sha256']==config['snapshot']['sha256']and p['objective_sha256']==sha(canonical(config['neural']['objective']))and p['source_authority_sha256']==sha(canonical(spec['authority_pins'])),'exact research data/objective/authority binding')
 need(p['input_roster']=={'kumo_tabular_s':2051}and p['maximum_updates']==config['neural']['max_updates']and p['maximum_epochs']==config['neural']['max_epochs']and p['model_initialization']=='fresh'and p['generated_outcomes_used_as_reward']is False,'bounded fresh model scope; unmeasured outputs never reward')
 return p

def main():
 root=Path(__file__).resolve().parent;raw=read(root/'config.json');c=json.loads(raw);pin=sha(raw);window=json.loads(read(root/'window.json'));now=time.time();mono=time.monotonic()+max(0,window['deadline_unix']-now)
 need(c['schema']=='jope.native-neural-campaign/v1'and c['host']=='xbabe2'and socket.gethostname().split('.')[0]=='xbabe2'and c['owned_root']==str(root),'owned explicit full-native campaign')
 need(window['config_sha256']==pin and window['created_unix']<=now<window['deadline_unix']and window['deadline_unix']-window['created_unix']==c['duration_seconds'],'fixed original window')
 for rel,h in c['owned_sources'].items():
  q=Path(rel);need(not q.is_absolute()and'..'not in q.parts,'source-relative closure');need(sha(read(root/q))==h,'owned source drift:'+rel)
 cg=next(x for x in Path('/proc/self/cgroup').read_text().splitlines()if x.startswith('0::'));group=Path('/sys/fs/cgroup')/cg[3:].lstrip('/');quota,period=(group/'cpu.max').read_text().split()
 need(quota!='max'and int(quota)<=8*int(period)and int((group/'memory.max').read_text())<=8*1024**3 and int((group/'memory.swap.max').read_text())==0 and int((group/'pids.max').read_text())<=64,'CPU8/RAM8GiB/swap0/tasks64 resource boundary')
 need(os.getpriority(os.PRIO_PROCESS,0)>=10 and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'and os.environ.get('TMPDIR')==str(root/'tmp'),'owned one-GPU runtime')
 need(all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'single-thread BLAS')
 sys.path[:0]=[str(root/'consumer'),str(root/'runtime')]
 import numpy as np,torch,safetensors
 actual={'python':sys.version.split()[0],'numpy':str(np.__version__),'torch':str(torch.__version__),'safetensors':str(safetensors.__version__)};need(actual==c['runtime_versions'],'exact numeric runtime')
 from jope_flywheel import training_snapshot_native as N
 from jope_flywheel.program_io import bind_campaign_deadline,tick,create
 from jope_flywheel.protocol import canonical
 import native_corpus_authority as native_authority
 from neural_native_train import run,OBJECTIVE,publish
 deadline=window['deadline_unix']-c['closure_reserve_seconds'];bind_campaign_deadline(deadline);bind_campaign_deadline(time.time()+max(0,mono-time.monotonic()-c['closure_reserve_seconds']))
 specraw=read(c['snapshot']['path']);need(sha(specraw)==c['snapshot']['sha256']and len(specraw)==c['snapshot']['bytes'],'exact full-native spec');spec=json.loads(specraw);validate_permit(c,spec)
 need(c['neural']['objective']==OBJECTIVE and c['neural']['max_seq']==640 and c['neural']['batch_size']==16 and c['neural']['max_updates']==512 and c['neural']['max_epochs']==1 and c['neural']['max_pending_emissions']==2048,'reviewed full-pass count/shape caps')
 store=N.LocalStore(c['read_roots']);authority=canonical(native_authority.check(spec))
 need(type(authority)is bytes,'canonical authority bytes required');loaded=N.load_snapshot(specraw,sha(specraw),store,authority_bytes=authority)
 need(loaded.receipt['counts']['loaded_kumo_contexts']==c['expected_contexts']and loaded.receipt['counts']['ce_winner_contexts']==c['expected_contexts']and loaded.receipt['proposal_programs']==c['expected_unique_teacher_programs'],'actual full selected teacher coverage')
 create(root/'initial-role-proof.json',authority);create(root/'runtime-proof.json',canonical({'versions':actual,'at':time.time(),'cuda_workspace':os.environ['CUBLAS_WORKSPACE_CONFIG'],'config_sha256':pin,'consumer_origins':{k:v.__file__ for k,v in sys.modules.items()if k.startswith('dltrain.')and hasattr(v,'__file__')}}))
 need(torch.cuda.is_available()and torch.cuda.device_count()==1,'one available CUDA device')
 def fresh():
  tick(deadline);proof=canonical(native_authority.check(spec));doc=json.loads(proof);need(doc['passed']is True and doc['source_pins']==spec['authority_pins']and 0<=time.time()-doc['checked_at_unix']<=60,'fresh exact source/current-role proof');publish(root,'latest-role-proof.json',doc)
 result=run(loaded,c['neural'],output=root/'training',deadline=deadline,authority_check=fresh,campaign_sha256=pin)
 publish(root,'status.json',{'schema':'jope.native-neural-campaign-status/v1','config_sha256':pin,'state':result['state'],'optimizer_steps':result['optimizer_steps'],'at':time.time(),'training_root':str(root/'training'),'generated_program_measurements':'See durable outbox; pending records are not numeric results.'})

if __name__=='__main__':
 try:main()
 except BaseException as error:
  root=Path(__file__).resolve().parent;body=json.dumps({'schema':'jope.native-neural-fatal/v1','at':time.time(),'error':type(error).__name__+': '+str(error)[:4000],'automatic_retry':False},sort_keys=True).encode()
  try:
   with(root/'fatal.json').open('xb')as f:f.write(body);f.flush();os.fsync(f.fileno())
  except BaseException:pass
  raise
