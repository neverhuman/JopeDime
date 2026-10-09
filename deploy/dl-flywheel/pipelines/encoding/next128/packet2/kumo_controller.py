"""Independent Kumo Small queue consumer. New disjoint1–32 packet per window.

The local controller holds no raw corpus archive, streams immutable packet files
between hosts, and never starts a second packet until S3 verification and its DB
stage transaction complete. Unknown work is preserved; there is no auto retry.
"""
import argparse,base64,fcntl,hashlib,json,math,os,shlex,subprocess,sys,time
from pathlib import Path
from campaign import PACKET_ID,ROOT,HOST,COUNT,GLOBAL_DEADLINE,DB_CONFIG_SHA,RUNTIME_SHA
HERE=Path(__file__).resolve().parent
W='/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final'
R=W+'/flywheel_kumo_fresh32_xbabe3_20261009'
SOURCE_ROOT='/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-kumo-xbabe3-next128-20261009/revision2/packet2/runtime'

def need(ok,msg):
 if not ok:raise ValueError(msg)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def ssh(host,args,body=None,timeout=60):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(args)],input=body,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
 need(p.returncode==0,host+': '+p.stderr.decode()[-3500:]);need(len(p.stdout)<=16*1024**2,'bounded SSH output');return p.stdout

def copy(host,root,files):
 need(sum(map(len,files.values()))<=16*1024**2,'small deployment batch')
 script="""import json,base64,hashlib,os,sys
from pathlib import Path
root=Path(sys.argv[1]);out={}
for rel,v in json.load(sys.stdin).items():
 p=root/rel;assert not Path(rel).is_absolute()and'..'not in Path(rel).parts and all(not q.is_symlink()for q in(p,*p.parents));b=base64.b64decode(v['body']);assert hashlib.sha256(b).hexdigest()==v['sha256'];p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b
 else:
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 out[rel]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':len(b)}
print(json.dumps(out))
"""
 return json.loads(ssh(host,['python3','-B','-c',script,root],canonical({n:{'body':base64.b64encode(b).decode(),'sha256':sha(b)}for n,b in files.items()})))

def bounded(host,unit,script,args,seconds,*,gpu=False,wait=True):
 need(0<seconds<=1800,'bounded runtime');cmd=['systemd-run','--user','--collect','--unit='+unit,'-p','CPUQuota='+('800%'if gpu else'100%'),'-p','MemoryMax='+('12G'if gpu else'512M'),'-p','MemorySwapMax=0','-p','TasksMax='+('128'if gpu else'32'),'-p','Nice='+('10'if gpu else'19'),'-p','KillMode=control-group','-p','RuntimeMaxSec='+str(seconds),'--setenv=CUDA_VISIBLE_DEVICES='+('0'if gpu else''),'--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1']
 if wait:cmd+=['--wait','--pipe']
 cmd+=['/usr/bin/python3','-B',script,*args];return cmd

def fresh_authority(root,msha,remaining):
 cmd=bounded('xbabe2','jope-kumo3fresh-role-'+str(time.time_ns()),root+'/evergreen_kumos_packet_authority.py',['--manifest',root+'/manifest.json','--manifest-sha',msha],min(60,remaining()))
 raw=ssh('xbabe2',cmd,timeout=min(65,remaining()+5));proof=json.loads(raw.splitlines()[-1]);need(proof['state']=='passed'and proof['manifest_sha256']==msha,'current authority failed')
 ahead=proof['checked_at_unix']+.1-time.time();need(ahead<=1,'authority clock skew');
 if ahead>0:time.sleep(ahead)
 need(0<=time.time()-proof['checked_at_unix']<30,'fresh authority age');body=canonical(proof);ref={'path':root+'/authority/'+sha(body)+'.json','sha256':sha(body),'bytes':len(body)};copy('xbabe3',root,{'authority/'+sha(body)+'.json':body});return ref,proof

def transfer(intake,common,remaining):
 index=intake['index'];deadline=intake['queue_deadline_unix'];args=['--index-sha',index['sha256'],'--deadline',str(deadline)];send=bounded('xbabe2','jope-kumo3fresh-send-'+intake['packet_id'][:12],common+'/kumo_transport.py',['send','--index',index['path'],*args],remaining());receive=bounded('xbabe3','jope-kumo3fresh-receive-'+intake['packet_id'][:12],common+'/kumo_transport.py',['receive',*args],remaining())
 # Only the pipe passes payload bytes through this controller. No local tarball.
 a=subprocess.Popen(['ssh','-o','BatchMode=yes','xbabe2',shlex.join(send)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 b=subprocess.Popen(['ssh','-o','BatchMode=yes','xbabe3',shlex.join(receive)],stdin=a.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE);a.stdout.close()
 try:
  out,err=b.communicate(timeout=remaining()+5);_,ae=a.communicate(timeout=min(10,remaining()));need(a.returncode==b.returncode==0,'packet transfer failed '+(ae+err).decode()[-2000:]);need(len(out)<16384,'bounded stage receipt');return json.loads(out.splitlines()[-1])
 finally:
  for child in(a,b):
   if child.poll()is None:child.kill();child.wait()

def save(root,name,value):
 p=root/(name+'.json')
 with p.open('xb')as f:f.write(canonical(value))
 return {'path':str(p),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}

def resource_precheck(common,config):
 script="""import importlib.util,sys,json,hashlib
from pathlib import Path
p=Path(sys.argv[1]);assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2];s=importlib.util.spec_from_file_location('rolling_resource',p);w=importlib.util.module_from_spec(s);s.loader.exec_module(w);ok,reasons,observed=w.Admission([]).check(start=True);print(json.dumps({'passed':ok,'reasons':reasons,'observed':observed}))
"""
 return json.loads(ssh('xbabe3',['/home/ubuntu/jopedime_probe_exp/ofm-20261008/venv/bin/python','-B','-c',script,common+'/evergreen_kumos_packet.py',config['worker']['sha256']]))

def packet(config,pin,common,local):
 resource=resource_precheck(common,config)
 if not resource['passed']:return {'state':'resource_wait','resource':resource}
 start=config['created_unix'];deadline=config['deadline_unix'];need(start<=time.time()<deadline<=GLOBAL_DEADLINE,'fixed unchanged packet window');mono=config['shared_monotonic_deadline'];need(time.monotonic()<mono,'unchanged shared monotonic deadline')
 def remaining():
  seconds=math.floor(min(deadline-time.time(),mono-time.monotonic()));need(seconds>5,'unchanged packet deadline');return seconds
 ident=PACKET_ID;folder=local/ident;folder.mkdir();save(folder,'window',{'created_unix':start,'deadline_unix':deadline,'packet_id':ident})
 need(remaining()>=660,'shared remaining before actual intake must reserve660 seconds');cmd=bounded('xbabe2','jope-kumo3fresh-intake-'+ident[:12],common+'/kumo_intake.py',['--config',common+'/config.json','--config-sha',pin,'--packet-id',ident,'--deadline',str(deadline)],min(180,remaining()));save(folder,'intake-intent',{'command':cmd});intake=json.loads(ssh('xbabe2',cmd,timeout=min(185,remaining()+5)).splitlines()[-1]);save(folder,'intake',intake)
 if intake['state']in('empty','blocked_claim','packet_already_exists'):return intake
 need(intake['state']=='prepared','intake not prepared');stage=transfer(intake,common,remaining);save(folder,'stage',stage);root=stage['root'];mref=intake['manifest'];mraw=ssh('xbabe2',['cat',mref['path']]);need(sha(mraw)==mref['sha256'],'manifest source');m=json.loads(mraw);save(folder,'manifest',m)
 ref,proof=fresh_authority(root,mref['sha256'],remaining);save(folder,'preflight-role',{'ref':ref,'proof':proof});args=['--manifest',mref['path'],'--manifest-sha',mref['sha256'],'--host','xbabe3','--authority-path',ref['path'],'--authority-sha',ref['sha256']]
 pre=bounded('xbabe3','jope-kumo3fresh-preflight-'+ident[:12],root+'/evergreen_kumos_packet.py',[*args,'--preflight-only'],min(180,remaining()),gpu=True);pre[pre.index('--setenv=CUDA_VISIBLE_DEVICES=0')]='--setenv=CUDA_VISIBLE_DEVICES='
 save(folder,'preflight-intent',{'command':pre});raw=ssh('xbabe3',pre,timeout=min(185,remaining()+5));receipt=json.loads(raw.splitlines()[-1]);need(receipt['state']=='preflight_pass'and receipt['manifest_sha256']==mref['sha256'],'actual CPU preflight required');save(folder,'preflight',receipt)
 ref,proof=fresh_authority(root,mref['sha256'],remaining);args[-3:]=[ref['path'],'--authority-sha',ref['sha256']];run=bounded('xbabe3','jope-kumo3fresh-rolling-'+ident[:12],root+'/evergreen_kumos_packet.py',args,min(remaining(),max(1,math.floor(m['deadline_unix']-time.time()))),gpu=True)
 save(folder,'launch-intent',{'command':run,'manifest':mref,'authority':ref});raw=ssh('xbabe3',run,timeout=remaining()+5);receipt=json.loads(raw.splitlines()[-1]);need(receipt['state']=='complete','encoder incomplete');save(folder,'encoded',receipt)
 role,proof=fresh_authority(root,mref['sha256'],remaining);pubstart=time.time();pub={'schema':'jope.flywheel-kumos-fresh32-publication/v1','host':'xbabe3','created_unix':pubstart,'deadline_unix':min(deadline,pubstart+600),'helper_sha256':config['publisher']['sha256'],'encoder_worker':{'path':root+'/evergreen_kumos_packet.py','sha256':m['worker_sha256']},'encoder_manifest':mref,'prior_publication':m['dataset_publication'],'authority':role,'out':root+'/publication/xbabe3','expected_variants':m['total_variants']}
 # Exact seals/status are generated by the immutable worker verifier, not guessed.
 script="""import importlib.util,json,sys,hashlib
from pathlib import Path
root=Path(sys.argv[1]);msha=sys.argv[2];raw=(root/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==msha;m=json.loads(raw);assert hashlib.sha256((root/'evergreen_kumos_packet.py').read_bytes()).hexdigest()==m['worker_sha256'];s=importlib.util.spec_from_file_location('w',root/'evergreen_kumos_packet.py');w=importlib.util.module_from_spec(s);s.loader.exec_module(w);w.ROOT=root;r=w.verify_campaign_seal(m,m['hosts']['xbabe3'],'xbabe3',msha);p=w.stage_seal(root/'runs/xbabe3/ofm',m,'xbabe3',msha,'ofm');print(json.dumps({'ledger':r['ledger'],'status_sha256':r['status_sha256'],'phase_ledgers':{'ofm':p}}))
"""
 seals=json.loads(ssh('xbabe3',['python3','-B','-c',script,root,mref['sha256']]));pub.update(seals);pubbody=canonical(pub);copy('xbabe3',root,{'publisher/config.json':pubbody,'publisher/kumo_publish.py':Path(config['local_publisher']).read_bytes(),'publisher/pilot_publish_core.py':Path(config['local_base']).read_bytes(),'publisher/campaign.py':(HERE/'campaign.py').read_bytes()});save(folder,'publication-config',pub)
 args=['--config',root+'/publisher/config.json','--config-sha',sha(pubbody)];cmd=bounded('xbabe3','jope-kumo3fresh-publish-'+ident[:12],root+'/publisher/kumo_publish.py',args,min(600,remaining()));save(folder,'publication-intent',{'command':cmd});raw=ssh('xbabe3',cmd,timeout=remaining()+5);end=json.loads(raw.splitlines()[-1]);need(end['state']=='complete','publication incomplete');statusraw=ssh('xbabe3',['cat',root+'/publication/xbabe3/status.json']);status=json.loads(statusraw);save(folder,'publication',status)
 statusref={'path':root+'/publication-status.json','sha256':sha(statusraw),'bytes':len(statusraw)};copy('xbabe2',root,{'publication-status.json':statusraw})
 commit={'manifest':mref,'status':statusref,'publisher':config['publisher'],'base':config['base'],'authority':{'path':root+'/evergreen_kumos_packet_authority.py','sha256':m['authority_helper_sha256']},'source_pins':config['source_pins']};cb=canonical(commit);copy('xbabe2',root,{'commit.json':cb});cmd=bounded('xbabe2','jope-kumo3fresh-commit-'+ident[:12],common+'/kumo_commit.py',['--config',root+'/commit.json','--config-sha',sha(cb)],min(180,remaining()));result=json.loads(ssh('xbabe2',cmd,timeout=remaining()+5).splitlines()[-1]);need(result['state']=='stage_committed'and result['variants']==m['total_variants'],'stage CAS incomplete');save(folder,'complete',result);return result

def validate_config(c,pin):
 need(c['schema']=='jope.kumos-xbabe3-fresh32-controller/v1' and c['approved_for_launch'] is True,'reviewed approved fixed configuration required')
 need(c['packet_id']==PACKET_ID and c['packet_size']==COUNT and c['max_packets']==1 and c['host']==HOST,'hard one fresh32 packet')
 need(c['database_config_sha256']==DB_CONFIG_SHA and c['runtime']['sha256']==RUNTIME_SHA,'runtime/DB identity')
 need(type(c['created_unix'])in(int,float)and type(c['deadline_unix'])in(int,float)and c['created_unix']<=time.time()<c['deadline_unix']<=GLOBAL_DEADLINE and 0<c['deadline_unix']-c['created_unix']<=1800,'frozen global window')
 need(type(c['shared_monotonic_deadline'])in(int,float)and 0<c['shared_monotonic_deadline']-time.monotonic()<=1800,'shared monotonic deadline');need(c['shared_scope']=='next128-four-fixed32-v1','shared scope');need(c['controller_sha256']==sha(Path(__file__).read_bytes()),'controller source')
 need(c['owned_runtime']==SOURCE_ROOT,'owned fixed runtime')
 need(c['scope_boundary']=={'additional_all_history_packet_row_ceiling':128,'old_process_budget_reset':False,'old_controller_changes':False,'new_packets':1,'new_variants':32},'fresh scope boundary differs')
 for ref in c['local_source_pins']:
  path=Path(ref['path']);need(path.is_file()and not path.is_symlink()and path.stat().st_size==ref['bytes']and sha(path.read_bytes())==ref['sha256'],'local source drift')
 return c

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);a=p.parse_args()
 raw=Path(a.config).read_bytes();need(sha(raw)==a.config_sha,'controller config');c=validate_config(json.loads(raw),a.config_sha);common=c['owned_runtime']
 local=HERE/'runs';local.mkdir(exist_ok=True)
 with(HERE/'controller.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  need(not(local/PACKET_ID).exists(),'one-shot campaign already attempted; no automatic retry')
  files={Path(r['path']).name:Path(r['path']).read_bytes()for r in c['local_source_pins']};files['config.json']=raw
  for host in('xbabe3','xbabe2'):copy(host,common,files)
  # One invocation only; blocked/failed results exit visibly, no empty polling.
  result=packet(c,a.config_sha,common,local);print(json.dumps(result),flush=True)
  need(result['state']=='stage_committed','single fresh packet did not complete: '+result['state'])
if __name__=='__main__':main()
