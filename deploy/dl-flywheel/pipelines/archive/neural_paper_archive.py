"""Versioned paper custody of a closed neural run; no scientific-result mutation."""
import hashlib,importlib.util,json,os,sqlite3,subprocess,time
from pathlib import Path

ROOT=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final')
CAP=256*1024**2
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def closed_unit(unit):
 need(unit=='jopedime-flywheel-neural-20261009-v1','owned unit only')
 r=subprocess.run(['systemctl','--user','show',unit,'--property=ActiveState,MainPID'],text=True,capture_output=True,check=True,timeout=5)
 fields=dict(s.split('=',1)for s in r.stdout.splitlines()if '='in s)
 need(fields.get('ActiveState')in('inactive','failed')and fields.get('MainPID')=='0','neural run must be closed')
 return fields
def selected(root):
 """Closed-run exact allowlist; GP binaries remain in existing result custody."""
 names=['config.json','registry.json','window.json','status.json','anchor.json','archive-config.json','research-admission-verification.json','loaded-receipt.json','role-proof.json','put-attempts.jsonl']
 result=[]
 for name in names:
  p=root/name
  need(p.is_file()and not p.is_symlink(),'required evidence missing: '+name);result.append((name,p))
 for folder in ('runtime','training','attempt-custody'):
  d=root/folder;need(d.is_dir()and not d.is_symlink(),'evidence directory missing')
  for p in sorted(d.rglob('*')):
   need(not p.is_symlink(),'symlink in evidence')
   if p.is_file()and p.suffix in ('.py','.json','.jsonl','.safetensors','.sqlite'):
    result.append((str(p.relative_to(root)),p))
 for name in ('neural_core.py','neural_loop.py','neural_cli.py','neural_admission.py','campaign_custody.py','flywheel_kumos_snapshot.py','neural-failure.json'):
  p=root/name
  if p.exists():result.append((name,p))
 need(0<len(result)<=512 and len({n for n,p in result})==len(result),'bounded unique evidence files')
 need(sum(p.stat().st_size for n,p in result)<=CAP,'total evidence cap256MiB')
 return result
def main():
 here=Path(__file__).resolve().parent
 config_raw=(here/'config.json').read_bytes();c=json.loads(config_raw)
 need(c['helper_sha256']==sha(Path(__file__).read_bytes()),'helper pin')
 need(c['schema']=='jope.neural-paper-custody-config/v1','config schema')
 root=Path(c['source_root']);need(root==ROOT/'flywheel_neural_20261009_v1','source confinement')
 need(all(not p.is_symlink()for p in(root,*root.parents)),'source symlink')
 need(sha((root/'config.json').read_bytes())==c['source_config_sha256'],'neural config pin')
 deadline=c['deadline_unix'];need(deadline-c['created_unix']==180 and c['created_unix']<=time.time()<deadline,'fixed180sec new custody window')
 source=Path(c['base_helper']['path']);need(sha(source.read_bytes())==c['base_helper']['sha256'],'pinned archive primitives')
 spec=importlib.util.spec_from_file_location('paper_base',source);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
 limits=b.cgroup_check();service=closed_unit(c['source_unit'])
 out=here/'result';out.mkdir(mode=0o700)
 status=json.loads(b.read_small(root/'status.json'))
 need(status['config_sha256']==c['source_config_sha256']and status['attempt_custody']['status']=='PASS','closed native attempt custody')
 selected_files=selected(root);entries=[]
 for step in range(status.get('optimizer_steps',0)+1):
  checkpoint=root/'training'/('checkpoint-%04d'%step)
  metadata=json.loads(b.read_small(checkpoint/'manifest.json'))
  tensor=checkpoint/'state.safetensors'
  stream,info=b.safe_open(tensor,64*1024**2);stream.close()
  need(metadata['step']==step and metadata['config_sha256']==c['source_config_sha256']and
       metadata['state_bytes']==info.st_size and metadata['state_sha256']==b.sha_file(tensor),'physical checkpoint coverage/hash')
 neural_config=json.loads((root/'config.json').read_bytes())
 for key in ('snapshot_acceptance','snapshot_status','snapshot_coverage','gpu_preflight'):
  ref=neural_config[key];path=Path(ref['path']);body=b.read_small(path,32*1024**2)
  need(sha(body)==ref['sha256']and len(body)==ref['bytes'],'exact referenced paper evidence: '+key)
  selected_files.append(('external/'+key+'.json',path))
 snapshot_status=json.loads(b.read_small(Path(neural_config['snapshot_status']['path'])))
 for key in ('snapshot','loaded_receipt'):
  ref=snapshot_status[key];path=Path(ref['path']);body=b.read_small(path,32*1024**2)
  need(sha(body)==ref['sha256']and len(body)==ref['bytes'],'exact referenced snapshot evidence: '+key)
  selected_files.append(('external/'+key+'.json',path))
 for name,path in selected_files:
  b.time_check(deadline)
  if path.suffix=='.sqlite':
   target=out/('db-'+sha(name.encode())[:16]+'.sqlite')
   for suffix in ('','-wal','-shm'):
    sidecar=Path(str(path)+suffix)
    if sidecar.exists():
     stream,_=b.safe_open(sidecar,64*1024**2);stream.close()
   connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.5)
   copied=sqlite3.connect(str(target))
   try:
    connection.execute('PRAGMA query_only=ON')
    connection.backup(copied,pages=128,progress=lambda status,remaining,total:b.time_check(deadline),sleep=.01)
    need(copied.execute('PRAGMA integrity_check').fetchone()[0]=='ok','SQLite evidence integrity')
   finally:connection.close();copied.close()
   path=target
  stream,info=b.safe_open(path,64*1024**2);stream.close()
  entries.append((name,path,b.sha_file(path),info.st_size))
 manifest={'schema':'jope.neural-paper-manifest/v1','source_config_sha256':c['source_config_sha256'],
   'source_root':str(root),'closed_service':service,'at':time.time(),'optimizer_steps':status.get('optimizer_steps',0),
   'terminal_state':status['state'],'artifacts':[{'path':n,'sha256':h,'bytes':size}for n,p,h,size in entries],
   'scope':'Physical checkpoints including optimizer/RNG, source, preprocessing, exact batch order/proofs, full loss and measured games. GP executable/data artifacts retain their separate exact-version S3 result references. Custody verifies evidence bytes; it does not establish generalization.'}
 raw=canonical(manifest);b.write_new(out/'manifest.json',raw,out)
 entries.append(('PAPER_MANIFEST.json',out/'manifest.json',sha(raw),len(raw)))
 package=out/'archive.tar';pins=b.package_tar(package,entries,out,deadline);b.verify_tar(package,pins,deadline)
 closed_unit(c['source_unit'])
 from botocore.config import Config
 import boto3
 client=boto3.Session(profile_name='veox-jepson-prod').client('s3',config=Config(connect_timeout=5,read_timeout=10,retries={'max_attempts':0},max_pool_connections=1))
 package_sha=b.sha_file(package);prefix=b.PREFIX+'neural/'+c['source_config_sha256']+'/'+package_sha+'/'
 uploaded=b.put_verified(client,prefix+'archive.tar',package,package_sha,package.stat().st_size,deadline)
 b.time_check(deadline)
 proof={'schema':'jope.neural-paper-custody/v1','state':'verified','at':time.time(),'source_config_sha256':c['source_config_sha256'],
   'optimizer_steps':status.get('optimizer_steps',0),'terminal_state':status['state'],'package':uploaded,'manifest_sha256':sha(raw),
   'files':len(pins),'checkpoint_files':sum(x['path'].endswith('.safetensors')for x in pins),'resource_limits':limits}
 proof_raw=canonical(proof)
 proof['receipt']=b.put_verified(client,prefix+'verification.json',proof_raw,sha(proof_raw),len(proof_raw),deadline)
 b.time_check(deadline)
 b.write_new(out/'status.json',canonical(proof),out);print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
