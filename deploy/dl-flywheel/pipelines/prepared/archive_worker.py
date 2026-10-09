from pathlib import Path
import json,hashlib,os,sys,time,tarfile,io,importlib.util
ROOT=Path(__file__).resolve().parent;R=ROOT.parent/'runtime';start=time.time();deadline=min(start+120,1791592146.380285)
def sha(b):return hashlib.sha256(b).hexdigest()
c=json.loads((ROOT/'archive-config.json').read_bytes());assert sha((ROOT/'archive_worker.py').read_bytes())==c['worker_sha256']
for n,h in c['sources'].items():assert sha((ROOT/n).read_bytes())==h
p=ROOT/'paper_backup_core.py';spec=importlib.util.spec_from_file_location('archive_core',p);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
status=json.loads(b.read_small(R/'status.json'));assert status['state']=='passed'and status['config_sha256']==c['preparation_config_sha256']and status['prepared']['sha256']==sha(b.read_small(R/'prepared.json',32*1024**2))
names=['config.json','window.json','status.json','prepared.json','loaded-receipt.json','batch-manifest.json','records.json','final-authority.json','prefetch.json','prepared_snapshot.py','prepared_worker.py','exact_artifact_cache.py','verified_cache.py']
rows=[];out=ROOT/'prepared-evidence.tar.gz'
with out.open('xb')as f:
 with tarfile.open(fileobj=f,mode='w:gz')as tar:
  for name in names:
   body=b.read_small(R/name,32*1024**2);info=tarfile.TarInfo('runtime/'+name);info.size=len(body);info.mtime=int(start);tar.addfile(info,io.BytesIO(body));rows.append({'path':info.name,'sha256':sha(body),'bytes':len(body)})
  for name in['test_prepared_snapshot.py','IMPLEMENTATION_HANDOFF.md']:
   body=b.read_small(ROOT/name);info=tarfile.TarInfo(name);info.size=len(body);info.mtime=int(start);tar.addfile(info,io.BytesIO(body));rows.append({'path':name,'sha256':sha(body),'bytes':len(body)})
meta={'schema':'jope.prepared-snapshot-paper-custody/v1','at_utc_time_ns':time.time_ns(),'at_monotonic_ns':time.monotonic_ns(),'source_preparation_status_sha256':sha(b.read_small(R/'status.json')),'preparation_config_sha256':status['config_sha256'],'prepared_artifact':status['prepared'],'entries':rows,'package_sha256':b.sha_file(out),'package_bytes':out.stat().st_size,'source_state_changed':False,'admission_committed':False,'optimizer_updates':0,'scope':'Closed CPU preparation evidence only; no training admission or GP retry.'}
with tarfile.open(out,'r:gz')as tar:
 members=tar.getmembers();assert len(members)==len(rows)
 for entry,member in zip(rows,members):assert member.isfile()and member.name==entry['path']and sha(tar.extractfile(member).read())==entry['sha256']
import boto3
from botocore.config import Config
client=boto3.Session(profile_name='veox-jepson-prod').client('s3',config=Config(connect_timeout=5,read_timeout=20,retries={'max_attempts':0}))
prefix=b.PREFIX+'prepared-snapshots/'+status['config_sha256']+'/'
package=b.put_verified(client,prefix+meta['package_sha256']+'/prepared-evidence.tar.gz',out,meta['package_sha256'],out.stat().st_size,deadline)
meta['package']=package;body=b.canonical(meta);m=b.put_verified(client,prefix+sha(body)+'/custody.json',body,sha(body),len(body),deadline)
receipt={'state':'verified','config_sha256':status['config_sha256'],'metadata':m,'package':package,'puts':2,'exact_version_gets':2,'source_updates':0,'optimizer_updates':0,'admission_committed':False}
with(ROOT/'archive-receipt.json').open('x')as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
