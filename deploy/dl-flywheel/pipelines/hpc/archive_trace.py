"""Archive closed host telemetry segments with exact-version S3 readback."""
import argparse,hashlib,json,os,time
from pathlib import Path

BUCKET='veox-jopedime-learning-use2'
PREFIX='experiments/gp-policy-dl/v2-trainonly/paper-evidence/20261009/hpc-trace-v1/'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def need(v,m):
 if not v:raise ValueError(m)
def verify_segment(root,seal_path,previous,seq,host):
 need(not seal_path.is_symlink() and seal_path.stat().st_size<4096,'seal bounds')
 seal_raw=seal_path.read_bytes();seal=json.loads(seal_raw);path=Path(seal['path'])
 need(path.parent==root and path.name==seal_path.name.replace('.sealed.json','.jsonl') and not path.is_symlink(),'segment path')
 need(path.stat().st_size==seal['bytes'] and 0<seal['bytes']<10*1024**2,'segment byte cap')
 body=path.read_bytes();need(sha(body)==seal['sha256'],'sealed bytes')
 for line in body.splitlines():
  event=json.loads(line);digest=event.pop('sha256');seq+=1
  need(event['host']==host and event['previous']==previous and event['seq']==seq and sha(canonical(event))==digest,'event chain')
  previous=digest
 need(seq==seal['last_seq'] and previous==seal['event_head'],'sealed chain head')
 return body,seal_raw,previous,seq

def put_verified(client,key,body,deadline):
 def tick():need(time.time()<deadline,'archive deadline')
 tick();need(key.startswith(PREFIX) and len(body)<=10*1024**2,'archive namespace/bounds')
 try:r=client.put_object(Bucket=BUCKET,Key=key,Body=body,ContentLength=len(body),IfNoneMatch='*',Metadata={'sha256':sha(body)})
 except Exception as e:
  if getattr(e,'response',{}).get('Error',{}).get('Code') not in ('412','PreconditionFailed','ConditionalRequestConflict'):raise
  tick();r=client.head_object(Bucket=BUCKET,Key=key)
 version=r.get('VersionId');need(version not in (None,'','null'),'versioning required');tick()
 r=client.get_object(Bucket=BUCKET,Key=key,VersionId=version)
 need(r.get('VersionId')==version and r['ContentLength']==len(body),'exact version size')
 with r['Body'] as stream:actual=stream.read(len(body)+1)
 tick();need(actual==body,'exact archived bytes')
 return {'bucket':BUCKET,'key':key,'version_id':version,'sha256':sha(body),'bytes':len(body),'verified_at_unix_ns':time.time_ns()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
 root=a.root;need(root.is_absolute() and all(not p.is_symlink()for p in(root,*root.parents)),'root bounds');host=os.uname().nodename;need(host in('xbabe1','xbabe2','xbabe3'),'known host')
 import boto3
 from botocore.config import Config
 client=boto3.Session(profile_name='veox-jepson-prod').client('s3',config=Config(connect_timeout=5,read_timeout=20,retries={'max_attempts':0}))
 previous='0'*64;seq=0;part=0;refs=[];total=0
 # This invocation owns its own receipts; an existing archive is never overwritten.
 receipts=root/'archive-v1';receipts.mkdir(exist_ok=False)
 while time.time()<a.deadline-60:
  path=root/f'events-{part:05d}.sealed.json'
  if not path.exists():
   if a.once:break
   time.sleep(min(15,max(0,a.deadline-time.time()-60)));continue
  need(part<2500,'fixed 5000-object archive budget')
  body,sealed,head,end=verify_segment(root,path,previous,seq,host)
  prefix=PREFIX+host+'/'+str(part).zfill(5)+'/'+sha(body)+'/'
  log_ref=put_verified(client,prefix+'events.jsonl',body,a.deadline)
  seal_ref=put_verified(client,prefix+'seal.json',sealed,a.deadline)
  receipt={'schema':'jope.hpc-host-archive/v1','host':host,'part':part,'segment':log_ref,'seal':seal_ref,'first_seq':seq+1,'last_seq':end,'event_head':head,'source_sha256':sha(Path(__file__).read_bytes())}
  with (receipts/f'{part:05d}.json').open('xb')as f:f.write(canonical(receipt));f.flush();os.fsync(f.fileno())
  previous,seq=head,end;part+=1;total+=len(body)+len(sealed);refs.append(receipt)
  status={'schema':'jope.hpc-archive-status/v1','at_unix_ns':time.time_ns(),'host':host,'state':'running','verified_segments':part,'verified_objects':2*part,'archived_events':seq,'verified_bytes':total,'event_head':head,'latest':receipt,'deadline_unix':a.deadline}
  tmp=receipts/'status.tmp';tmp.write_bytes(canonical(status));os.replace(tmp,receipts/'status.json')
 print(json.dumps({'host':host,'segments':part,'events':seq,'bytes':total}))
if __name__=='__main__':main()
