"""Recover only exact pinned H352 TRAIN evidence into H392-owned copies."""
from pathlib import Path
import json,hashlib,tarfile,re,datetime
import boto3
P=Path(__file__).parent;D=Path('/home/ubuntu/jope-data/h392-hwv');D.mkdir(parents=True,exist_ok=True)
S3=boto3.Session(profile_name='veox-jepson-prod').client('s3',region_name='us-east-2')
metrics=json.loads((P/'source/h352_metrics_reference.json').read_text());a=metrics['archive_manifest']
r=S3.get_object(Bucket=a['bucket'],Key=a['key'],VersionId=a['version_id']);assert r['VersionId']==a['version_id'];raw=r['Body'].read();r['Body'].close();assert hashlib.sha256(raw).hexdigest()==a['sha256'] and len(raw)==a['bytes']
manifest=json.loads(raw);assert manifest['wave']=='H352' and manifest['status']=='EXACT_VERSION_GET_ALL_MEMBERS_VERIFIED'
(D/'nl15').mkdir(exist_ok=True);(D/'nl15/recovery-manifest.json').write_bytes(raw)
class Reader:
 def __init__(self,b):self.b=b;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  b=self.b.read(n);self.h.update(b);self.n+=len(b);return b
saved={};parts=[]
for part in manifest['parts']:
 assert part['bucket']==a['bucket'] and part['key'].startswith('experiments/rd_breakthrough_20261030/H352/jrb01-20261105T075500Z/closeout/')
 r=S3.get_object(Bucket=part['bucket'],Key=part['key'],VersionId=part['version_id']);assert r['VersionId']==part['version_id'];reader=Reader(r['Body']);seen=set()
 with tarfile.open(fileobj=reader,mode='r|gz') as tar:
  for member in tar:
   name=member.name;assert member.isfile() and name in part['members'] and name not in seen;seen.add(name)
   want=bool(re.fullmatch(r'h352-jrb01/P[012]/fold_[0-3]/root_3000[123]/(fit_lock.json|score_once.json|native_model_review.json|native_fit_audit.jsonl)',name)) or bool(re.fullmatch(r'h352-jrb01/NL15_P[012](_locks.json|.npy)',name)) or name in ('h352-jrb01/picks.npz','h352-jrb01/raw_family_gains.npz','h352-jrb01/metrics.json','h352-jrb01/operational_evidence/UNIT-FINAL.json') or bool(re.fullmatch(r'h352-jrb01/P[012]/fold_[0-3]/root_3000[123]/refit/[^/]+\.txt',name))
   h=hashlib.sha256();size=0;f=tar.extractfile(member);out=None
   if want:
    rel=Path(name).relative_to('h352-jrb01');assert '..' not in rel.parts and not rel.is_absolute();outpath=D/'nl15'/rel;outpath.parent.mkdir(parents=True,exist_ok=True);assert not outpath.exists();out=outpath.open('wb')
   for block in iter(lambda:f.read(1024*1024),b''):
    h.update(block);size+=len(block)
    if out:out.write(block)
   if out:out.close();outpath.chmod(0o444)
   expected=part['members'][name];assert h.hexdigest()==expected['sha256'] and size==expected['bytes'],name
   if want:saved[str(rel)]=expected
 while reader.read(1024*1024):pass
 r['Body'].close();assert reader.h.hexdigest()==part['sha256'] and reader.n==part['bytes'] and seen==set(part['members'])
 parts.append({k:v for k,v in part.items() if k!='members'});print(json.dumps(dict(part=part['key'].rsplit('/',1)[-1],whole_SHA=True,all_members=len(seen),saved_total=len(saved))),flush=True)
assert len(list((D/'nl15').glob('P*/fold_*/root_*/fit_lock.json')))==36
assert len(list((D/'nl15').glob('P*/fold_*/root_*/score_once.json')))==36
assert len(list((D/'nl15').glob('P*/fold_*/root_*/refit/*.txt')))==1080
record=dict(status='EXACT_VERSION_GET_AND_ALL_MEMBERS_SHA_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wave='H352',manifest=a,parts=parts,all_source_members_verified=sum(len(p['members']) for p in manifest['parts']),saved_members=saved,BASE_refits=0,NL15_refits=0,new_model_predictions=0)
(P/'nl15-fetch-receipt.json').write_text(json.dumps(record,indent=1)+'\n');print(json.dumps(dict(status=record['status'],saved=len(saved),all_members=record['all_source_members_verified'])),flush=True)
