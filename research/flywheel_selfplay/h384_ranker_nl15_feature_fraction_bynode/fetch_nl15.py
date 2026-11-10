"""Fresh version-pinned H352 archive; verify every member, retain outcomes only."""
from pathlib import Path
import hashlib,json,re,tarfile
import boto3
from native_audit import parse,native_delta
P=Path(__file__).resolve().parent.parent;D=Path('/home/ubuntu/jope-data/h384-dsc01');HERE=Path(__file__).resolve().parent
BUCKET='veox-jopedime-learning-use2'
KEY='experiments/rd_breakthrough_20261030/H352/jrb01-20261105T075500Z/closeout/closeout-manifest.json'
VER='RtNLAfmRJbXqdXQx7nxZLZ8dWHPcVYQC';SHA='d3b47c5fbc1ecc3c6e6d4f89dfef7d6a3f5216f45c6dd1589ce94fb413350f60'
class Reader:
 def __init__(self,b):self.b=b;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  b=self.b.read(n);self.h.update(b);self.n+=len(b);return b

def main():
 s=boto3.Session(profile_name='veox-jepson-prod').client('s3',region_name='us-east-2')
 obj=s.get_object(Bucket=BUCKET,Key=KEY,VersionId=VER);assert obj['VersionId']==VER;raw=obj['Body'].read();obj['Body'].close();assert hashlib.sha256(raw).hexdigest()==SHA
 manifest=json.loads(raw);assert manifest['wave']=='H352' and manifest['status']=='EXACT_VERSION_GET_ALL_MEMBERS_VERIFIED'
 (P/'h352-archive-manifest.json').write_bytes(raw)
 saved={};rankers=vetos=members=0;bynode_values=[];part_proofs=[];veto_ref=json.loads((HERE/'base_veto_sha256.json').read_text())['members']
 for part in manifest['parts']:
  obj=s.get_object(Bucket=BUCKET,Key=part['key'],VersionId=part['version_id']);assert obj['VersionId']==part['version_id'];reader=Reader(obj['Body']);seen=set()
  with tarfile.open(fileobj=reader,mode='r|gz') as tar:
   for member in tar:
    name=member.name;assert member.isfile() and name in part['members'] and name not in seen and '..' not in Path(name).parts and not Path(name).is_absolute();seen.add(name)
    pin=part['members'][name];assert member.size==pin['bytes'];rel=Path(name).relative_to('h352-jrb01');rs=str(rel)
    model=bool(re.fullmatch(r'P[012]/fold_[0-3]/root_3000[123]/refit/[^/]+\.txt',rs))
    keep=bool(re.fullmatch(r'P[012]/fold_[0-3]/root_3000[123]/(?:fit_lock|score_once|guard_receipt|input_identity)\.json',rs)) or rs in ['metrics.json','raw_family_gains.npz','picks.npz','fit_done.json','lock_review.json','regression.json','train-input-identities.json'] or bool(re.fullmatch(r'NL15_P[012](?:_locks\.json|\.npy)',rs))
    h=hashlib.sha256();data=bytearray() if keep or model else None
    with tar.extractfile(member) as f:
     for chunk in iter(lambda:f.read(1<<20),b''):
      h.update(chunk)
      if data is not None:data.extend(chunk)
    assert h.hexdigest()==pin['sha256'],name
    if model:
     native=parse(data.decode());veto=rs.endswith('-veto.txt')
     if veto:assert pin['sha256']==veto_ref['run/'+rs]['sha256'];vetos+=1
     else:
      assert native['num_leaves']=='15' and native['max_depth']=='3' and native['bagging_by_query']=='0' and float(native['bagging_fraction'])==.8 and native['bagging_freq']=='1'
      bynode_values.append(float(native['feature_fraction_bynode']))
      native_delta(dict(random_state=int(native['seed']),learning_rate=float(native['learning_rate']),n_estimators=int(native['num_iterations'])),native,False);rankers+=1
    if keep:
     path=D/'nl15'/rel;path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();path.write_bytes(data);path.chmod(0o444);saved[rs]=pin
  while reader.read(1<<20):pass
  obj['Body'].close();assert seen==set(part['members']) and reader.h.hexdigest()==part['sha256'] and reader.n==part['bytes']
  members+=len(seen);part_proofs.append(dict(key=part['key'],version_id=part['version_id'],sha256=part['sha256'],bytes=reader.n,members_verified=len(seen)))
  print(json.dumps(dict(part_verified=len(part_proofs),members_verified=members)),flush=True)
 assert members==manifest['source_files']==2691 and len(part_proofs)==27 and rankers==972 and vetos==108
 assert sum(k.endswith('/fit_lock.json') for k in saved)==36 and sum(k.endswith('/score_once.json') for k in saved)==36
 report=dict(status='PASS',comparator='H352 NL15',archive_manifest=dict(bucket=BUCKET,key=KEY,version_id=VER,sha256=SHA),parts=part_proofs,members_verified=members,saved=saved,native_ranker_configurations_verified=rankers,native_feature_fraction_bynode_values=sorted(set(bynode_values)),native_feature_fraction_bynode_count=len(bynode_values),BASE_veto_SHA_matches=vetos,optimizer_fits=0,new_model_predictions=0)
 (P/'nl15-fetch-receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['parts','saved']}))
if __name__=='__main__':main()
