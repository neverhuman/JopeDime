"""Exact-version, SHA-verified frozen BASE evidence; never fits or reads protected panels."""
from pathlib import Path
import hashlib,json,tarfile,os,re
import boto3
P=Path(__file__).resolve().parent.parent;D=Path('/home/ubuntu/jope-data/h374-gov03');D.mkdir(exist_ok=True,parents=True)
B='veox-jopedime-learning-use2'
S=boto3.Session(profile_name='veox-jepson-prod').client('s3')
inventory=Path(__file__).resolve().parent/'base_archive_member_inventory.json'
assert hashlib.sha256(inventory.read_bytes()).hexdigest()=='79d2fd3f50501176fd3d06af25149337f126a6eaf2819e13070e1d5334edfb62'
INV=json.loads(inventory.read_text())
KEY='experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz'
VER='W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW';SHA='4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5'
class HashReader:
 def __init__(self,body):self.body=body;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  b=self.body.read(n);self.h.update(b);self.n+=len(b);return b
r=S.get_object(Bucket=B,Key=KEY,VersionId=VER);assert r['VersionId']==VER;reader=HashReader(r['Body']);saved={}
with tarfile.open(fileobj=reader,mode='r|gz') as tar:
 for item in tar:
  name=item.name
  want=(name.startswith('run/P') and name.endswith('/fit_lock.json')) or (name.startswith('run/BASE_') and name.endswith(('.npy','.json'))) or name in ('run/regression.json','run/train-input-identities.json')
  ranker=bool(re.fullmatch(r'run/P[012]/fold_[0-3]/root_3000[1-3]/refit/[^/]+\.txt',name))
  if want or ranker:
   assert item.isfile() and name in INV and '..' not in Path(name).parts
   data=tar.extractfile(item).read();assert hashlib.sha256(data).hexdigest()==INV[name]['sha256']
   path=D/('base_native' if ranker else 'base')/name[4:];path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(data);path.chmod(0o444);saved[name]=INV[name]
while reader.read(1024*1024):pass
r['Body'].close();assert reader.h.hexdigest()==SHA
assert len([x for x in saved if x.endswith('/fit_lock.json')])==36
assert sum('/refit/' in x and x.endswith('.txt') for x in saved)==1080
key='experiments/rd_breakthrough_20261030/H315/picks.npz';ver='jCll6z9FGwmpNg.cocb305LcVzFYuieR';dig='e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615'
r=S.get_object(Bucket=B,Key=key,VersionId=ver);assert r['VersionId']==ver;data=r['Body'].read();r['Body'].close();assert hashlib.sha256(data).hexdigest()==dig
p=D/'base/h315_picks.npz';p.write_bytes(data);p.chmod(0o444)
receipt=dict(h316=dict(key=KEY,version_id=VER,sha256=SHA,streamed_bytes=reader.n,members=saved),h315=dict(key=key,version_id=ver,sha256=dig),status='EXACT_VERSION_GET_AND_SHA_PASS')
(P/'base-fetch-receipt.json').write_text(json.dumps(receipt,indent=1)+'\n');print(json.dumps(dict(base_locks=36,panels=9,exact_version_sha=True,bytes=reader.n)),flush=True)
