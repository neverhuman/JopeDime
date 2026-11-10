"""Exact-version BASE recovery and native row-bagging spec check; no fit/predict."""
from pathlib import Path
import hashlib,json,tarfile,re,io,datetime
import boto3
P=Path(__file__).resolve().parent.parent;D=Path('/home/ubuntu/jope-data/h345-solhwv');D.mkdir(exist_ok=True,parents=True)
B='veox-jopedime-learning-use2';S=boto3.Session(profile_name='veox-jepson-prod').client('s3')
INV=json.loads(Path('/home/ubuntu/jope-rescue-20260921/h316-solhwv/archive-member-inventory.json').read_text())
KEY='experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz';VER='W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW';SHA='4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5'
sha=lambda b:hashlib.sha256(b).hexdigest()
class HashReader:
 def __init__(self,body):self.body=body;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  b=self.body.read(n);self.h.update(b);self.n+=len(b);return b
models={};seedmap={};mismatch=[]
def native(name,blob):
 params=dict(re.findall(r'^\[([^:]+): (.*)\]$',blob.decode().split('parameters:',1)[-1],re.M));fraction=float(params['bagging_fraction']);freq=int(params['bagging_freq']);seed=int(params['seed']);bseed=int(params['bagging_seed']);assert params['objective']=='lambdarank'
 if (fraction,freq)!=(.8,1):mismatch.append(dict(name=name,bagging_fraction=fraction,bagging_freq=freq))
 if str(seed) in seedmap:assert seedmap[str(seed)]==bseed
 seedmap[str(seed)]=bseed
 assert name not in models;models[name]=dict(sha256=sha(blob),bagging_fraction=fraction,bagging_freq=freq,seed=seed,bagging_seed=bseed,feature_fraction=float(params['feature_fraction']),feature_fraction_bynode=float(params['feature_fraction_bynode']),params_sha256=sha(json.dumps(params,sort_keys=True).encode()))
r=S.get_object(Bucket=B,Key=KEY,VersionId=VER);reader=HashReader(r['Body']);saved={};catalogs={};inners=0;finals=0
with tarfile.open(fileobj=reader,mode='r|gz') as outer:
 for item in outer:
  name=item.name
  control=(name.startswith('run/P') and name.endswith('/fit_lock.json')) or (name.startswith('run/BASE_') and name.endswith(('.npy','.json'))) or name in ('run/regression.json','run/train-input-identities.json')
  catalog=bool(re.fullmatch(r'run/P[012]/fold_[0-3]/root_3000[123]/inner_archive.json',name));inner=bool(re.fullmatch(r'run/P[012]/fold_[0-3]/root_3000[123]/inner_members.tar.gz',name));final=bool(re.fullmatch(r'run/P[012]/fold_[0-3]/root_3000[123]/refit/f[0-3]-r3000[123]-m[012]-s[0-9]+-(?:ranker|bag[0-7])\.txt',name))
  if not(control or catalog or inner or final):continue
  assert item.isfile() and name in INV and '..' not in Path(name).parts;blob=outer.extractfile(item).read();assert sha(blob)==INV[name]['sha256'];saved[name]=INV[name]
  if control:
   path=D/'base'/name[4:];path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(blob);path.chmod(0o444)
  if catalog:catalogs[str(Path(name).parent)]=json.loads(blob)
  if final:native(name,blob);finals+=1
  if inner:
   cat=catalogs[str(Path(name).parent)];assert sha(blob)==cat['sha256'] and cat['exact_version_get_sha256_verified'];seen=set();n=0
   with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as nested:
    for member in nested:
     assert member.isfile() and member.name in cat['members'] and member.name not in seen and '..' not in Path(member.name).parts;seen.add(member.name);data=nested.extractfile(member).read();assert sha(data)==cat['members'][member.name]
     if re.search(r'-(?:ranker|bag[0-7])\.txt$',member.name):native(str(Path(name).parent)+'/'+member.name,data);n+=1
   assert seen==set(cat['members']) and n==324;inners+=n
while reader.read(1024*1024):pass
r['Body'].close();assert reader.h.hexdigest()==SHA and reader.n==882198184
assert inners==11664 and finals==972 and len(models)==12636
key='experiments/rd_breakthrough_20261030/H315/picks.npz';ver='jCll6z9FGwmpNg.cocb305LcVzFYuieR';dig='e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615';r=S.get_object(Bucket=B,Key=key,VersionId=ver);blob=r['Body'].read();r['Body'].close();assert sha(blob)==dig;path=D/'base/h315_picks.npz';path.write_bytes(blob);path.chmod(0o444)
(P/'base-native-params.json').write_text(json.dumps(models,indent=1)+'\n');(P/'base-seed-map.json').write_text(json.dumps(seedmap,sort_keys=True)+'\n')
report=dict(status='PASS' if not mismatch else 'INCOMPLETE_SPEC_MISMATCH',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),ranker_models=len(models),inner_rankers=inners,final_rankers=finals,expected=dict(bagging_fraction=.8,bagging_freq=1),all_effective_native_bagging_fraction_point8_freq1=not mismatch,mismatches=mismatch,seed_mapping_entries=len(seedmap),native_inventory_sha256=sha((P/'base-native-params.json').read_bytes()),seed_map_sha256=sha((P/'base-seed-map.json').read_bytes()),source=dict(bucket=B,key=KEY,version_id=VER,sha256=SHA),scope='All archived BASE inner main+8bag rankers and final BASE refits; NORM excluded; native parameter read-only, no fitting/scoring')
(P/'base-native-spec.json').write_text(json.dumps(report,indent=2)+'\n');(P/'base-fetch-receipt.json').write_text(json.dumps(dict(status='EXACT_VERSION_GET_AND_SHA_PASS',h316=dict(key=KEY,version_id=VER,sha256=SHA,streamed_bytes=reader.n,members=saved),h315=dict(key=key,version_id=ver,sha256=dig)),indent=1)+'\n');print(json.dumps(report),flush=True)
if mismatch:raise RuntimeError('INCOMPLETE_SPEC_MISMATCH; no treatment fit')
