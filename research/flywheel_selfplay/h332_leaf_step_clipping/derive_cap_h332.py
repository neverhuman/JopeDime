"""Exact-version BASE recovery and fixed cap, before preregistration or any fit."""
from pathlib import Path
import hashlib,json,tarfile,io,re,datetime
import numpy as np
import boto3
from cap_parser import raw_leaf_magnitudes,fixed_cap
P=Path(__file__).resolve().parent.parent;D=Path('/home/ubuntu/jope-data/h332-solhwv');S=P/'source'
BUCKET='veox-jopedime-learning-use2';KEY='experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz';VER='W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW';SHA='4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5'
INV=json.loads(Path('/home/ubuntu/jope-rescue-20260921/h316-solhwv/archive-member-inventory.json').read_text())
sha=lambda x:hashlib.sha256(x).hexdigest()
class HashReader:
 def __init__(self,body):self.body=body;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  b=self.body.read(n);self.h.update(b);self.n+=len(b);return b

def main():
 D.mkdir(exist_ok=True,parents=True);client=boto3.Session(profile_name='veox-jepson-prod').client('s3');r=client.get_object(Bucket=BUCKET,Key=KEY,VersionId=VER);reader=HashReader(r['Body'])
 saved={};catalogs={};models={};chunks=[];skipped_veto=0;inner_archive_shas={}
 with tarfile.open(fileobj=reader,mode='r|gz') as outer:
  for member in outer:
   name=member.name
   control=(name.startswith('run/P') and name.endswith('/fit_lock.json')) or (name.startswith('run/BASE_') and name.endswith(('.npy','.json'))) or name in ['run/regression.json','run/train-input-identities.json']
   catalog=bool(re.fullmatch(r'run/P[012]/fold_[0-3]/root_3000[123]/inner_archive.json',name))
   inner=bool(re.fullmatch(r'run/P[012]/fold_[0-3]/root_3000[123]/inner_members.tar.gz',name))
   if not (control or catalog or inner):continue
   assert member.isfile() and name in INV and '..' not in Path(name).parts
   blob=outer.extractfile(member).read();assert sha(blob)==INV[name]['sha256'];saved[name]=INV[name]
   if control:
    path=D/'base'/name[4:];path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(blob);path.chmod(0o444)
   if catalog:catalogs[str(Path(name).parent)]=json.loads(blob)
   if inner:
    parent=str(Path(name).parent);cat=catalogs[parent];assert sha(blob)==cat['sha256'] and cat['exact_version_get_sha256_verified'];inner_archive_shas[parent]=sha(blob);count=0
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as nested:
     for item in nested:
      assert item.isfile() and item.name in cat['members'] and '..' not in Path(item.name).parts and not Path(item.name).is_absolute()
      data=nested.extractfile(item).read();assert sha(data)==cat['members'][item.name]
      if not item.name.endswith('.txt'):continue
      if item.name.endswith('-veto.txt'):skipped_veto+=1;continue
      assert re.fullmatch(r'inner_tau(?:0.05|0.1|0.15)_fold[0-3]/f[0-3]-r3000[123]-m[012]-s[0-9]+-(?:ranker|bag[0-7])\.txt',item.name),item.name
      key=parent[4:]+'/'+item.name;assert key not in models
      values,lr,trees=raw_leaf_magnitudes(data);chunks.append(values);models[key]=dict(sha256=sha(data),learning_rate=lr,leaf_count=len(values),trees=trees);count+=1
    assert count==324,(parent,count)
    print(json.dumps(dict(cap_source_task=parent[4:],ranker_models=count,pooled_models=len(models))),flush=True)
 while reader.read(1024*1024):pass
 r['Body'].close();assert reader.h.hexdigest()==SHA and reader.n==882198184
 assert len(models)==11664 and len(inner_archive_shas)==36 and skipped_veto==1296
 key='experiments/rd_breakthrough_20261030/H315/picks.npz';ver='jCll6z9FGwmpNg.cocb305LcVzFYuieR';dig='e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615'
 r=client.get_object(Bucket=BUCKET,Key=key,VersionId=ver);blob=r['Body'].read();r['Body'].close();assert sha(blob)==dig
 path=D/'base/h315_picks.npz';path.write_bytes(blob);path.chmod(0o444)
 receipt=dict(status='EXACT_VERSION_GET_AND_SHA_PASS',h316=dict(key=KEY,version_id=VER,sha256=SHA,streamed_bytes=reader.n,members=saved),h315=dict(key=key,version_id=ver,sha256=dig))
 (P/'base-fetch-receipt.json').write_text(json.dumps(receipt,indent=1)+'\n')
 pooled=np.concatenate(chunks);cap=fixed_cap(pooled);(D/'cap').mkdir(exist_ok=True);np.save(D/'cap/raw_unshrunk_abs_leaves.npy',pooled)
 (S/'cap_models.json').write_text(json.dumps(models,sort_keys=True,separators=(',',':'))+'\n')
 config=dict(wave='H332',status='CAP_FIXED_BEFORE_FITS',cap=cap,quantile=.90,quantile_method='numpy linear',formula='abs(native leaf_value / model learning_rate)',models=len(models),pooled_leaves=len(pooled),skipped_veto_models=skipped_veto,source=dict(bucket=BUCKET,key=KEY,version_id=VER,sha256=SHA),cap_model_inventory_sha256=sha((S/'cap_models.json').read_bytes()),pooled_values_sha256=sha((D/'cap/raw_unshrunk_abs_leaves.npy').read_bytes()),inner_archive_sha256=inner_archive_shas,scope='ALL archived BASE inner main+8bag rankers across TRAIN partitions/roots; no veto/refits/NORM; global historical TRAIN exposure disclosed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (S/'cap_config.json').write_text(json.dumps(config,indent=2)+'\n');(P/'cap-receipt.json').write_text(json.dumps(config,indent=2)+'\n');print(json.dumps(config),flush=True)
if __name__=='__main__':main()
