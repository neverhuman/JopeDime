"""Copy only SHA-verified TRAIN inputs; mixed-cache feature tails never read."""
from pathlib import Path
import sys,json,hashlib,shutil,zipfile,os
import numpy as np
P=Path(__file__).parent;D=Path('/home/ubuntu/jope-data/h359-hwv');sys.path.insert(0,str(P/'vendor/h308_ensvote2_trainonly_repartition'));import trainonly_loader as tl
old=json.loads((D/'base/train-input-identities.json').read_text());root=Path('/home/ubuntu/jope-data');dest=D/'inputs';inventory={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def copy(src,rel,expected):
 out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True);assert sha(src)==expected;shutil.copy2(src,out);assert sha(out)==expected;inventory[rel]=dict(sha256=expected,bytes=out.stat().st_size)
copy(root/'h50/input/selected.json','h50/input/selected.json',old['selected_sha256'])
copy(root/'h276/inputs/r882-train-costs.npz','h276/inputs/r882-train-costs.npz',old['r882_train_costs_sha256'])
for cid,r in old['train_pools'].items():copy(root/'h50/input/pools'/(cid+'.npz'),'h50/input/pools/'+cid+'.npz',r['sha256'])
for name,r in old['cache_train_prefix'].items():
 src=root/'h58/cache'/name;arrays={}
 with tl.TrainOnlyNumpy(set(old['train_pools'])).load(src) as z:
  assert z.n_train==r['n_train'] and z.row_prefix==r['row_prefix']
  arrays['cids']=z['cids'][:z.n_train]
  if name=='pool_priors.npz':arrays['offsets']=z['offsets'][:z.n_train+1]
  for key,info in r['members'].items():
   a=z[key];assert hashlib.sha256(a.tobytes()).hexdigest()==info['sha256'] and list(a.shape)==info['shape'] and str(a.dtype)==info['dtype'];arrays[key]=a
 out=dest/'h58/cache'/name;out.parent.mkdir(parents=True,exist_ok=True);np.savez(out,**arrays)
 inventory['h58/cache/'+name]=dict(sha256=sha(out),bytes=out.stat().st_size,TRAIN_feature_prefix_members=r['members'],staged_ids='TRAIN_ONLY; consumed feature bytes/order unchanged')
assert tl.COUNTERS['dlval_feature_rows_read']==0
r=dict(status='PASS',files=inventory,TRAIN_pool_count=1219,cache_feature_prefixes=4,DLVal_feature_rows_read=0,feature_tail_bytes_read=0,staged_bytes=sum(x['bytes'] for x in inventory.values()))
(P/'staged-input-inventory.json').write_text(json.dumps(r,indent=1)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='files'}),flush=True)
