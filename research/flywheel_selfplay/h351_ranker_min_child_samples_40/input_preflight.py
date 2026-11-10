from pathlib import Path
import json,sys,hashlib,datetime,subprocess
P=Path(__file__).resolve().parent.parent;D=Path('/home/ubuntu/jope-data/h351-gov03');sys.path.insert(0,str(P/'vendor/h308_ensvote2_trainonly_repartition'))
import trainonly_loader as tl

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
try:
 old=json.loads((D/'base/train-input-identities.json').read_text());root=Path('/home/ubuntu/jope-data/h50/input')
 assert sha(root/'selected.json')==old['selected_sha256']
 assert sha('/home/ubuntu/jope-data/h276/inputs/r882-train-costs.npz')==old['r882_train_costs_sha256']
 for c,rec in old['train_pools'].items():assert sha(root/'pools'/f'{c}.npz')==rec['sha256'],c
 for name,rec in old['cache_train_prefix'].items():
  with tl.TrainOnlyNumpy(set(old['train_pools'])).load(Path('/home/ubuntu/jope-data/h58/cache')/name) as z:
   assert z.n_train==rec['n_train'] and z.row_prefix==rec['row_prefix']
   for key,info in rec['members'].items():
    a=z[key];assert hashlib.sha256(a.tobytes()).hexdigest()==info['sha256'] and list(a.shape)==info['shape'] and str(a.dtype)==info['dtype']
 assert tl.COUNTERS['dlval_feature_rows_read']==0
 report=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),train_pool_sha256_matches=len(old['train_pools']),train_prefix_feature_members=4,selected_and_r882_train_costs='EXACT SHA256',DLVal_feature_rows_read=0,reference_manifest_sha256=sha(D/'base/train-input-identities.json'),scope='All current TRAIN program/fitness/gate pools and all consumed cache feature prefixes exactly match the archived BASE inputs; no file bodies outside TRAIN prefixes.')
 (D/'input_identity_validation.json').write_text(json.dumps(report,indent=2)+'\n');(P/'input_identity_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
except BaseException:
 subprocess.run(['systemctl','--user','stop','h351-run.service'],check=False);raise
