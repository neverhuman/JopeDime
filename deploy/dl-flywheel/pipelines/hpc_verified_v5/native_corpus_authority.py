"""Read-only current native role proof. No datasets, models or mutations.

Run on xbabe2 with the frozen native snapshot specification. Only pinned role
metadata is read. Latest target/registry pointers must still be these exact
files. The caller binds this helper SHA separately and uses the proof for at
most60seconds; repeat before optimizer admission and each work packet.
"""
import argparse,hashlib,json,time
from pathlib import Path
R=Path('/home/ubuntu/scratch/jopedime-dl-training-v1/receipts')
MAX_TOTAL=16*1024**2

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def check(spec):
 pins=spec['authority_pins'];need(set(pins)>={'target','role','layers'},'authority closure');docs={};total=0
 for name,r in pins.items():
  p=Path(r['path']);need(p.is_relative_to(R)and '..'not in p.parts and all(not q.is_symlink()for q in(p,*p.parents))and p.is_file(),'authority path')
  need(type(r['bytes'])is int and 0<r['bytes']<=4*1024**2,'authority file cap');total+=r['bytes'];need(total<=MAX_TOTAL,'authority read cap')
  with p.open('rb')as f:b=f.read(r['bytes']+1)
  need(len(b)==r['bytes']and sha(b)==r['sha256'],'authority changed:'+name);docs[name]=json.loads(b)
 target=docs['target'];reg=docs['layers'];role=docs['role']
 need(target.get('schema')=='jope.dltrain-gp-target-after-layers/v1'and role.get('schema')=='jope.dltrain-one-holdout-role-map/v1','authority schema')
 def latest(directory,pattern):
  ps=list(directory.glob(pattern));need(bool(ps),'authority latest missing');return max(ps,key=lambda p:(p.stat().st_mtime_ns,p.name))
 need(latest(R/'gp-hwm-index','gp-target-one-holdout-after-layers-v*.json')==Path(pins['target']['path']),'newer target exists')
 need(latest(R/'exclusions','TRAINING-EXCLUSION-LAYERS-v*.json')==Path(pins['layers']['path']),'newer exclusions exist')
 need(target['registry']['path']==pins['layers']['path']and target['registry']['sha256']==pins['layers']['sha256'],'target registry binding')
 need(target['inputs'].get(pins['role']['path'])==pins['role']['sha256'],'target role binding')
 bypath={r['path']:(r,docs[k])for k,r in pins.items()};protected=set();views=set()
 for r in reg['layers']:
  need(r['path']in bypath and bypath[r['path']][0]['sha256']==r['sha256'],'missing pinned exclusion layer')
  d=bypath[r['path']][1]
  # Registry includes four base authorities as well as context/view layers.
  # The hash-bound target already applies their exact sealed/rights rules;
  # historical GP-1 heldout rows are training under the later role authority.
  if d.get('schema')=='jope.dltrain-training-exclusion-layer/v1':
   for row in d['entries']:protected.add(row['context_sha256']);views.update(row.get('view_ids',[]))
  else:need(d.get('schema')in {'jope.dltrain-one-holdout-role-map/v1','jope.dl-p3-exclusion-list/v1','jope.gp-hwm-excluded-by-ruling/v1','jope.dltrain-gp-target/v1'},'unknown exclusion authority schema')
 need(len(target['views'])==len(set(target['views']))==target['gp_target'],'target counts')
 return {'schema':'jope.native-corpus-current-authority/v1','passed':True,'checked_at_unix':time.time(),'source_pins':pins,
   'helper_sha256':sha(Path(__file__).read_bytes()),'eligible_views_sha256':sha(canonical(sorted(target['views']))),
   'eligible_views':len(target['views']),'protected_context_ids':sorted(protected),'protected_views':sorted(views),
   'research_split_only':True,'untouched_holdout_claim':False}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True);p.add_argument('--spec-sha',required=True);a=p.parse_args()
 b=Path(a.spec).read_bytes();need(len(b)<=64*1024**2 and sha(b)==a.spec_sha,'snapshot spec hash/size');print(canonical(check(json.loads(b))).decode())
if __name__=='__main__':main()
