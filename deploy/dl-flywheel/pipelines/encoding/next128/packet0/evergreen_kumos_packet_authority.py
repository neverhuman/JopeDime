"""Read-only current-target/exclusion gate on xbabe2 for a fixed encoder manifest."""
import argparse,hashlib,json,math,socket,time,importlib.util,sys
from pathlib import Path
R=Path('/home/ubuntu/scratch/jopedime-dl-training-v1')
CONFIG_SHA='cd9ee8f882735c42b49d694b99dbbc4187035804099bc49f0ca81f8ec77a9b0a'
REGISTRY_SHA='8fe409848aadd6256644466faca6a0ceab73ab67131d30243701e0d93d9d0593'

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(ref,cap=32*1024**2):
 p=Path(ref['path']);need(p.is_absolute()and'..'not in p.parts and all(not q.is_symlink()for q in(p,*p.parents))and p.is_file(),'bounded regular source');need(p.stat().st_size<=cap,'authority read cap')
 with p.open('rb')as f:b=f.read(cap+1)
 need(len(b)<=cap and sha(b)==ref['sha256'],'authority SHA/cap');return b

def check(m,msha,*,latest_root=R,now=time.time):
 need(m['generator_config']['sha256']==CONFIG_SHA and m['parent_registry']['sha256']==REGISTRY_SHA,'pinned authority')
 conf=json.loads(read(m['generator_config']));need(conf['parent_registry']['sha256']==REGISTRY_SHA,'registry identity')
 # Reread the current authoritative registry path, not only the copied snapshot.
 parents=[json.loads(x)for x in read(conf['parent_registry']).splitlines()];by={p['parent_lineage_sha256']:p for p in parents};need(len(by)==len(parents)==1068,'registry support')
 docs={r['path']:json.loads(read(r))for r in conf['authority_pins']};targets=[p for p in docs if '/gp-target-one-holdout-after-layers-'in p];need(len(targets)==1,'one target authority')
 tp=targets[0];t=docs[tp];rp=t['registry']['path'];need(rp in docs,'exclusion registry missing')
 paths=list((latest_root/'receipts/gp-hwm-index').glob('gp-target-one-holdout-after-layers-v*.json'));need(paths and max(paths,key=lambda p:p.stat().st_mtime)==Path(tp),'newer target authority exists')
 paths=list((latest_root/'receipts/exclusions').glob('TRAINING-EXCLUSION-LAYERS-v*.json'));need(paths and max(paths,key=lambda p:p.stat().st_mtime)==Path(rp),'newer exclusion registry exists')
 excluded=set();excluded_views=set()
 for ref in docs[rp]['layers']:
  need(ref['path']in docs,'unpinned exclusion layer');d=docs[ref['path']]
  if d.get('schema')=='jope.dltrain-training-exclusion-layer/v1':
   for e in d['entries']:excluded.add(e['context_sha256']);excluded_views.update(e.get('view_ids',[]))
 eligible=set(t['views']);support=[]
 wp=Path(m['worker_path']);read({'path':str(wp),'sha256':m['worker_sha256']})
 spec=importlib.util.spec_from_file_location('verified_packet_coverage',wp);worker=importlib.util.module_from_spec(spec);sys.modules[spec.name]=worker;spec.loader.exec_module(worker)
 for (host,identity),v in sorted(worker.coverage(m).items()):
  parent=by[v['lineage']];members=set(parent['retained_lineage_context_sha256'])
  need(parent['view_id']in eligible and parent['view_id']not in excluded_views and not(members&excluded),'selected parent/lineage no longer allowed')
  need(parent['partition']==v['partition'],'partition changed')
  support.append({'host':host,'variant_identity':identity,'parent_lineage_sha256':v['lineage'],'parent_context_sha256':parent['parent_context_sha256'],'view_id':parent['view_id'],'partition':parent['partition'],'parent_record_sha256':sha(canonical(parent)),'retained_contexts_checked':len(members),'eligible_view':True,'excluded_contexts_or_views':False})
 need(len(support)==len({r['variant_identity']for r in support})==m['total_variants'],'exact packet support')
 return {'schema':'jope.flywheel-evergreen-encoding-authority/v1','state':'passed','checked_at_unix':now(),'host':'xbabe2','manifest_sha256':msha,'authority_helper_sha256':m['authority_helper_sha256'],'generator_config_sha256':CONFIG_SHA,'parent_registry_sha256':REGISTRY_SHA,'target_path':tp,'exclusion_registry_path':rp,'authority_pins':conf['authority_pins'],'support':support,'scope':'Current eligible target and excluded context/view closure, including retained lineage membership. Development split caveats inherited; no pristine holdout or training admission claim.'}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);a=p.parse_args();need(socket.gethostname().split('.')[0]=='xbabe2','authority host must be xbabe2')
 m=json.loads(read({'path':a.manifest,'sha256':a.manifest_sha}));need(sha(Path(__file__).read_bytes())==m['authority_helper_sha256'],'authority helper source changed');print(json.dumps(check(m,a.manifest_sha),sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
