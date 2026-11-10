"""One unscored read-only derivation over every archived BASE refit ranker leaf."""
from pathlib import Path
import datetime,hashlib,json,re,sys,math
import numpy as np
import lightgbm as lgb
HERE=Path(__file__).resolve().parent;E=HERE.parent;D=Path('/home/ubuntu/jope-data/h342-gov03')
def leaves(node):
 if 'leaf_index' in node or 'split_index' not in node:
  assert 'leaf_weight' in node,'Native leaf_weight is required; no inference from count or value'
  return [float(node['leaf_weight'])]
 return leaves(node['left_child'])+leaves(node['right_child'])
def main():
 assert not (HERE/'minhess_value.json').exists() and not (E/'minhess-derivation.json').exists()
 assert json.loads((D/'regression.json').read_text())['status']=='PASS'
 fetch=json.loads((E/'base-fetch-receipt.json').read_text());assert fetch['status']=='EXACT_VERSION_GET_AND_SHA_PASS'
 inv=fetch['h316']['members'];paths=sorted(n for n in inv if re.fullmatch(r'run/P[012]/fold_[0-3]/root_3000[1-3]/refit/[^/]+\.txt',n) and 'veto' not in n)
 assert len(paths)==972
 values=[];models=[];floors=set()
 for name in paths:
  p=D/'base_native'/name[4:];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==inv[name]['sha256'] and p.stat().st_size==inv[name]['bytes']
  text=raw.decode();native=dict(re.findall(r'^\[([^:\]]+): (.*)\]$',text,re.M))
  assert native['objective'].split()[0]=='lambdarank' and native['boosting']=='gbdt' and native['device_type']=='cpu' and native['num_threads']=='1'
  base=float(native['min_sum_hessian_in_leaf']);floors.add(base)
  model=lgb.Booster(model_str=text);dump=model.dump_model();v=[]
  for tree in dump['tree_info']:
   w=leaves(tree['tree_structure']);assert len(w)==tree['num_leaves'];v.extend(w)
  assert v and all(math.isfinite(w) and w>=0 for w in v);values.extend(v)
  models.append(dict(archive_member=name,sha256=inv[name]['sha256'],trees=len(dump['tree_info']),leaves=len(v),effective_min_sum_hessian_in_leaf=base))
 assert len(floors)==1 and next(iter(floors))==0.001
 a=np.asarray(values,dtype='<f8');q=float(np.quantile(a,.25,method='linear'));value=float(format(q,'.3g'));base=next(iter(floors));status='PASS' if value>base else 'INCOMPLETE_SPEC_MISMATCH'
 np.save(D/'base_leaf_weights.npy',a)
 receipt=dict(wave='H342',status=status,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),derivation='25th percentile of leaf_weight across every leaf of all972 archived BASE refit rankers (108main+864bag); each archived model path counted once',quantile=.25,quantile_method='linear',rounding='3 significant figures using Python format(.3g)',unrounded_quantile=q,MINHESS=value,BASE_effective_min_sum_hessian_in_leaf=base,models_count=len(models),leaves_count=len(a),leaf_weights_f64le_sha256=hashlib.sha256(a.tobytes()).hexdigest(),leaf_weights_npy_sha256=hashlib.sha256((D/'base_leaf_weights.npy').read_bytes()).hexdigest(),leaf_weights_min=float(a.min()),leaf_weights_max=float(a.max()),source_archive={k:v for k,v in fetch['h316'].items() if k!='members'},model_inventory_sha256=hashlib.sha256(json.dumps(models,sort_keys=True,separators=(',',':')).encode()).hexdigest(),models=models,lightgbm_version=lgb.__version__,calibration_fits=0,treatment_fits=0,model_predictions=0,calibration_probe='NOT_NEEDED: archived BASE ranker models available',no_label_or_outcome_selection=True)
 raw=json.dumps(receipt,indent=2)+'\n';(E/'minhess-derivation.json').write_text(raw);(D/'minhess-derivation.json').write_text(raw)
 small={k:v for k,v in receipt.items() if k not in ['models','source_archive']};small['derivation_receipt_sha256']=hashlib.sha256(raw.encode()).hexdigest();small['source_archive_sha256']=fetch['h316']['sha256'];(HERE/'minhess_value.json').write_text(json.dumps(small,indent=2)+'\n')
 print(json.dumps({k:v for k,v in small.items() if 'sha256' not in k}),flush=True)
 if status!='PASS':raise SystemExit(3)
if __name__=='__main__':main()
