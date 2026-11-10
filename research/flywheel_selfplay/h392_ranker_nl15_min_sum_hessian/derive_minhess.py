"""One read-only percentile over every archived H352 NL15 ranker leaf."""
from pathlib import Path
from array import array
import datetime,hashlib,io,json,math,os,re,sys,tarfile,time,threading
import numpy as np

P=Path(__file__).resolve().parent.parent;S=P/'source';D=Path('/home/ubuntu/jope-data/h392-hwv')
def leaf_weights(node):
 if 'left_child' in node:
  yield from leaf_weights(node['left_child']);yield from leaf_weights(node['right_child'])
 else:
  value=float(node['leaf_weight']);assert math.isfinite(value) and value>=0
  yield value
def pooled_value(values):
 values=np.asarray(values,dtype=np.float64);assert values.size and np.isfinite(values).all() and (values>=0).all()
 q=float(np.quantile(values,.25,method='linear'));return q,float(format(q,'.3g'))
class Reader:
 def __init__(self,body):self.body=body;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  b=self.body.read(n);self.h.update(b);self.n+=len(b);return b
def main():
 assert os.uname().nodename.split('.')[0]=='xbabe2' and os.getpriority(os.PRIO_PROCESS,0)==19
 assert not (S/'hessian_value.json').exists(),'The single value must never be recomputed/reselected'
 import guard_h392 as guard
 threading.Thread(target=lambda:guard.monitor('h392-derive.service'),daemon=True).start()
 while not (P/'reference-unit-exit.json').exists():
  assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,11,6,0,45,tzinfo=datetime.timezone.utc)
  time.sleep(2)
 assert json.loads((P/'reference-unit-exit.json').read_text())['EXIT_STATUS']=='0'
 fetch=json.loads((P/'nl15-fetch-receipt.json').read_text());assert fetch['status']=='EXACT_VERSION_GET_AND_ALL_MEMBERS_SHA_PASS'
 manifest=json.loads((D/'nl15/recovery-manifest.json').read_text());assert hashlib.sha256((D/'nl15/recovery-manifest.json').read_bytes()).hexdigest()==fetch['manifest']['sha256']
 import lightgbm as lgb
 import archive_h392 as ar
 from native_audit import parse
 weights=array('d');models=[];baselines=set();inner_receipts={};inner_tars=set();parts=[]
 def model(raw,name,source):
  assert not name.endswith('-veto.txt');native=parse(raw.decode());assert native['objective'].split()[0]=='lambdarank' and native['device_type']=='cpu' and native['num_threads']=='1'
  assert int(native['num_leaves'])==15 and int(native['max_depth'])==3 and int(native['max_bin'])==255 and float(native['lambda_l1'])==0 and float(native['lambda_l2'])==1 and float(native['path_smooth'])==0 and int(native['bagging_freq'])==1
  baselines.add(float(native['min_sum_hessian_in_leaf']));b=lgb.Booster(model_str=raw.decode());dump=b.dump_model();start=len(weights)
  for tree in dump['tree_info']:weights.extend(leaf_weights(tree['tree_structure']))
  assert len(weights)>start
  models.append(dict(file=name,sha256=hashlib.sha256(raw).hexdigest(),trees=len(dump['tree_info']),leaves=len(weights)-start,source=source,native_baseline=float(native['min_sum_hessian_in_leaf'])))
 final=list(sorted((D/'nl15').glob('P*/fold_*/root_*/refit/*.txt')))
 for path in final:
  if path.name.endswith('-veto.txt'):continue
  rel=str(path.relative_to(D/'nl15'));raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==fetch['saved_members'][rel]['sha256'];model(raw,rel,'verified recovered final native model')
 assert len(models)==972
 for part in manifest['parts']:
  if not any(re.fullmatch(r'h352-jrb01/P[012]/fold_[0-3]/root_3000[123]/inner_(archive.json|members.tar.gz)',n) for n in part['members']):continue
  response=ar.client().get_object(Bucket=part['bucket'],Key=part['key'],VersionId=part['version_id']);assert response['VersionId']==part['version_id'];reader=Reader(response['Body']);seen=set()
  with tarfile.open(fileobj=reader,mode='r|gz') as tar:
   for member in tar:
    assert member.isfile() and member.name in part['members'] and member.name not in seen;seen.add(member.name)
    selected=bool(re.fullmatch(r'h352-jrb01/P[012]/fold_[0-3]/root_3000[123]/inner_(archive.json|members.tar.gz)',member.name));h=hashlib.sha256();n=0;raw=bytearray() if selected else None;f=tar.extractfile(member)
    for chunk in iter(lambda:f.read(1024*1024),b''):
     h.update(chunk);n+=len(chunk)
     if raw is not None:raw.extend(chunk)
    expected=part['members'][member.name];assert h.hexdigest()==expected['sha256'] and n==expected['bytes']
    if not selected:continue
    task=str(Path(member.name).parent.relative_to('h352-jrb01'))
    if member.name.endswith('inner_archive.json'):
     inner_receipts[task]=json.loads(raw);continue
    receipt=inner_receipts[task];assert receipt['sha256']==h.hexdigest() and receipt['exact_version_get_sha256_verified'];inner_tars.add(task);inner_seen=set();rankers=0
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as inner:
     for entry in inner:
      assert entry.isfile() and entry.name not in inner_seen;inner_seen.add(entry.name);payload=inner.extractfile(entry).read();assert hashlib.sha256(payload).hexdigest()==receipt['members'][entry.name]
      if entry.name.endswith('.txt') and not entry.name.endswith('-veto.txt'):model(payload,task+'/'+entry.name,dict(outer_key=part['key'],outer_version_id=part['version_id'],inner_member=member.name));rankers+=1
    assert inner_seen==set(receipt['members']) and rankers==324
  while reader.read(1024*1024):pass
  response['Body'].close();assert seen==set(part['members']) and reader.h.hexdigest()==part['sha256'] and reader.n==part['bytes']
  parts.append({k:v for k,v in part.items() if k!='members'});print(json.dumps(dict(verified_part=part['key'].rsplit('/',1)[-1],rankers=len(models),leaves=len(weights))),flush=True)
 assert len(inner_tars)==36 and len(models)==12636 and len({m['file'] for m in models})==12636 and len(baselines)==1
 baseline=next(iter(baselines));values=np.frombuffer(weights,dtype=np.float64);q,value=pooled_value(values)
 inventory=D/'hessian_model_inventory.json';inventory.write_text(json.dumps(models,indent=1)+'\n')
 record=dict(wave='H392',status='PASS' if value>baseline else 'INCOMPLETE_SPEC_MISMATCH',parameter='min_sum_hessian_in_leaf',sklearn_alias='min_child_weight',baseline=baseline,value=value,raw_q25=q,percentile=25,quantile_method='numpy.quantile(method=linear)',rounding='Python format .3g, then float; three significant figures; exactly one value computed',scope='ALL archived H352 NL15 ranker native models: 972 final + 11664 inner, excluding BASE copies and all veto classifiers',models=len(models),final_models=972,inner_models=11664,inner_archives=36,trees=sum(m['trees'] for m in models),leaves=len(weights),pooled_leaf_weights_sha256=hashlib.sha256(values.astype('<f8',copy=False).tobytes()).hexdigest(),pooled_leaf_weights_dtype='<f8 in sorted final then sorted archive/member/tree DFS order',model_inventory=str(inventory.relative_to(D)),model_inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),min_leaf_weight=float(values.min()),max_leaf_weight=float(values.max()),all_native_baselines=sorted(baselines),derivation='Read-only LightGBM Booster(model_str).dump_model tree_structure leaf_weight; every leaf equally weighted, no family/model normalization',archived_manifest=fetch['manifest'],exact_version_parts=parts,whole_and_every_member_verified=True,calibration_fits=0,optimizer_fits=0,new_predictions=0,computed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (S/'hessian_value.json').write_text(json.dumps(record,indent=2)+'\n');(P/'hessian-value.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('exact_version_parts','archived_manifest')}),flush=True)
 if value<=baseline:raise SystemExit(2)
if __name__=='__main__':main()
