"""Read-only exact-version H352 recovery; retain only verified baseline metadata."""
from pathlib import Path
import datetime,hashlib,json,re,tarfile,sys,threading,os
import boto3,numpy as np
import guard_h371 as corrected_guard
E=Path(__file__).resolve().parent.parent;C=E/'committed-H352';D=Path('/home/ubuntu/jope-data/h371-gov03');N=D/'nl15'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def spec_zero(params,where):
 if 'path_smooth' not in params or float(params['path_smooth'])!=0.0:
  report=dict(status='INCOMPLETE_SPEC_MISMATCH',where=where,archived_H352_path_smooth=params.get('path_smooth'),optimizer_fits=0,capacity_probes=0)
  (E/'SPEC_MISMATCH.json').write_text(json.dumps(report,indent=2)+'\n');raise SystemExit(3)
def main():
 D.mkdir(exist_ok=True);N.mkdir(exist_ok=False);threading.Thread(target=corrected_guard.monitor,args=(os.environ.get('H371_OWN_UNIT','h371-nl15-prepare2.service'),),daemon=True).start()
 reference=json.loads((C/'metrics.json').read_text());ref=reference['archive_manifest']
 session=boto3.Session(profile_name='veox-jepson-prod');assert session.client('sts').get_caller_identity()['Account']=='714044927654';s3=session.client('s3',region_name='us-east-2')
 get=s3.get_object(Bucket=ref['bucket'],Key=ref['key'],VersionId=ref['version_id']);raw=get['Body'].read();get['Body'].close();assert get['VersionId']==ref['version_id'] and sha(raw)==ref['sha256'];(E/'nl15-closeout-manifest.json').write_bytes(raw);manifest=json.loads(raw)
 native={};reviews={};parts=[];all_names=set();retained=[];headers={};audit_files=0;native_calls=0
 for i,part in enumerate(manifest['parts']):
  tmp=D/'nl15-part.tar.gz'
  if tmp.exists():
   assert i==0 and sha(tmp.read_bytes())==part['sha256'] and tmp.stat().st_size==part['bytes']
  else:
   get=s3.get_object(Bucket=part['bucket'],Key=part['key'],VersionId=part['version_id']);assert get['VersionId']==part['version_id'];h=hashlib.sha256();size=0
   with tmp.open('xb') as f:
    for b in iter(lambda:get['Body'].read(4*1024*1024),b''):f.write(b);h.update(b);size+=len(b)
   get['Body'].close();assert h.hexdigest()==part['sha256'] and size==part['bytes']
  seen=set()
  with tarfile.open(tmp,'r:gz') as tf:
   for member in tf:
    assert member.isfile() and member.name in part['members'] and member.name not in all_names and member.name.startswith('h352-jrb01/')
    raw=tf.extractfile(member).read();expected=part['members'][member.name];assert len(raw)==expected['bytes'] and sha(raw)==expected['sha256'];seen.add(member.name);all_names.add(member.name);rel=member.name[len('h352-jrb01/'):]
    if re.fullmatch(r'P[012]/fold_[0-3]/root_3000[123]/refit/[^/]+\.txt',rel):
     params=dict(re.findall(r'^\[([^:\]]+): (.*)\]$',raw.decode(),re.M));veto=rel.endswith('-veto.txt')
     assert params['boosting']=='gbdt' and params['device_type']=='cpu' and params['num_threads']=='1'
     assert float(params['lambda_l1'])==(1 if veto else 0) and float(params['lambda_l2'])==(2 if veto else 1)
     assert int(params['min_data_in_leaf'])==(25 if veto else 20) and float(params['min_gain_to_split'])==0 and float(params['min_sum_hessian_in_leaf'])==.001 and float(params['sigmoid'])==1
     if not veto:assert params['max_depth']=='3' and params['objective'].split()[0]=='lambdarank' and params['metric']=='ndcg' and int(params['num_leaves'])==15 and float(params['learning_rate']) in [.03,.05,.1] and int(params['num_iterations']) in [50,100,200]
     if not veto:spec_zero(params,rel)
     native[rel]=dict(sha256=sha(raw),native_params=params,role='veto' if veto else 'ranker')
    if rel.endswith('/native_fit_audit.jsonl'):
     audit=[json.loads(line) for line in raw.decode().splitlines()];assert len(audit)==1014;audit_files+=1
     for row in audit:
      kind=row['kind'];n=row['native_params'];key=kind+':'+n['seed']
      if kind=='LGBMRanker':spec_zero(n,rel)
      canonical={k:v for k,v in n.items() if kind=='LGBMClassifier' or k not in ('learning_rate','num_iterations')}
      if key in headers:assert headers[key]==canonical,(key,'noncanonical archived native header')
      headers[key]=canonical;native_calls+=1
    if rel.endswith('/native_model_review.json'):
     record=json.loads(raw)
     for model in record['models']:
      if model['file'].startswith('refit/'):reviews[str(Path(rel).parent/model['file'])]=model
    keep=('/' not in rel and (rel.startswith('NL15_') or rel in ['metrics.json','picks.npz','raw_family_gains.npz','regression.json','fit_done.json','lock_review.json'])) or bool(re.fullmatch(r'P[012]/fold_[0-3]/root_3000[123]/(fit_lock|score_once|guard_receipt)\.json',rel))
    if keep:
     p=N/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);retained.append(rel)
  assert seen==set(part['members']);parts.append(dict(key=part['key'],version_id=part['version_id'],sha256=part['sha256'],members=len(seen),whole_and_every_member_verified=True));tmp.unlink();print(json.dumps(dict(part=i+1,total_parts=27,members_verified=len(all_names))),flush=True)
 assert len(parts)==27 and len(all_names)==2691 and set(native)==set(reviews)
 for name,r in native.items():
  model=reviews[name];assert r['sha256']==model['sha256'] and r['native_params']==model['native_parameters']
 assert sum(r['role']=='ranker' for r in native.values())==972 and sum(r['role']=='veto' for r in native.values())==108
 expected_veto=json.loads((C/'base_veto_sha256.json').read_text())['members']
 for name,r in native.items():
  if r['role']=='veto':assert expected_veto['run/'+name]['sha256']==r['sha256']
 fixture=next((r,reviews[name]) for name,r in native.items() if name.endswith('-ranker.txt'))
 (E/'nl15-native-reference.json').write_text(json.dumps(dict(constructor_params=fixture[1]['constructor_parameters'],native_params=fixture[0]['native_params'],model_sha256=fixture[0]['sha256']),indent=2)+'\n')
 (E/'nl15-native-inventory.json').write_text(json.dumps(native,indent=2)+'\n')
 locks=0
 for part in ['P0','P1','P2']:
  panel=np.load(N/f'NL15_{part}.npy');assert panel.shape==(3,1219);pins=json.loads((N/f'NL15_{part}_locks.json').read_text());assert pins==reference['fit_lock_sha256']['NL15_'+part]
  for k in range(4):
   for j,seed in enumerate([30001,30002,30003]):
    td=N/part/f'fold_{k}'/f'root_{seed}';h=sha((td/'fit_lock.json').read_bytes());marker=json.loads((td/'score_once.json').read_text());assert pins[f'{k}:{seed}']==h and marker['fit_lock_sha256']==h and marker['state']=='COMPLETE';locks+=1
  for j,seed in enumerate([30001,30002,30003]):
   cell=next(x for x in reference['cells'] if x['partition']==part and x['root_seed']==seed and x['arm']=='NL15');assert sha(np.asarray(panel[j],dtype=np.int32).tobytes())==cell['picks_sha256']
 with np.load(N/'raw_family_gains.npz') as z:
  for arm in ['BASE','NL15']:
   assert z[arm].shape==(9,205)
   for values,cell in zip(z[arm],[x for x in reference['cells'] if x['arm']==arm]):np.testing.assert_allclose(values.mean(),cell['family_mean_gain'],rtol=0,atol=2e-14)
 close=Path('/home/ubuntu/jope_plan/evidence/H352/jrb01-20261105-t0755/CLOSEOUT.json');raw=close.read_bytes();assert sha(raw)=='042905333cc018c3baef067f670a86261ef6f0919d732f676c92465c62256eeb';unit=json.loads(raw)['unit_final'];cpu=int(unit['CPUUsageNSec'])/1e9;assert cpu==26189.516914 and unit['Result']=='success' and unit['ExecMainStatus']=='0'
 (E/'h352_actual_resources.json').write_text(json.dumps(dict(cpu_usage_nsec=int(unit['CPUUsageNSec']),cpu_seconds=cpu,invocation_id=unit['InvocationID'],closeout_sha256=sha(raw),ranker_cpu_share_upper_bound=1.0,forecast_reference_cpu_seconds=cpu,reason='Owner H371 forecast uses actual H352 CPU unchanged; PS20 runtime unmeasured; no timing fits'),indent=2)+'\n')
 assert audit_files==36 and native_calls==36504
 (E/'nl15-native-headers.json').write_text(json.dumps(headers,sort_keys=True,separators=(',',':'))+'\n')
 result=dict(status='PASS',wave='H371',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference_commit='02dfee2f7e0b8543324494dd4f99d074cd6b93b6',manifest=ref,parts=parts,members_verified=len(all_names),panels=9,locks=locks,score_markers=locks,rankers_verified=972,veto_SHA_identities_verified=108,native_inventory_sha256=sha((E/'nl15-native-inventory.json').read_bytes()),native_path_smooth=[0.0],native_fit_audits=audit_files,native_fits_verified=native_calls,full_header_reference_sha256=sha((E/'nl15-native-headers.json').read_bytes()),retained=retained,BASE_refits=0,NL15_refits=0,model_predictions=0)
 (E/'nl15-prepared.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['parts','retained']}),flush=True)
if __name__=='__main__':main()
