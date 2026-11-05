"""Read recovered H352 native text/audits only; no fitting or predictions."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,datetime
from native_audit import parse
P=Path(__file__).parent.parent;S=P/'source';D=Path('/home/ubuntu/jope-data/h392-hwv')
def main():
 fetch=json.loads((P/'nl15-fetch-receipt.json').read_text());assert fetch['status']=='EXACT_VERSION_GET_AND_ALL_MEMBERS_SHA_PASS'
 rows=[];rankers=[];vetos=[]
 for path in sorted((D/'nl15').glob('P*/fold_*/root_*/refit/*.txt')):
  rel=str(path.relative_to(D/'nl15'));raw=path.read_bytes();expected=fetch['saved_members'][rel];assert hashlib.sha256(raw).hexdigest()==expected['sha256'] and len(raw)==expected['bytes']
  native=parse(raw.decode());veto=path.name.endswith('-veto.txt')
  assert native['boosting']=='gbdt' and native['device_type']=='cpu' and native['num_threads']=='1'
  assert native['objective'].split()[0]==('binary' if veto else 'lambdarank')
  if veto:assert float(native['lambda_l1'])==1 and float(native['lambda_l2'])==2
  r=dict(file=rel,sha256=expected['sha256'],max_bin=int(native['max_bin']),num_leaves=int(native['num_leaves']),max_depth=int(native['max_depth']),lambda_l1=float(native['lambda_l1']),lambda_l2=float(native['lambda_l2']),min_data_in_leaf=int(native['min_data_in_leaf']))
  rows.append(r);(vetos if veto else rankers).append(r)
 assert len(rankers)==972 and len(vetos)==108
 spec=all(x['max_bin']==255 and x['num_leaves']==15 and x['max_depth']==3 and x['lambda_l1']==0 and x['lambda_l2']==1 and x['min_data_in_leaf']==20 for x in rankers)
 record=dict(status='PASS' if spec else 'INCOMPLETE_SPEC_MISMATCH',wave='H392',reference_wave='H352',effective_lambda_l1=sorted({x['lambda_l1'] for x in rankers}),effective_max_bin=sorted({x['max_bin'] for x in rankers}),effective_num_leaves=sorted({x['num_leaves'] for x in rankers}),effective_max_depth=sorted({x['max_depth'] for x in rankers}),rankers_verified=972,veto_models_verified=108,models=rows,optimizer_fits=0,new_model_predictions=0)
 (P/'nl15-native-config.json').write_text(json.dumps(record,indent=1)+'\n')
 assert spec,'INCOMPLETE_SPEC_MISMATCH before any treatment fit'
 values=defaultdict(lambda:defaultdict(set));seed_values=defaultdict(lambda:defaultdict(lambda:defaultdict(set)));counts=defaultdict(int)
 for path in sorted((D/'nl15').glob('P*/fold_*/root_*/native_fit_audit.jsonl')):
  for line in path.open():
   r=json.loads(line);kind=r['kind'];native=r['native_params'];seed=str(r['constructor_params']['random_state']);counts[kind]+=1
   if kind=='LGBMRanker':assert float(native['lambda_l1'])==0 and int(native['max_bin'])==255 and int(native['num_leaves'])==15 and int(native['max_depth'])==3
   for key,value in native.items():values[kind][key].add(value);seed_values[kind][seed][key].add(value)
 assert dict(counts)=={'LGBMRanker':29484,'LGBMClassifier':7020}
 ref={}
 for kind,fields in values.items():
  constants={k:next(iter(v)) for k,v in fields.items() if len(v)==1};variable=sorted(k for k,v in fields.items() if len(v)>1 and k!='num_iterations')
  ref[kind]=dict(constant_native_fields=constants,variable_native_fields=variable,seed_fields={seed:{k:sorted(v[k]) for k in variable} for seed,v in seed_values[kind].items()},ignored_dynamic_native_fields={'num_iterations':'Compared to every actual constructor n_estimators; selected rounds are a consequence of unchanged frozen selection rules.'})
 (S/'h352_native_reference.json').write_text(json.dumps(dict(status='PASS',reference='Recovered committed H352 full native fit records; only min_sum_hessian_in_leaf may change in rankers',counts=dict(counts),roles=ref),indent=1)+'\n')
 cpu=json.loads((D/'nl15/operational_evidence/UNIT-FINAL.json').read_text());assert cpu['ExecMainCode']=='1' and cpu['ExecMainStatus']=='0' and cpu['Result']=='success'
 seconds=int(cpu['CPUUsageNSec'])/1e9
 (S/'forecast_reference.json').write_text(json.dumps(dict(reference_wave='H352 actual full36',reference_cpu_seconds=seconds,reference_CPUUsageNSec=cpu['CPUUsageNSec'],source_member='h352-jrb01/operational_evidence/UNIT-FINAL.json',source_sha256=fetch['saved_members']['operational_evidence/UNIT-FINAL.json']['sha256'],fallback_H347_cpu_seconds=29770.57,fallback_used=False,measured_invocation=cpu['InvocationID']),indent=2)+'\n')
 print(json.dumps(dict(status='PASS',native_rankers=972,veto=108,full_native_fit_records=dict(counts),forecast_reference_cpu_seconds=seconds)),flush=True)
if __name__=='__main__':main()
