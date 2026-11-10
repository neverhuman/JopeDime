"""Read archived native BASE defaults without fitting or predicting."""
from pathlib import Path
import datetime,hashlib,json,re
import lightgbm
from native_audit import parse
HERE=Path(__file__).resolve().parent;P=HERE.parent;D=P

def main():
 assert not (HERE/'objective_value.json').exists()
 fetch=json.loads((P/'base-fetch-receipt.json').read_text());assert fetch['status']=='EXACT_VERSION_GET_AND_SHA_PASS'
 assert lightgbm.__version__=='4.6.0'
 models=[];objectives=set();sigmoids=set()
 for name,info in sorted(fetch['h316']['members'].items()):
  if not re.fullmatch(r'run/P[012]/fold_[0-3]/root_3000[1-3]/refit/[^/]+\.txt',name) or 'veto' in name:continue
  raw=(D/'base_native'/name[4:]).read_bytes();assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256']
  native=parse(raw.decode());assert native['boosting']=='gbdt' and native['device_type']=='cpu' and native['num_threads']=='1'
  assert float(native['min_sum_hessian_in_leaf'])==.001 and native['label_gain']==''
  objective=native['objective'].split()[0];objectives.add(objective);sigmoids.add(float(native['sigmoid']))
  models.append(dict(member=name,sha256=info['sha256'],objective=objective,sigmoid=float(native['sigmoid'])))
 assert len(models)==972
 ok=objectives=={'lambdarank'} and sigmoids=={1.0}
 r=dict(wave='H349',status='PASS' if ok else 'INCOMPLETE_SPEC_MISMATCH',utc=datetime.datetime.now(datetime.UTC).isoformat(),XENDCG='rank_xendcg',BASE_effective_native_objective='lambdarank' if objectives=={'lambdarank'} else sorted(objectives),BASE_effective_native_sigmoid=1.0 if sigmoids=={1.0} else sorted(sigmoids),model_count=972,lightgbm_version=lightgbm.__version__,native_model_inventory_sha256=hashlib.sha256(json.dumps(models,sort_keys=True,separators=(',',':')).encode()).hexdigest(),source_archive_sha256=fetch['h316']['sha256'],calibration_fits=0,treatment_fits=0,model_predictions=0,method='All972 exact-version archived refit rankers; fixed rank_xendcg with objective_seed=original random_state, no calibration/tuning')
 for p in [P/'native-base-objective.json',HERE/'objective_value.json']:p.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r),flush=True)
 if not ok:raise RuntimeError('INCOMPLETE_SPEC_MISMATCH')
if __name__=='__main__':main()
