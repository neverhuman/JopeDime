"""Read native BASE defaults from exact-version copies; no fits/predictions."""
from pathlib import Path
import datetime,hashlib,json,re
import lightgbm
from native_audit import parse
HERE=Path(__file__).resolve().parent;P=HERE.parent;D=Path('/home/ubuntu/jope-data/h344-jrb01')

def main():
    assert not (HERE/'sigmoid_value.json').exists()
    fetch=json.loads((P/'base-fetch-receipt.json').read_text())
    assert fetch['status']=='EXACT_VERSION_GET_AND_SHA_PASS'
    assert lightgbm.__version__=='4.6.0'
    models=[];sigmoids=set()
    for name,info in sorted(fetch['h316']['members'].items()):
        if not re.fullmatch(r'run/P[012]/fold_[0-3]/root_3000[1-3]/refit/[^/]+\.txt',name) or 'veto' in name:continue
        path=D/'base_native'/name[4:];raw=path.read_bytes()
        assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256']
        native=parse(raw.decode())
        assert native['objective'].split()[0]=='lambdarank' and native['boosting']=='gbdt'
        assert native['device_type']=='cpu' and native['num_threads']=='1'
        assert float(native['min_sum_hessian_in_leaf'])==0.001 and native['label_gain']==''
        sigmoids.add(float(native['sigmoid']))
        models.append(dict(member=name,sha256=info['sha256'],sigmoid=float(native['sigmoid'])))
    assert len(models)==972
    status='PASS' if sigmoids=={1.0} else 'INCOMPLETE_SPEC_MISMATCH'
    receipt=dict(wave='H344',status=status,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        SIGMA=0.5,BASE_effective_native_sigmoid=1.0 if sigmoids=={1.0} else sorted(sigmoids),
        model_count=972,lightgbm_version=lightgbm.__version__,
        native_model_inventory_sha256=hashlib.sha256(json.dumps(models,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        source_archive_sha256=fetch['h316']['sha256'],calibration_fits=0,treatment_fits=0,model_predictions=0,
        method='Native [sigmoid] in all972 archived BASE refit rankers; fixed0.5 chosen by owner, no calibration/tuning')
    (P/'native-base-sigmoid.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (HERE/'sigmoid_value.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if status!='PASS':raise RuntimeError('INCOMPLETE_SPEC_MISMATCH: BASE native sigmoid is not1.0')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
