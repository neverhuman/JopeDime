"""Fixed synthetic timing fixture; complete36 forecast precedes any campaign fit."""
from pathlib import Path
import datetime,json,time,gc,os
import numpy as np

HERE=Path(__file__).resolve().parent
P=HERE.parent
DATA=Path('/home/ubuntu/jope-data/h329-gov03')


def estimate(ratio,workers,start=None):
    old=json.loads((HERE/'runtime_forecast_initial.json').read_text())
    start=start or datetime.datetime.now(datetime.timezone.utc)
    cpu=old['reference_cpu_seconds']*(.1+.9*ratio)
    seconds=cpu/(workers*.9)*1.15+300
    end=start+datetime.timedelta(seconds=seconds)
    deadline=datetime.datetime(2026,11,4,20,45,tzinfo=datetime.timezone.utc)
    return dict(workers=workers,ranker_multiplier=ratio,reference_cpu_seconds=old['reference_cpu_seconds'],
                ranker_cpu_share_assumption=.9,parallel_efficiency=.9,safety_factor=1.15,
                full36_forecast_seconds=seconds,score_reserve_seconds=300,forecast_start=start.isoformat(),
                forecast_end=end.isoformat(),compute_stop=deadline.isoformat(),
                launch_full_36=end<=deadline-datetime.timedelta(seconds=120),
                caveat='Synthetic timing ratio is a planning estimate; actual TRAIN runtime remains uncertain. All36 tasks remain mandatory.')


def main():
    assert json.loads((P/'prereg-published.json').read_text())['published']
    assert json.loads((HERE/'runtime_forecast_initial.json').read_text())['provisional_fits']
    assert json.loads((DATA/'regression.json').read_text())['status']=='PASS'
    assert json.loads((P/'input_identity_validation.json').read_text())['status']=='PASS'
    assert not (DATA/'timing_fixture.json').exists()
    import run_h329 as r
    import dart_patch
    model=r.initialize();base=model.G['topk'].fit_ranker;dart,_=dart_patch.patched_function(base)
    # No real corpus features, labels or members are used by this fixture.
    rng=np.random.default_rng(329000);groups=512;k=8
    X=rng.normal(size=(groups*k,568)).astype(np.float32)
    y=np.concatenate([rng.permutation(k) for _ in range(groups)]).astype(np.int32)
    rows=[]
    for i,seed in enumerate([329001,329002,329003]):
        result={}
        for name in (['GBDT','DART'] if i%2==0 else ['DART','GBDT']):
            fun=base if name=='GBDT' else dart
            t=time.monotonic();obj=fun(X,y,groups,k,.05,seed);seconds=time.monotonic()-t
            params=obj.get_params();assert params['boosting_type']==('gbdt' if name=='GBDT' else 'dart')
            assert params['n_estimators']==200 and params['n_jobs']==1
            if name=='DART':assert params['drop_seed']==seed
            result[name]=seconds
            out=DATA/'timing_fixture';out.mkdir(exist_ok=True)
            obj.booster_.save_model(str(out/f'{seed}-{name}.txt'))
            del obj;gc.collect()
        result.update(seed=seed,ratio=result['DART']/result['GBDT']);rows.append(result)
    record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind='SYNTHETIC_TIMING_ONLY',
                synthetic_ranker_fits=6,campaign_cells_fitted=0,BASE_refits=0,
                data_rng=329000,shape=list(X.shape),groups=groups,rounds=200,learning_rate=.05,
                fixed_pairs=rows,forecast_multiplier=max(x['ratio'] for x in rows),
                selection='worst of3 paired fixed-seed ratios; no parameter tuning')
    r.durable(DATA/'timing_fixture.json',record)
    forecast=estimate(record['forecast_multiplier'],6);forecast['timing_fixture']=record
    r.durable(DATA/'runtime_forecast.json',forecast)
    r.durable(P/'runtime-forecast-refined.json',forecast)
    print(json.dumps(forecast),flush=True)


if __name__=='__main__':main()
