"""One corrected pre-campaign capacity gate; reuse all completed controls, no refits."""
from pathlib import Path
import json,os,subprocess,sys,threading
import guard_h329 as guard
from resource_probe import probe
from forecast_h329 import estimate

HERE=Path(__file__).resolve().parent
P=HERE.parent
D=Path('/home/ubuntu/jope-data/h329-gov03')


def main():
    assert not list(D.glob('P*/fold_*/root_*/fit_lock.json'))
    assert not list(D.glob('P*/fold_*/root_*/dart_constructor_audit.jsonl'))
    assert json.loads((D/'regression.json').read_text())['status']=='PASS'
    assert json.loads((D/'timing_fixture.json').read_text())['synthetic_ranker_fits']==6
    threading.Thread(target=guard.monitor,daemon=True).start()
    subprocess.run([sys.executable,str(HERE/'input_preflight.py')],check=True)
    cores,idle,raw=probe();(P/'corrected-capacity-mpstat.json').write_text(raw)
    fixture=json.loads((D/'timing_fixture.json').read_text())
    forecast=estimate(fixture['forecast_multiplier'],len(cores)) if cores else dict(launch_full_36=False,reason='No eligible assigned core')
    forecast.update(eligible_cores=cores,both_smt_five_second_idle=idle,measurement='one5second interval; no lowest1second selection')
    (P/'runtime-forecast-corrected.json').write_text(json.dumps(forecast,indent=2)+'\n')
    (D/'runtime_forecast.json').write_text(json.dumps(forecast,indent=2)+'\n')
    print(json.dumps(dict(corrected_forecast=forecast)),flush=True)
    if not forecast['launch_full_36']:
        (D/'INCOMPLETE_RUNTIME_FORECAST_CORRECTED.json').write_text(json.dumps(forecast,indent=2)+'\n')
        raise SystemExit(2)
    env=dict(os.environ,H329_WORKERS=str(len(cores)))
    (D/'pre_fit_resource.json').write_text(json.dumps(dict(eligible_cores=cores,both_smt_idle=idle,measurement_seconds=5)))
    for phase in ['fit','score']:
        subprocess.run(['taskset','-c',','.join(map(str,cores)),sys.executable,str(HERE/'run_h329.py'),phase],env=env,check=True)


if __name__=='__main__':main()
