"""Complete36 forecast from retained GBDT timing, before any H334 fit."""
from pathlib import Path
import datetime,json
HERE=Path(__file__).resolve().parent
STOP=datetime.datetime(2026,11,4,23,59,tzinfo=datetime.timezone.utc)

def estimate(cap,start=None):
 start=start or datetime.datetime.now(datetime.timezone.utc)
 fixture=json.loads((HERE/'h329_timing.json').read_text());reference=json.loads((HERE/'gbdt_reference.json').read_text());resources=json.loads((HERE/'h329_resources.json').read_text())
 workers=len(cap['eligible_cores']);assert 1<=workers<=8
 ratio=max(x['DART']/x['GBDT'] for x in fixture['fixed_pairs'])
 # Same90%ranker CPU-share assumption as H329. This is an inferred GBDT
 # equivalent, not measured PSMOOTH runtime or a performance claim.
 measured_dart_cpu=resources['cpu_usage_nsec']/1e9
 inferred_gbdt=measured_dart_cpu/(.1+.9*ratio)
 observed_reference=reference['reference_cpu_seconds']
 cpu=max(observed_reference,inferred_gbdt)
 seconds=cpu/(workers*.9)*1.15+300
 end=start+datetime.timedelta(seconds=seconds)
 optimistic=start+datetime.timedelta(seconds=min(observed_reference,inferred_gbdt)/workers)
 return dict(wave='H334',status='PREFIT_FORECAST',forecast_start=start.isoformat(),workers=workers,eligible_cores=cap['eligible_cores'],mandatory_tasks=36,
 h329_fixed_gbdt_seconds=[x['GBDT'] for x in fixture['fixed_pairs']],h329_worst_dart_to_gbdt_ratio=ratio,h329_measured_dart_total_cpu_seconds=measured_dart_cpu,ranker_cpu_share_assumption=.9,h329_inferred_gbdt_cpu_seconds=inferred_gbdt,
 measured_full36_gbdt_reference_cpu_seconds=observed_reference,reference_note=reference['reference_wave'],forecast_reference_cpu_seconds=cpu,
 path_smooth_multiplier_assumption=1.0,parallel_efficiency=.9,safety_factor=1.15,score_reserve_seconds=300,forecast_seconds=seconds,forecast_end=end.isoformat(),optimistic_baseline_reference_end=optimistic.isoformat(),optimistic_note='Smaller existing GBDT-equivalent reference, perfect efficiency, no safety, no scoring reserve; comparison only. Neither projection is a measured H334 runtime or a rigorous runtime lower bound.',compute_stop=STOP.isoformat(),close_stop='2026-11-05T00:15:00Z',launch_full_36=workers>=4 and end<STOP,campaign_fits=0,synthetic_fits=0,BASE_refits=0,outer_scores=0,capacity_retries=0,caveat='PSMOOTH runtime not measured; no acceleration assumed, no reduced design, borrowed cores, deadline reset or repeated capacity sampling.')

if __name__=='__main__':
 raise SystemExit('Forecast is called once from controller after its single capacity probe')
