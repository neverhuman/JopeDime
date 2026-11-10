"""Scaled H352 measured CPU; conservative 2x ranker-share assumption, no timing fits."""
from pathlib import Path
import datetime,json
HERE=Path(__file__).resolve().parent
STOP=datetime.datetime(2026,11,5,23,15,tzinfo=datetime.timezone.utc)
BASE_RANKER_FITS=819;BASE_VETO_FITS=195
RANKER_COST_MULTIPLIER=2.0
COST_MULTIPLIER=(RANKER_COST_MULTIPLIER*BASE_RANKER_FITS+BASE_VETO_FITS)/(BASE_RANKER_FITS+BASE_VETO_FITS)
def reference():
 raw=json.loads((HERE/'actual_resources_reference.json').read_text())['reference_cpu_seconds']
 return raw,raw*COST_MULTIPLIER
def estimate(cap,start=None):
 start=start or datetime.datetime.now(datetime.timezone.utc);workers=len(cap['eligible_cores']);assert 1<=workers<=8
 raw,cpu=reference();seconds=cpu/(workers*.9)*1.15+300;end=start+datetime.timedelta(seconds=seconds)
 return dict(wave='H389',status='PREFIT_FORECAST',forecast_start=start.isoformat(),workers=workers,eligible_cores=cap['eligible_cores'],mandatory_tasks=36,reference_wave='H352 actual full36',baseline_actual_cpu_seconds=raw,forecast_reference_cpu_seconds=cpu,ranker_cost_multiplier=RANKER_COST_MULTIPLIER,total_cost_multiplier=COST_MULTIPLIER,share_assumption='CPU share approximated by native fit counts819ranker/195veto; double all ranker cost, fixed veto. This is a conservative preregistered proxy, not a measured CPU decomposition. Actual extra bags add312ranker fits/task,1131 vs819; all non-bag model fits stay fixed.',parallel_efficiency=.9,safety_factor=1.15,score_reserve_seconds=300,forecast_seconds=seconds,forecast_end=end.isoformat(),compute_stop=STOP.isoformat(),close_stop='2026-11-05T23:30:00Z',launch_full_36=workers>=4 and end<STOP,campaign_fits=0,synthetic_fits=0,BASE_refits=0,outer_scores=0,capacity_retries=0,caveat='No acceleration, cut cells, borrowed cores, reset or repeated sampling. Mandatory first-task measured forecast.')
def after_first_cell(cell,workers,start=None):
 start=start or datetime.datetime.now(datetime.timezone.utc);raw,cpu=reference();per_task=max(cpu/36,cell['cpu_seconds']);seconds=35*per_task/(workers*.9)*1.15+300;end=start+datetime.timedelta(seconds=seconds)
 return dict(wave='H389',stage='FIRST_COMPLETED_CELL',forecast_start=start.isoformat(),first_completed_cell=cell,workers=workers,completed_tasks=1,remaining_tasks=35,baseline_actual_cpu_seconds=raw,baseline_reference_cpu_seconds=cpu,total_cost_multiplier=COST_MULTIPLIER,per_task_cpu_reference_seconds=per_task,parallel_efficiency=.9,safety_factor=1.15,score_reserve_seconds=300,forecast_seconds=seconds,forecast_end=end.isoformat(),compute_stop=STOP.isoformat(),launch_full_36=workers>=4 and end<STOP,outer_scores=0,note='remaining35CPU tasks; ongoingcredit0; max(scaled H352actualCPU/36,firstactualCPU), fixed deadline/no second probe.')
if __name__=='__main__':raise SystemExit('Called only by controller and first completed cell')
