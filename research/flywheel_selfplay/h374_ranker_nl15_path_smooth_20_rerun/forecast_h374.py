"""Full36 admission uses H352 measured CPU; no H374 timing fits."""
from pathlib import Path
import datetime,json
HERE=Path(__file__).resolve().parent
STOP=datetime.datetime(2026,11,5,17,30,tzinfo=datetime.timezone.utc)
def estimate(cap,start=None):
 start=start or datetime.datetime.now(datetime.timezone.utc)
 workers=len(cap['eligible_cores']);assert 1<=workers<=8
 cpu=json.loads((HERE/'h352_actual_resources.json').read_text())['forecast_reference_cpu_seconds']
 seconds=cpu/(workers*.9)*1.15+300
 end=start+datetime.timedelta(seconds=seconds)
 return dict(wave='H374',status='PREFIT_FORECAST',forecast_start=start.isoformat(),workers=workers,eligible_cores=cap['eligible_cores'],mandatory_tasks=36,reference_wave='H352 measured26189.516914 CPU seconds',forecast_reference_cpu_seconds=cpu,parallel_efficiency=.9,safety_factor=1.15,score_reserve_seconds=300,forecast_seconds=seconds,forecast_end=end.isoformat(),compute_stop=STOP.isoformat(),close_stop='2026-11-05T17:45:00Z',launch_full_36=workers>=4 and end<STOP,campaign_fits=0,synthetic_fits=0,BASE_refits=0,outer_scores=0,capacity_retries=0,caveat='PS20 cost is unmeasured; no acceleration assumed, partial design, borrowed cores, deadline reset or repeated capacity sampling.')
def after_first_cell(cell,workers,start=None):
 start=start or datetime.datetime.now(datetime.timezone.utc)
 baseline=json.loads((HERE/'h352_actual_resources.json').read_text())['forecast_reference_cpu_seconds']
 per_task=max(baseline/36,cell['cpu_seconds'])
 remaining_seconds=35*per_task/(workers*.9)*1.15+300
 end=start+datetime.timedelta(seconds=remaining_seconds)
 return dict(wave='H374',stage='FIRST_COMPLETED_CELL',forecast_start=start.isoformat(),first_completed_cell=cell,workers=workers,completed_tasks=1,remaining_tasks=35,baseline_reference_cpu_seconds=baseline,per_task_cpu_reference_seconds=per_task,parallel_efficiency=.9,safety_factor=1.15,score_reserve_seconds=300,forecast_seconds=remaining_seconds,forecast_end=end.isoformat(),compute_stop=STOP.isoformat(),launch_full_36=workers>=4 and end<STOP,outer_scores=0,note='Conservative remaining35CPU tasks; ongoing work credit0; reference=max(H352CPU/36,first completed cell actualCPU). No second capacity probe.')
if __name__=='__main__':raise SystemExit('Called once by controller after its single capacity probe')
