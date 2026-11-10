"""Full36 CPU forecast from H352 actual CPU; no timing/calibration fits."""

from pathlib import Path
import datetime, json

HERE = Path(__file__).resolve().parent
STOP = datetime.datetime(2026, 11, 5, 15, 15, tzinfo=datetime.timezone.utc)


def cpu_reference():
    return (
        int(
            json.loads((HERE / "h352_actual_resources.json").read_text())[
                "CPUUsageNSec"
            ]
        )
        / 1e9
    )


def estimate(cap, start=None):
    start = start or datetime.datetime.now(datetime.timezone.utc)
    n = len(cap["eligible_cores"])
    assert 1 <= n <= 8
    cpu = cpu_reference()
    seconds = cpu / (n * 0.9) * 1.15 + 300
    end = start + datetime.timedelta(seconds=seconds)
    return dict(
        wave="H363",
        status="PREFIT_FORECAST",
        forecast_start=start.isoformat(),
        workers=n,
        eligible_cores=cap["eligible_cores"],
        mandatory_tasks=36,
        reference_wave="H352 measured full36",
        forecast_reference_cpu_seconds=cpu,
        parallel_efficiency=0.9,
        safety_factor=1.15,
        score_reserve_seconds=300,
        forecast_seconds=seconds,
        forecast_end=end.isoformat(),
        compute_stop=STOP.isoformat(),
        close_stop="2026-11-05T15:30:00Z",
        launch_full_36=n >= 4 and end < STOP,
        campaign_fits=0,
        synthetic_fits=0,
        BASE_refits=0,
        NL15_refits=0,
        outer_scores=0,
        capacity_retries=0,
        caveat="MCS10 cost unmeasured; no speedup assumed, no reduced design/reprobe/deadline reset",
    )


def after_first_cell(cell, workers, start=None):
    start = start or datetime.datetime.now(datetime.timezone.utc)
    baseline = cpu_reference()
    per = max(baseline / 36, cell["cpu_seconds"])
    seconds = 35 * per / (workers * 0.9) * 1.15 + 300
    end = start + datetime.timedelta(seconds=seconds)
    return dict(
        wave="H363",
        stage="FIRST_COMPLETED_CELL",
        forecast_start=start.isoformat(),
        first_completed_cell=cell,
        workers=workers,
        completed_tasks=1,
        remaining_tasks=35,
        baseline_reference_cpu_seconds=baseline,
        per_task_cpu_reference_seconds=per,
        parallel_efficiency=0.9,
        safety_factor=1.15,
        score_reserve_seconds=300,
        forecast_seconds=seconds,
        forecast_end=end.isoformat(),
        compute_stop=STOP.isoformat(),
        launch_full_36=workers >= 4 and end < STOP,
        outer_scores=0,
        note="Remaining35 at max(H352CPU/36,actualfirsttaskCPU); ongoing credit0; no new probe or reset",
    )
