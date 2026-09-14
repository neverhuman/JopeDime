"""Time-weighted task occupancy; CUDA memory residency is not active work."""
import math


def update(history, now, ranks, eligible, kernel_mean=None, explanation=None):
    if type(now) not in (int, float) or not math.isfinite(now):
        raise ValueError('Finite observation timestamp required')
    if ranks is not None and (type(ranks) is not int or not 0 <= ranks <= 8):
        raise ValueError('Observed rank count must be an integer in 0..8 or unknown')
    if type(eligible) is not bool:
        raise ValueError('Eligibility observation must be Boolean')
    if kernel_mean is not None and (type(kernel_mean) not in (int, float)
            or not math.isfinite(kernel_mean) or not 0 <= kernel_mean <= 100):
        raise ValueError('Kernel utilization must be a finite percentage or unknown')
    samples=history.setdefault('samples',[])
    previous=samples[-1] if samples else None
    if previous and now <= previous['timestamp']:
        return summarize(history,now)
    if previous and previous['ranks'] and ranks == 0:
        history['idle_started']=now
    if ranks and history.get('idle_started') is not None:
        history.setdefault('handoffs',[]).append(dict(timestamp=now,seconds=now-history.pop('idle_started')))
    samples.append(dict(timestamp=now,ranks=ranks,eligible=eligible,kernel_mean=kernel_mean,explanation=explanation))
    while len(samples)>1 and samples[1]['timestamp'] < now-3600:
        samples.pop(0)
    history['handoffs']=[row for row in history.get('handoffs',[]) if row['timestamp']>=now-3600]
    return summarize(history,now)


def summarize(history,now):
    samples=history.get('samples',[])
    elapsed=eligible=occupied=unknown=kernel_seconds=kernel_total=0.
    for before,after in zip(samples,samples[1:]):
        dt=after['timestamp']-max(before['timestamp'],now-3600)
        if dt<=0:continue
        elapsed+=dt
        if after['timestamp']-before['timestamp']>30 or before['ranks'] is None:
            unknown+=dt;continue
        if before['eligible']:
            eligible+=dt;occupied+=dt*before['ranks']
        if before['kernel_mean'] is not None:
            kernel_seconds+=dt;kernel_total+=dt*before['kernel_mean']
    gaps=sorted(row['seconds'] for row in history.get('handoffs',[]))
    p95=gaps[math.ceil(.95*len(gaps))-1] if gaps else None
    observed=elapsed-unknown
    occupancy=occupied/(8*observed) if observed else None
    full_hour=occupied/(8*3600)
    sample_acceptance='pass' if elapsed>=3600 and unknown==0 and full_hour>=.90 \
        and p95 is not None and p95<30 else 'inconclusive' if elapsed<3600 else 'fail'
    return dict(schema='parallel.gpu_occupancy.v2',window_seconds=elapsed,eligible_seconds=eligible,
        observed_gpu_task_seconds=occupied,unobserved_seconds=unknown,task_occupancy=occupancy,
        full_hour_occupancy=full_hour,full_hour_denominator=8*3600,
        kernel_utilization_mean=kernel_total/kernel_seconds if kernel_seconds else None,
        handoff_gap_p95_seconds=p95,handoffs=len(gaps),sampling_interval_seconds=10,
        sample_window_acceptance=sample_acceptance,
        acceptance='inconclusive',
        qualification_blockers=['No durable predeclared observation interval',
            'Original funded task/output and fit-start reconciliation evidence not verified'],
        measurement='diagnostic observer samples; occupancy uses 8×3600; these samples cannot qualify useful work')
