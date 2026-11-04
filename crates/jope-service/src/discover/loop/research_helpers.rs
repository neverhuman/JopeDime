use super::*;

#[cfg(feature = "research")]
pub(crate) fn parallel_evaluate_research(
    executor: &impl Executor,
    requests: &[Request],
    workers: usize,
    wall_ms: u64,
    mut persist: impl FnMut(usize, Response, u64) -> Result<()>,
) -> Result<()> {
    use std::sync::{atomic::AtomicBool, mpsc};
    let cursor = AtomicUsize::new(0);
    let halted = AtomicBool::new(false);
    let started = ElapsedSpan::start();
    let (sender, receiver) = mpsc::sync_channel(workers.max(1));
    std::thread::scope(|scope| {
        for _ in 0..workers.min(requests.len()) {
            let sender = sender.clone();
            let cursor = &cursor;
            let halted = &halted;
            let started = &started;
            scope.spawn(move || {
                while !halted.load(Ordering::Acquire) {
                    if started.millis() >= u128::from(wall_ms) {
                        break;
                    }
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(request) = requests.get(index) else {
                        break;
                    };
                    let started = Instant::now();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        research_response(request, executor.evaluate(request))
                    }))
                    .map_err(|_| anyhow::anyhow!("evaluator aborted; outcome remains unknown"))
                    .and_then(|result| result);
                    let wall = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                    if sender.send((index, result, wall)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(sender);
        let mut error = None;
        let mut received = 0;
        for (index, response, wall) in receiver {
            received += 1;
            if error.is_none() {
                let result = response.and_then(|response| persist(index, response, wall));
                if let Err(failure) = result {
                    halted.store(true, Ordering::Release);
                    error = Some(failure);
                }
            }
        }
        if error.is_none() && received != requests.len() {
            error = Some(anyhow::anyhow!(
                "capture wall budget exhausted; undispatched reservations remain unknown"
            ));
        }
        error.map_or(Ok(()), Err)
    })
}

#[cfg(feature = "research")]
pub(crate) fn research_response(request: &Request, response: Response) -> Result<Response> {
    ensure!(
        response.attempt == request.attempt
            && response.failure != Some(WorkerFailure::StaleAttempt)
            && response.valid()
            && (response.failure.is_some() || response.scores.len() == 3),
        "invalid or stale capture response; outcome remains unknown"
    );
    Ok(response)
}
