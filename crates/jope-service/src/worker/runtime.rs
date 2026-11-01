//! Child protocol loop; the evaluator borrows one immutable matrix mapping.

use super::{
    frame::{decode_frame, encode_frame, FrameError},
    process, Request, Response, Scores, WorkerFailure,
};
use crate::dataset::jmat::{self, Matrix};
use memmap2::Mmap;
use std::{
    fs::File,
    io::{self, IsTerminal, Write},
    panic::{catch_unwind, AssertUnwindSafe},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::Instant,
};

/// Run the hidden worker on stdin/stdout. This function owns the process's stdin lifetime:
/// EOF terminates the process even if a candidate is stuck inside an evaluator.
///
/// Matrix files must be immutable content-addressed store artifacts for the mapping lifetime.
/// The callback must not retain the borrowed bytes, write stdout, or perform store effects.
pub fn serve(
    protocol: u32,
    mut evaluate: impl FnMut(&Request, &Matrix<Mmap>) -> Result<Scores, WorkerFailure>,
) -> io::Result<()> {
    if protocol != 1 || io::stdin().is_terminal() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "__worker requires --protocol 1 and piped stdin",
        ));
    }
    process::protect_parent()?;
    let (tx, rx) = mpsc::sync_channel(1);
    let busy = Arc::new(AtomicBool::new(false));
    let input_busy = Arc::clone(&busy);
    thread::Builder::new()
        .name("worker-stdin".into())
        .spawn(move || {
            let mut input = io::stdin().lock();
            loop {
                let value = match decode_frame(&mut input) {
                    Ok(value) => value,
                    Err(FrameError::TruncatedHeader { received: 0 }) => std::process::exit(0),
                    Err(_) => std::process::exit(2),
                };
                let request: Request = match serde_json::from_value(value) {
                    Ok(request) => request,
                    Err(_) => std::process::exit(2),
                };
                if input_busy.swap(true, Ordering::AcqRel) || tx.try_send(request).is_err() {
                    std::process::exit(2);
                }
            }
        })?;
    let mut matrix: Option<(PathBuf, Matrix<Mmap>)> = None;
    let mut baseline = 0;
    let mut evaluations = 0_u32;
    let mut output = io::stdout().lock();
    while let Ok(request) = rx.recv() {
        let started = Instant::now();
        let result = (|| {
            if request.limits.wall_ms == 0
                || request.limits.rss_bytes == 0
                || request.attempt.attempt == 0
            {
                return Err(WorkerFailure::InvalidRequest);
            }
            if matrix
                .as_ref()
                .is_some_and(|(path, _)| path != &request.matrix_path)
            {
                return Err(WorkerFailure::InvalidRequest);
            }
            if matrix.is_none() {
                let file =
                    File::open(&request.matrix_path).map_err(|_| WorkerFailure::InvalidRequest)?;
                #[allow(unsafe_code)]
                let map = {
                    // SAFETY: the immutable store artifact stays unmodified and untruncated for this read-only mapping.
                    unsafe { jmat::map_read_only(&file) }
                }
                .map_err(|_| WorkerFailure::InvalidRequest)?;
                matrix = Some((request.matrix_path.clone(), map));
                baseline = crate::sampler::sample(std::process::id())
                    .map_err(|_| WorkerFailure::Unavailable)?
                    .1;
            }
            match catch_unwind(AssertUnwindSafe(|| {
                evaluate(&request, &matrix.as_ref().unwrap().1)
            })) {
                Ok(result) => result,
                Err(_) => Err(WorkerFailure::Panicked),
            }
        })();
        let wall_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        let response = match result {
            Ok(scores) => {
                let value = Response {
                    attempt: request.attempt.clone(),
                    scores: scores.values,
                    fold_scores: scores.fold_scores,
                    predictions_digest: scores.predictions_digest,
                    wall_ms,
                    failure: None,
                };
                if value.valid() {
                    value
                } else {
                    Response::failed(request.attempt, WorkerFailure::NonFinite, wall_ms)
                }
            }
            Err(failure) => Response::failed(request.attempt, failure, wall_ms),
        };
        let bytes = encode_frame(&serde_json::to_value(&response).map_err(io::Error::other)?)
            .map_err(io::Error::other)?;
        evaluations += 1;
        let rss = crate::sampler::sample(std::process::id())?.1;
        let retire = evaluations >= 500
            || (baseline > 0 && rss > baseline.saturating_mul(2))
            || response.failure.is_some();
        // Clear before publishing the response: the next request is then safe immediately.
        busy.store(false, Ordering::Release);
        output.write_all(&bytes)?;
        output.flush()?;
        if retire {
            // Distinguish planned retirement from any unexpected exit, including exit(0).
            // The complete response is already in the pipe; this process holds no store effects.
            std::process::exit(process::RETIRE_EXIT_CODE);
        }
    }
    Ok(())
}
