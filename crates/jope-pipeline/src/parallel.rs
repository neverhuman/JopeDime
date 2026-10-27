//! Bounded CPU parallelism for the search.
//!
//! A host fills its admitted cores with `islands x threads` workers. That product is refused,
//! before anything is written, when it exceeds the CPUs this process may run on
//! (`sched_getaffinity`), `std::thread::available_parallelism` (which also honours cgroup CPU
//! quotas) or an explicit `--cpu-ceiling` in millicores. Which thread finishes first never
//! changes a result: candidates are admitted in slot order, evaluated concurrently, and folded
//! back in slot order. So while the evaluation budget ends every phase, the worker count changes
//! no decision either. The seconds bound is checked before each fit starts, on its worker, so
//! when it binds, how many candidates start before it depends on timing and so on the worker
//! count; the receipt records such a phase as stopped by `seconds`.
//!
//! A run launched as a child process is stopped the same cooperative way: SIGTERM or SIGINT set
//! a flag the search checks before every fit, so in-flight fits finish, nothing new starts, and
//! the run returns [`Cancelled`] without writing a receipt.

use anyhow::{ensure, Context, Result};

/// A run stopped by a termination signal between evaluations. It wrote no receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "stopped by a termination signal between evaluations; the interrupted run wrote no \
             receipt",
        )
    }
}

impl std::error::Error for Cancelled {}

/// Whether `error` is, or wraps, [`Cancelled`].
pub fn is_cancelled(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| cause.is::<Cancelled>())
}

/// The exit status of a run a signal stopped: 128 + SIGTERM.
pub const CANCELLED_EXIT: u8 = 143;

/// A flag SIGTERM and SIGINT set. A second signal while the first is being honoured exits at once
/// with [`CANCELLED_EXIT`].
#[cfg(target_os = "linux")]
pub fn termination_flag() -> Result<std::sync::Arc<std::sync::atomic::AtomicBool>> {
    use signal_hook::consts::{SIGINT, SIGTERM};
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    for signal in [SIGTERM, SIGINT] {
        signal_hook::flag::register_conditional_shutdown(
            signal,
            i32::from(CANCELLED_EXIT),
            flag.clone(),
        )
        .context("register the second-signal shutdown")?;
        signal_hook::flag::register(signal, flag.clone()).context("register the stop flag")?;
    }
    Ok(flag)
}

/// Off Linux no signal is wired: the flag never trips.
#[cfg(not(target_os = "linux"))]
pub fn termination_flag() -> Result<std::sync::Arc<std::sync::atomic::AtomicBool>> {
    Ok(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
        false,
    )))
}

/// How often [`finish_flag`] looks for its stop file.
pub const STOP_FILE_POLL: std::time::Duration = std::time::Duration::from_millis(250);

/// A flag that asks a search to *finish*, as distinct from [`termination_flag`]'s cancel: nothing
/// new starts, in-flight fits complete, and the run then ranks and reports what it has measured,
/// exactly as when its seconds budget runs out. SIGUSR1 sets it (on Linux), and so does
/// `stop_file` coming into existence, which a watcher thread checks every [`STOP_FILE_POLL`].
/// Setting it twice, or after the search ended, changes nothing.
pub fn finish_flag(
    stop_file: Option<&std::path::Path>,
) -> Result<std::sync::Arc<std::sync::atomic::AtomicBool>> {
    use std::sync::atomic::Ordering;
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    #[cfg(target_os = "linux")]
    signal_hook::flag::register(signal_hook::consts::SIGUSR1, flag.clone())
        .context("register the finish signal")?;
    if let Some(path) = stop_file {
        let path = path.to_path_buf();
        let watched = flag.clone();
        std::thread::Builder::new()
            .name("stop-file".to_owned())
            .spawn(move || {
                while !watched.load(Ordering::SeqCst) {
                    if path.exists() {
                        watched.store(true, Ordering::SeqCst);
                        break;
                    }
                    std::thread::sleep(STOP_FILE_POLL);
                }
            })
            .context("start the stop-file watcher")?;
    }
    Ok(flag)
}

/// CPUs in this process's affinity mask.
#[cfg(target_os = "linux")]
pub fn affinity_cpus() -> Result<usize> {
    use nix::sched::{sched_getaffinity, CpuSet};
    use nix::unistd::Pid;
    let set = sched_getaffinity(Pid::from_raw(0)).context("sched_getaffinity")?;
    Ok((0..CpuSet::count())
        .filter(|&cpu| set.is_set(cpu).unwrap_or(false))
        .count())
}

/// CPUs this process may use (no affinity call off Linux).
#[cfg(not(target_os = "linux"))]
pub fn affinity_cpus() -> Result<usize> {
    Ok(std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
}

/// Admit `islands x threads` workers, or refuse and say which bound they exceed.
pub fn admit(threads: usize, islands: usize, cpu_ceiling_millicores: Option<u64>) -> Result<usize> {
    ensure!(threads >= 1, "--threads must be at least 1");
    ensure!(islands >= 1, "--islands must be at least 1");
    let workers = threads
        .checked_mul(islands)
        .context("--islands x --threads overflows")?;
    let affinity = affinity_cpus()?;
    ensure!(
        workers <= affinity,
        "--islands {islands} x --threads {threads} = {workers} worker(s) exceeds the {affinity} \
         CPU(s) sched_getaffinity allows this process"
    );
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    ensure!(
        workers <= available,
        "--islands {islands} x --threads {threads} = {workers} worker(s) exceeds \
         available_parallelism ({available})"
    );
    if let Some(ceiling) = cpu_ceiling_millicores {
        ensure!(
            (workers as u64).saturating_mul(1000) <= ceiling,
            "--islands {islands} x --threads {threads} = {workers} worker(s) needs {} millicores, \
             above --cpu-ceiling {ceiling}",
            (workers as u64).saturating_mul(1000)
        );
    }
    Ok(workers)
}

/// The stack every candidate fit runs on, whatever the worker count: the campaigns' candidate
/// stack (`binary42::CANDIDATE_STACK_BYTES`).
///
/// The calling thread's stack is not the run's to choose (a CLI main thread gets the ulimit,
/// usually 8 MiB; a test or embedding thread, and every default rayon worker, 2 MiB), and a fit
/// that overflows it aborts the whole process: a stack overflow is not a panic
/// `search::evaluate`'s `catch_unwind` can catch. Fitting everything on threads of this size is
/// what keeps the worker count from deciding whether a deep fit measures or kills the run.
pub const EVALUATION_STACK_BYTES: usize = 32 * 1024 * 1024;

/// The evaluation pool: `workers` threads with [`EVALUATION_STACK_BYTES`] each. One worker is a
/// pool too, so `--threads 1` fits on the same stack as `--threads 8`.
pub fn pool(workers: usize) -> Result<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers.max(1))
        .stack_size(EVALUATION_STACK_BYTES)
        .thread_name(|i| format!("jope-pipeline-eval-{i}"))
        .build()
        .context("build the evaluation thread pool")
}

/// Run `work` on one thread with the evaluation stack: for fits outside a search's pool (the
/// test-split refit, single-cut helpers).
pub fn on_evaluation_stack<T: Send>(work: impl FnOnce() -> T + Send) -> Result<T> {
    Ok(pool(1)?.install(work))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stop_file_sets_the_finish_flag_and_its_absence_does_not() {
        use std::sync::atomic::Ordering;
        let dir = tempfile::tempdir().unwrap();
        let stop = dir.path().join("stop");
        let flag = finish_flag(Some(&stop)).unwrap();
        std::thread::sleep(STOP_FILE_POLL * 3);
        assert!(!flag.load(Ordering::SeqCst), "no stop file, no finish");
        std::fs::write(&stop, b"").unwrap();
        let waited = std::time::Instant::now();
        while !flag.load(Ordering::SeqCst) {
            assert!(
                waited.elapsed() < STOP_FILE_POLL * 20,
                "the watcher never saw the stop file"
            );
            std::thread::sleep(STOP_FILE_POLL / 5);
        }
    }

    #[test]
    fn one_worker_is_admitted_and_oversubscription_is_refused() {
        assert_eq!(admit(1, 1, None).unwrap(), 1);
        let cpus = affinity_cpus().unwrap();
        let error = format!("{:#}", admit(cpus + 1, 1, None).unwrap_err());
        assert!(error.contains("sched_getaffinity"), "{error}");
        assert!(admit(0, 1, None).is_err());
        assert!(admit(1, 0, None).is_err());
        assert!(admit(usize::MAX, 2, None).is_err());
    }

    /// Uses at least `bytes` of stack: every frame keeps a live 32 KiB buffer, read again after
    /// the recursive call so it cannot be dropped or the call turned into a loop.
    fn deep(bytes: usize) -> u64 {
        let mut frame = [0u8; 32 * 1024];
        std::hint::black_box(&mut frame);
        let rest = if bytes > frame.len() {
            deep(bytes - frame.len())
        } else {
            0
        };
        std::hint::black_box(&frame);
        rest + u64::from(frame[0]) + 1
    }

    /// Deeper than rayon's default worker stack (2 MiB) and a CLI main thread's usual 8 MiB.
    const DEEP: usize = 10 << 20;

    /// A fit that needs more stack than a default thread must not measure at one worker and abort
    /// the process at two: a stack overflow is not a panic `catch_unwind` can catch. Every worker
    /// count, and a fit outside a search, runs on the same evaluation stack.
    #[test]
    fn a_deep_fit_survives_on_more_than_one_worker() {
        use rayon::prelude::*;
        let frames = (DEEP / (32 * 1024)) as u64;
        for workers in [1, 2] {
            let pool = pool(workers).unwrap();
            assert_eq!(pool.current_num_threads(), workers);
            let depths: Vec<u64> = pool.install(|| {
                (0..workers * 2)
                    .into_par_iter()
                    .map(|_| deep(DEEP))
                    .collect()
            });
            assert_eq!(depths, vec![frames; workers * 2], "{workers} worker(s)");
        }
        assert_eq!(on_evaluation_stack(|| deep(DEEP)).unwrap(), frames);
    }

    #[test]
    fn the_cpu_ceiling_bounds_islands_times_threads() {
        assert_eq!(admit(1, 1, Some(1000)).unwrap(), 1);
        let error = format!("{:#}", admit(1, 1, Some(999)).unwrap_err());
        assert!(error.contains("--cpu-ceiling 999"), "{error}");
        if affinity_cpus().unwrap() >= 2
            && std::thread::available_parallelism().map_or(1, std::num::NonZero::get) >= 2
        {
            assert_eq!(admit(2, 1, Some(2000)).unwrap(), 2);
            assert!(admit(2, 1, Some(1999)).is_err());
            assert!(admit(1, 2, Some(1500)).is_err());
        }
    }
}
