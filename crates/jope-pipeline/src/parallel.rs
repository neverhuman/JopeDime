//! Bounded CPU parallelism for the search.
//!
//! A host fills its admitted cores with `islands x threads` workers. That product is refused,
//! before anything is written, when it exceeds the CPUs this process may run on
//! (`sched_getaffinity`), `std::thread::available_parallelism` (which also honours cgroup CPU
//! quotas) or an explicit `--cpu-ceiling` in millicores. Parallelism never changes a result:
//! candidates are admitted in slot order, evaluated concurrently, and folded back in slot order,
//! so selection is independent of which thread finished first.

use anyhow::{ensure, Context, Result};

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

/// A dedicated pool for `workers > 1`; `None` evaluates on the calling thread exactly as the
/// sequential search always has.
pub fn pool(workers: usize) -> Result<Option<rayon::ThreadPool>> {
    if workers <= 1 {
        return Ok(None);
    }
    Ok(Some(
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|i| format!("jope-pipeline-eval-{i}"))
            .build()
            .context("build the evaluation thread pool")?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

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
