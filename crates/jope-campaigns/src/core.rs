use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::fs;

pub(crate) fn physical_cores() -> Result<Vec<usize>> {
    physical_cores_from(
        std::path::Path::new("/sys/devices/system/cpu"),
        &allowed_cpus()?,
    )
}

fn physical_cores_from(cpu_root: &std::path::Path, allowed: &[usize]) -> Result<Vec<usize>> {
    let mut cores = BTreeMap::<(i32, i32), usize>::new();
    for entry in fs::read_dir(cpu_root)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(number) = name.to_str().and_then(|text| text.strip_prefix("cpu")) else {
            continue;
        };
        let Ok(cpu) = number.parse::<usize>() else {
            continue;
        };
        if !allowed.contains(&cpu) {
            continue;
        }
        let topology = entry.path().join("topology");
        let package = read_i32(&topology.join("physical_package_id"))?;
        let core = read_i32(&topology.join("core_id"))?;
        cores.entry((package, core)).or_insert(cpu);
    }
    let result = cores.into_values().collect::<Vec<_>>();
    if result.is_empty() {
        bail!("Linux topology exposed no allowed physical cores");
    }
    Ok(result)
}

fn allowed_cpus() -> Result<Vec<usize>> {
    // SAFETY: sched_getaffinity writes to this live, correctly sized set for the calling thread;
    // CPU_ISSET below only reads indices within CPU_SETSIZE after the syscall succeeds.
    unsafe {
        // SAFETY: cpu_set_t contains only integer mask words, for which zeroed is valid.
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Err(std::io::Error::last_os_error()).context("read process CPU affinity");
        }
        Ok((0..libc::CPU_SETSIZE as usize)
            .filter(|cpu| libc::CPU_ISSET(*cpu, &set))
            .collect())
    }
}

fn read_i32(path: &std::path::Path) -> Result<i32> {
    fs::read_to_string(path)?
        .trim()
        .parse()
        .with_context(|| format!("parse Linux topology value {}", path.display()))
}

pub(crate) fn require_core_plan(count: usize) -> Result<Vec<usize>> {
    core_plan(physical_cores()?, count)
}

fn core_plan(cores: Vec<usize>, count: usize) -> Result<Vec<usize>> {
    if cores.len() < count {
        bail!(
            "campaign requires {count} distinct physical cores, Linux affinity exposes {}",
            cores.len()
        );
    }
    Ok(cores.into_iter().take(count).collect())
}

pub(crate) fn pin_current_thread(cpu: usize) -> Result<()> {
    if cpu >= libc::CPU_SETSIZE as usize {
        bail!(
            "CPU index {cpu} exceeds affinity mask capacity {}",
            libc::CPU_SETSIZE
        );
    }
    // SAFETY: the checked index fits the set, and pthread_setaffinity_np borrows its sized
    // storage only for this call. pthread_self identifies the live calling thread.
    unsafe {
        // SAFETY: cpu_set_t contains only integer mask words, for which zeroed is valid.
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        let code = libc::pthread_setaffinity_np(
            libc::pthread_self(),
            std::mem::size_of::<libc::cpu_set_t>(),
            &set,
        );
        if code != 0 {
            bail!(
                "pin current thread to CPU {cpu}: {}",
                std::io::Error::from_raw_os_error(code)
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_range_cpu_indices_return_errors_without_affinity_changes() {
        let before = allowed_cpus().unwrap();
        for cpu in [libc::CPU_SETSIZE as usize, usize::MAX] {
            let error = pin_current_thread(cpu).unwrap_err();
            assert!(error.to_string().contains("exceeds affinity mask capacity"));
            assert_eq!(allowed_cpus().unwrap(), before);
        }
    }

    #[test]
    fn pinning_changes_only_the_calling_threads_affinity() {
        let before = allowed_cpus().unwrap();
        let cpu = before[0];
        std::thread::spawn(move || {
            pin_current_thread(cpu).unwrap();
            assert_eq!(allowed_cpus().unwrap(), vec![cpu]);
        })
        .join()
        .unwrap();
        assert_eq!(allowed_cpus().unwrap(), before);
    }

    #[test]
    fn core_plan_is_distinct_and_large_enough_on_campaign_host() {
        let available = physical_cores().unwrap();
        if available.len() < 60 {
            assert_eq!(
                require_core_plan(60).unwrap_err().to_string(),
                format!(
                    "campaign requires 60 distinct physical cores, Linux affinity exposes {}",
                    available.len()
                )
            );
        } else {
            assert_distinct_plan(&require_core_plan(60).unwrap(), 60);
        }
        let requested = available.len().min(60);
        let plan = require_core_plan(requested).unwrap();
        assert_distinct_plan(&plan, requested);
        assert!(plan.iter().all(|cpu| available.contains(cpu)));

        // A fixed 64-core, two-thread topology exercises the real enumeration and
        // planning code even when this process has fewer than 60 physical cores.
        let topology = tempfile::tempdir().unwrap();
        for cpu in 0..128 {
            let path = topology.path().join(format!("cpu{cpu}/topology"));
            fs::create_dir_all(&path).unwrap();
            fs::write(path.join("physical_package_id"), "0").unwrap();
            fs::write(path.join("core_id"), (cpu % 64).to_string()).unwrap();
        }
        let reserved = (0..51).chain(64..115).collect::<Vec<_>>();
        let cores = physical_cores_from(topology.path(), &reserved).unwrap();
        assert_eq!(cores.len(), 51);
        assert_eq!(
            core_plan(cores, 60).unwrap_err().to_string(),
            "campaign requires 60 distinct physical cores, Linux affinity exposes 51"
        );
        let all = (0..128).collect::<Vec<_>>();
        let cores = physical_cores_from(topology.path(), &all).unwrap();
        assert_eq!(cores.len(), 64);
        let plan = core_plan(cores, 60).unwrap();
        assert_distinct_plan(&plan, 60);
        let physical_ids = plan
            .iter()
            .map(|cpu| cpu % 64)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(physical_ids.len(), 60);
    }

    fn assert_distinct_plan(plan: &[usize], expected: usize) {
        let unique = plan
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(plan.len(), expected);
        assert_eq!(unique.len(), expected);
    }
}
