use super::*;

pub(super) fn terminate_workers(workers: &mut [WorkerProcess]) {
    for worker in workers.iter_mut() {
        if worker.child.try_wait().ok().flatten().is_some() {
            continue;
        }
        let pid = worker.child.id() as i32;
        // SAFETY: kill reads no memory; try_wait just saw this child unreaped, so its PID
        // (never 0 or 1) is not reused and -pid can only name the group the child leads.
        unsafe {
            libc::kill(-pid, libc::SIGTERM);
        }
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        let all_done = workers
            .iter_mut()
            .all(|worker| worker.child.try_wait().ok().flatten().is_some());
        if all_done {
            return;
        }
        thread::sleep(Duration::from_millis(20));
    }
    for worker in workers.iter_mut() {
        if worker.child.try_wait().ok().flatten().is_none() {
            let pid = worker.child.id() as i32;
            // SAFETY: kill reads no memory; try_wait just saw this child unreaped, so its PID
            // (never 0 or 1) is not reused and -pid can only name the group the child leads.
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
            let _ = worker.child.wait();
        }
    }
}

pub(super) fn choose_cpus(required: usize) -> Result<Vec<usize>> {
    let mut cores = BTreeMap::<(String, String), Vec<usize>>::new();
    for entry in fs::read_dir("/sys/devices/system/cpu")? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(number) = name
            .strip_prefix("cpu")
            .and_then(|value| value.parse::<usize>().ok())
        else {
            continue;
        };
        if fs::read_to_string(entry.path().join("online"))
            .ok()
            .is_some_and(|value| value.trim() == "0")
        {
            continue;
        }
        let topology = entry.path().join("topology");
        let package = fs::read_to_string(topology.join("physical_package_id"))?;
        let core = fs::read_to_string(topology.join("core_id"))?;
        cores
            .entry((package.trim().to_string(), core.trim().to_string()))
            .or_default()
            .push(number);
    }
    for cpus in cores.values_mut() {
        cpus.sort_unstable();
    }
    let mut selected = cores
        .values()
        .filter_map(|cpus| cpus.first().copied())
        .collect::<Vec<_>>();
    selected.sort_unstable();
    let mut siblings = cores
        .values()
        .flat_map(|cpus| cpus.iter().skip(1).copied())
        .collect::<Vec<_>>();
    siblings.sort_unstable();
    selected.extend(siblings);
    selected.truncate(required);
    if selected.len() != required {
        bail!(
            "only {} usable logical CPUs found; need {required}",
            selected.len()
        );
    }
    Ok(selected)
}

pub(super) fn write_worker_registry(root: &Path, workers: &[WorkerProcess]) -> Result<()> {
    let records = workers
        .iter()
        .map(|worker| {
            serde_json::json!({
                "worker_id": worker.id,
                "pid": worker.child.id(),
                "cpu": worker.cpu,
            })
        })
        .collect::<Vec<_>>();
    atomic_replace(
        &root.join("workers.v1.json"),
        &json_bytes(&serde_json::json!({
            "schema": "jope-dime.golden42-v3-workers.v1",
            "controller_pid": std::process::id(),
            "recorded_at_unix_ms": unix_ms(),
            "workers": records,
        }))?,
    )
}
