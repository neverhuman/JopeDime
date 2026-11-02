use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn append_telemetry(
    root: &Path,
    run_id: &str,
    generation: usize,
    completed: usize,
    pending: usize,
    workers: &mut [WorkerProcess],
    cache_hits: usize,
    invalids: usize,
    diversity: usize,
    receipts: &BTreeMap<usize, CandidateReceipt>,
    generation_started: Instant,
) -> Result<()> {
    let health = resource_health()?;
    let mut workers_alive = 0usize;
    let mut workers_affinity_ok = 0usize;
    let mut worker_states = Vec::with_capacity(workers.len());
    for worker in workers.iter_mut() {
        let alive = worker.child.try_wait()?.is_none();
        if alive {
            workers_alive += 1;
        }
        let state = worker_telemetry(worker, alive)?;
        if state.allowed_cpu_list == worker.cpu.to_string() {
            workers_affinity_ok += 1;
        }
        worker_states.push(state);
    }
    let mut latencies = receipts
        .values()
        .map(|receipt| {
            receipt
                .fit_ns
                .saturating_add(receipt.codec_ns)
                .saturating_add(receipt.cpp_load_ns)
                .saturating_add(receipt.cpp_predict_ns)
        })
        .collect::<Vec<_>>();
    latencies.sort_unstable();
    let candidate_latency_p50_ms = latencies
        .get(latencies.len() / 2)
        .map_or(0.0, |value| *value as f64 / 1_000_000.0);
    let minutes = generation_started.elapsed().as_secs_f64() / 60.0;
    let throughput_candidates_per_minute = if minutes > 0.0 {
        completed.saturating_sub(cache_hits) as f64 / minutes
    } else {
        0.0
    };
    let operator_success = workers_alive == workers.len()
        && workers_affinity_ok == workers.len()
        && health.ram >= MIN_RAM
        && health.swap < MAX_SWAP
        && health.disk >= MIN_DISK;
    let record = Telemetry {
        schema: "jope-dime.golden42-v3-telemetry.v1",
        unix_ms: unix_ms(),
        generation,
        completed,
        pending,
        workers_alive,
        workers_affinity_ok,
        worker_states,
        cpu_load_1m: health.load,
        available_ram_bytes: health.ram,
        swap_used_bytes: health.swap,
        free_disk_bytes: health.disk,
        cache_hits,
        invalids,
        diversity,
        candidate_latency_p50_ms,
        throughput_candidates_per_minute,
        operator_success,
        run_id,
    };
    let path = root.join("telemetry.v1.jsonl");
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    // Serialize record and newline before appending; write_all may retry short writes.
    // An I/O failure can still leave an incomplete record.
    let mut line = serde_json::to_vec(&record)?;
    line.push(b'\n');
    file.write_all(&line)?;
    file.flush()?;
    Ok(())
}

pub(super) fn worker_telemetry(worker: &WorkerProcess, alive: bool) -> Result<WorkerTelemetry> {
    let pid = worker.child.id();
    let status = fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
    let status_value = |name: &str| {
        status
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_once(':'))
            .map_or_else(String::new, |(_, value)| value.trim().to_string())
    };
    let allowed_cpu_list = status_value("Cpus_allowed_list:");
    let state = status_value("State:");
    let rss_bytes = status_value("VmRSS:")
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
        .saturating_mul(1024);
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    let cpu_time_ticks = stat
        .rfind(") ")
        .map(|at| &stat[at + 2..])
        .map(|tail| tail.split_whitespace().collect::<Vec<_>>())
        .and_then(|fields| {
            let user = fields.get(11)?.parse::<u64>().ok()?;
            let system = fields.get(12)?.parse::<u64>().ok()?;
            Some(user.saturating_add(system))
        })
        .unwrap_or(0);
    Ok(WorkerTelemetry {
        worker_id: worker.id,
        pid,
        assigned_cpu: worker.cpu,
        allowed_cpu_list,
        state,
        cpu_time_ticks,
        rss_bytes,
        alive,
    })
}

pub(super) fn resource_health() -> Result<Health> {
    let memory = fs::read_to_string("/proc/meminfo")?;
    let value = |name: &str| -> Result<u64> {
        let kib = memory
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .context("missing /proc/meminfo field")?
            .parse::<u64>()?;
        Ok(kib.saturating_mul(1024))
    };
    let swap = value("SwapTotal:")?.saturating_sub(value("SwapFree:")?);
    let disk_text = command_text("df", &["-Pk", &workspace_root().display().to_string()])?;
    let disk = disk_text
        .lines()
        .nth(1)
        .and_then(|line| line.split_whitespace().nth(3))
        .context("parse df free space")?
        .parse::<u64>()?
        .saturating_mul(1024);
    let load = fs::read_to_string("/proc/loadavg")?
        .split_whitespace()
        .next()
        .context("parse load average")?
        .parse::<f64>()?;
    Ok(Health {
        ram: value("MemAvailable:")?,
        swap,
        disk,
        load,
    })
}

pub(super) fn wait_for_resources(
    root: &Path,
    run_id: &str,
    generation: usize,
    completed: usize,
    pending: usize,
    cache_hits: usize,
    invalids: usize,
) -> Result<()> {
    loop {
        let health = resource_health()?;
        if health.ram >= MIN_RAM && health.swap < MAX_SWAP && health.disk >= MIN_DISK {
            return Ok(());
        }
        let record = serde_json::json!({
            "schema": "jope-dime.golden42-v3-resource-pause.v1",
            "run_id": run_id,
            "generation": generation,
            "completed": completed,
            "pending": pending,
            "cache_hits": cache_hits,
            "invalids": invalids,
            "available_ram_bytes": health.ram,
            "swap_used_bytes": health.swap,
            "free_disk_bytes": health.disk,
            "unix_ms": unix_ms(),
        });
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(root.join("resource-pauses.v1.jsonl"))?;
        // Serialize record and newline before appending; write_all may retry short writes.
        // An I/O failure can still leave an incomplete record.
        let mut line = serde_json::to_vec(&record)?;
        line.push(b'\n');
        file.write_all(&line)?;
        thread::sleep(Duration::from_secs(5));
    }
}
