use super::*;

pub(super) fn is_golden46_run(run_id: &str) -> bool {
    run_id == GOLDEN46_RUN_PREFIX
        || run_id == format!("{GOLDEN46_RUN_PREFIX}{GOLDEN46_TRANSFER_SUFFIX}")
}

pub(super) fn root(run_id: &str) -> PathBuf {
    if run_id == format!("{GOLDEN46_RUN_PREFIX}{GOLDEN46_TRANSFER_SUFFIX}") {
        return campaign_state_root()
            .join("runs")
            .join(GOLDEN46_RUN_PREFIX)
            .join("transfer-arms")
            .join("target-family-informed");
    }
    campaign_state_root().join("runs").join(run_id)
}

pub(super) fn dataset_root(run_id: &str, dataset_hash: &str) -> PathBuf {
    root(run_id).join("datasets").join(dataset_hash)
}

pub(super) fn checkpoint_path(run_id: &str, dataset_hash: &str, generation: usize) -> PathBuf {
    dataset_root(run_id, dataset_hash)
        .join("checkpoints")
        .join(format!("generation-{generation:03}.v1.json"))
}

pub(super) fn confirmation_path(run_id: &str, dataset_hash: &str) -> PathBuf {
    dataset_root(run_id, dataset_hash).join("confirmation.v1.json")
}

pub(super) fn result_path(run_id: &str, dataset_hash: &str) -> PathBuf {
    dataset_root(run_id, dataset_hash).join("final-result.v1.json")
}

pub(super) fn atomic_immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        if fs::read(path)? == bytes {
            return Ok(());
        }
        bail!("immutable artifact collision at {}", path.display());
    }
    let parent = path.parent().context("artifact path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact"),
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path)?;
    OpenOptions::new().read(true).open(parent)?.sync_all()?;
    Ok(())
}

pub(super) fn ssh(node: &str, args: &[&str], label: &str) -> Result<Vec<u8>> {
    let output = Command::new("ssh")
        .arg(node)
        .arg("--")
        .args(args)
        .output()
        .with_context(|| format!("launch {label} on {node}"))?;
    if !output.status.success() {
        bail!(
            "{label} failed on {node}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
}

pub(super) fn parse_kib_field(text: &str, name: &str) -> Result<u64> {
    let value = text
        .lines()
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            (fields.next()? == name)
                .then(|| fields.next()?.parse::<u64>().ok())
                .flatten()
        })
        .with_context(|| format!("/proc/meminfo has no {name}"))?;
    value
        .checked_mul(1024)
        .context("memory byte count overflow")
}

pub(super) fn inspect_node(node: &str) -> Result<NodeResources> {
    let local_hostname = String::from_utf8(
        Command::new("hostname")
            .arg("-s")
            .output()
            .context("read controller hostname")?
            .stdout,
    )?
    .trim()
    .to_string();
    let controller_pid = std::process::id().to_string();
    let online_cores = String::from_utf8(ssh(
        node,
        &["getconf", "_NPROCESSORS_ONLN"],
        "read online cores",
    )?)?
    .trim()
    .parse::<usize>()?;
    let memory = String::from_utf8(ssh(node, &["cat", "/proc/meminfo"], "read memory")?)?;
    let available_ram_bytes = parse_kib_field(&memory, "MemAvailable:")?;
    let swap_total = parse_kib_field(&memory, "SwapTotal:")?;
    let swap_free = parse_kib_field(&memory, "SwapFree:")?;
    let swap_used_bytes = swap_total.saturating_sub(swap_free);
    let state = jope_paths::to_string(&paths::jope_state());
    let disk = String::from_utf8(ssh(node, &["df", "-Pk", &state], "read free disk")?)?;
    let free_kib = disk
        .lines()
        .last()
        .and_then(|line| line.split_whitespace().nth(3))
        .context("df returned no available-block field")?
        .parse::<u64>()?;
    let free_disk_bytes = free_kib
        .checked_mul(1024)
        .context("disk byte count overflow")?;
    let processes = String::from_utf8(ssh(
        node,
        &["ps", "-eo", "pid=,comm=,args="],
        "read process table",
    )?)?;
    let competing_processes = processes
        .lines()
        .filter(|line| is_competing_process(line, node, &local_hostname, &controller_pid))
        .map(str::trim)
        .map(str::to_string)
        .collect::<Vec<_>>();
    Ok(NodeResources {
        node: node.to_string(),
        online_cores,
        available_ram_bytes,
        free_disk_bytes,
        swap_used_bytes,
        competing_processes,
    })
}

pub(super) fn scq_job_is_active(job: &Value) -> bool {
    !matches!(
        job.get("state").and_then(Value::as_str),
        Some("completed" | "failed" | "cancelled")
    )
}

pub(super) fn is_competing_process(
    line: &str,
    node: &str,
    local_hostname: &str,
    controller_pid: &str,
) -> bool {
    let lower = line.to_ascii_lowercase();
    let comm = line.split_whitespace().nth(1).unwrap_or_default();
    let is_jope_executable = comm.starts_with("jope-dime-regre");
    let is_scq_executable = comm == "scq";
    let is_this_controller =
        node == local_hostname && line.split_whitespace().next() == Some(controller_pid);
    let is_campaign_controller = lower.contains("run-golden42-aggressive")
        || lower.contains("run-golden46")
        || lower.contains("run-tabarena-lite-islands");
    let is_campaign_worker = lower.contains("test-adaptive-worker --request")
        || lower.contains("golden42-aggressive-batch-worker")
        || lower.contains("test-adaptive-island-batch-worker");
    let is_campaign_scq_job = is_scq_executable && lower.contains("jope-g42-");
    let is_scq_control_plane = lower.contains("scqd")
        || lower.contains("scq controller serve")
        || lower.contains("scq agent status");
    (is_jope_executable || is_scq_executable)
        && !is_scq_control_plane
        && !is_campaign_controller
        && !is_campaign_worker
        && !is_campaign_scq_job
        && !is_this_controller
}

pub(super) fn preflight() -> Result<PreflightReceipt> {
    let scq = paths::scq_binary();
    let output = Command::new(scq)
        .args(["--json", "ls"])
        .output()
        .context("launch SCQ queue inspection")?;
    if !output.status.success() {
        bail!("SCQ queue inspection failed");
    }
    let queue: Value = serde_json::from_slice(&output.stdout)?;
    let active_scq_jobs = queue
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .context("SCQ queue response has no jobs array")?
        .iter()
        .filter(|job| scq_job_is_active(job))
        .filter(|job| {
            !job.get("id")
                .and_then(Value::as_str)
                .is_some_and(|id| id.starts_with("jope-g42-"))
        })
        .filter(|job| {
            job.get("state").and_then(Value::as_str) != Some("queued")
                || job.get("node").and_then(Value::as_str).is_some()
        })
        .map(|job| {
            job.get("id")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string()
        })
        .collect::<Vec<_>>();
    if !active_scq_jobs.is_empty() {
        bail!("SCQ has competing active jobs");
    }
    let nodes = scq_batch::NODES
        .iter()
        .map(|node| inspect_node(node))
        .collect::<Result<Vec<_>>>()?;
    for node in &nodes {
        if node.online_cores < scq_batch::AGGRESSIVE_CPUS {
            bail!(
                "{} has fewer than {} online cores",
                node.node,
                scq_batch::AGGRESSIVE_CPUS
            );
        }
        if node.available_ram_bytes < MIN_AVAILABLE_RAM_BYTES {
            bail!("{} has less than 48 GiB available RAM", node.node);
        }
        if node.free_disk_bytes < MIN_FREE_DISK_BYTES {
            bail!("{} has less than 100 GiB free disk", node.node);
        }
        if node.swap_used_bytes >= MAX_SWAP_USED_BYTES {
            bail!(
                "{} uses {} bytes of swap; the aggressive campaign requires less than 1 GiB",
                node.node,
                node.swap_used_bytes
            );
        }
        if !node.competing_processes.is_empty() {
            bail!("{} has a competing Cargo/Rust/Jope/SCQ job", node.node);
        }
    }
    Ok(PreflightReceipt {
        schema: PREFLIGHT_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        required_cores: scq_batch::AGGRESSIVE_CPUS,
        required_available_ram_bytes: MIN_AVAILABLE_RAM_BYTES,
        required_free_disk_bytes: MIN_FREE_DISK_BYTES,
        maximum_swap_used_bytes_exclusive: MAX_SWAP_USED_BYTES,
        active_scq_jobs,
        nodes,
    })
}
