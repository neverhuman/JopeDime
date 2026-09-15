//! Read-only native operational observations. This module is not an admission authority.
use anyhow::{ensure, Context, Result};
use jain_smartcluster::{
    Client, JobRecord, JobState, ListFilter, ListRequest, ProjectId, TenantId, Workload,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
pub mod science;

#[derive(Clone)]
pub struct Config {
    pub socket: PathBuf,
    pub cgroup: PathBuf,
    pub output: PathBuf,
    pub tenant: TenantId,
    pub project: ProjectId,
    pub scope: String,
}
#[derive(Serialize)]
pub struct Snapshot {
    schema: &'static str,
    observed_at_unix_ms: u64,
    scope: String,
    resources: Vec<Value>,
    queues: Vec<Value>,
    continuations: Vec<Value>,
    corpus: Option<Value>,
    checkpoints: Vec<Value>,
    comparisons: Vec<Value>,
    winners: Vec<Value>,
    notes: Vec<String>,
    source_errors: Vec<String>,
    scientific_sources: Vec<Value>,
    scientific_verified_at_unix_ms: Option<u64>,
}
fn text(s: impl AsRef<str>) -> String {
    s.as_ref()
        .chars()
        .filter(|c| !c.is_control())
        .take(512)
        .collect()
}
fn now() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}
fn physical(path: &Path) -> Result<()> {
    ensure!(
        path.is_absolute() && fs::canonicalize(path)? == path,
        "physical absolute path required: {}",
        path.display()
    );
    Ok(())
}
fn read_small(path: &Path) -> Result<String> {
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    ensure!(file.metadata()?.is_file(), "regular source required");
    let mut bytes = Vec::new();
    file.take(65537).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 65536, "source exceeds 64 KiB");
    Ok(String::from_utf8(bytes)?)
}
fn cpu_count(value: &str) -> Result<usize> {
    let mut cpus = BTreeSet::new();
    for range in value.trim().split(',') {
        let (a, b) = range.split_once('-').unwrap_or((range, range));
        let a: u32 = a.parse()?;
        let b: u32 = b.parse()?;
        ensure!(a <= b && b < 65536, "invalid effective CPU range");
        cpus.extend(a..=b);
    }
    ensure!(!cpus.is_empty(), "empty effective CPU set");
    Ok(cpus.len())
}
fn usage(cgroup: &Path) -> Result<u64> {
    let stat = read_small(&cgroup.join("cpu.stat"))?;
    stat.lines()
        .find_map(|line| line.strip_prefix("usage_usec "))
        .context("cpu usage missing")?
        .parse()
        .map_err(Into::into)
}
fn cpu_percent(before: u64, after: u64, elapsed: Duration, cpus: usize) -> Result<f64> {
    ensure!(
        after >= before && !elapsed.is_zero() && cpus > 0,
        "CPU sample reset or invalid interval"
    );
    let percent =
        (after - before) as f64 / elapsed.as_secs_f64() / 1_000_000.0 / cpus as f64 * 100.0;
    ensure!(
        percent.is_finite() && percent <= 100.5,
        "CPU sample exceeds allocation basis"
    );
    Ok(percent.min(100.0))
}
async fn kernel(cgroup: &Path) -> Result<Value> {
    physical(cgroup)?;
    ensure!(
        cgroup.starts_with("/sys/fs/cgroup"),
        "kernel cgroup path required"
    );
    let cpuset = read_small(&cgroup.join("cpuset.cpus.effective"))?;
    let cpus = cpu_count(&cpuset)?;
    let before = usage(cgroup)?;
    let start = Instant::now();
    tokio::time::sleep(Duration::from_millis(500)).await;
    let after = usage(cgroup)?;
    ensure!(
        read_small(&cgroup.join("cpuset.cpus.effective"))? == cpuset,
        "CPU allocation changed during sample"
    );
    let cpu = cpu_percent(before, after, start.elapsed(), cpus)?;
    let memory: u64 = read_small(&cgroup.join("memory.current"))?.trim().parse()?;
    Ok(
        json!({"name":text(format!("Native cgroup (100% = {cpus} effective logical CPUs)")),"cpu_percent":cpu,"memory_bytes":memory}),
    )
}
fn gpu_rows(raw: &str) -> Result<Vec<Value>> {
    let mut rows = Vec::new();
    for line in raw.lines() {
        let fields: Vec<_> = line.split(',').map(str::trim).collect();
        ensure!(
            fields.len() == 3 && rows.len() < 16,
            "invalid bounded NVML sample"
        );
        let percent: f64 = fields[1].parse()?;
        let memory: u64 = fields[2].parse()?;
        ensure!((0.0..=100.0).contains(&percent), "invalid GPU utilization");
        rows.push(json!({"name":text(format!("GPU {} (NVML device-wide)",fields[0])),"gpu_percent":percent,"gpu_memory_bytes":memory.checked_mul(1048576).context("GPU memory overflow")?}));
    }
    ensure!(!rows.is_empty(), "NVML returned no GPUs");
    Ok(rows)
}
/// Fixed NVIDIA utility uses NVML. No shell, workload launch, or CUDA initialization.
async fn gpu() -> Result<Vec<Value>> {
    let mut child = Command::new("/usr/bin/nvidia-smi")
        .args([
            "--query-gpu=uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .env_clear()
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() > Duration::from_secs(2) {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("NVML utility exceeded two-second deadline");
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    ensure!(status.success(), "NVML utility failed");
    let mut bytes = Vec::new();
    child
        .stdout
        .take()
        .context("NVML stdout missing")?
        .take(65537)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 65536, "NVML output exceeds bound");
    gpu_rows(std::str::from_utf8(&bytes)?)
}
async fn jobs(client: &Client, config: &Config) -> Result<Vec<JobRecord>> {
    let mut all = Vec::new();
    let mut cursor = None;
    let mut seen = BTreeSet::new();
    loop {
        let page = client
            .list(ListRequest {
                filter: ListFilter {
                    tenant: Some(config.tenant.clone()),
                    project: Some(config.project.clone()),
                    states: [
                        JobState::Queued,
                        JobState::Admitted,
                        JobState::Starting,
                        JobState::Running,
                        JobState::Paused,
                        JobState::Suspended,
                        JobState::RetryWait,
                    ]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                },
                cursor,
                limit: 128,
            })
            .await?;
        for job in page.jobs {
            ensure!(
                seen.insert(job.id),
                "job pagination changed or duplicated identity"
            );
            all.push(job);
        }
        if page.next_cursor.is_none() {
            return Ok(all);
        }
        ensure!(
            all.len() < 1024,
            "active job census exceeds 1024; counts withheld"
        );
        cursor = page.next_cursor;
    }
}
fn queue_rows(jobs: &[JobRecord], at: u64, blocked: Option<&str>) -> Result<Vec<Value>> {
    let mut grouped: BTreeMap<String, Vec<&JobRecord>> = BTreeMap::new();
    for job in jobs {
        let name = match &job.spec.workload {
            Workload::TrustedService { service, .. } => service.to_string(),
            _ => "other native workload".into(),
        };
        grouped.entry(name).or_default().push(job);
    }
    ensure!(grouped.len() <= 120, "too many service queues");
    let mut rows = Vec::new();
    for (name, jobs) in grouped {
        let ready = jobs.iter().filter(|j| j.state == JobState::Queued).count();
        let running = jobs.iter().filter(|j| j.state == JobState::Running).count();
        let oldest = jobs
            .iter()
            .filter(|j| j.state == JobState::Queued)
            .map(|j| at.saturating_sub(j.created_at.as_unix_millis()) / 1000)
            .max();
        let idle = if let Some(reason) = blocked {
            Some(reason.to_string())
        } else if ready == 0 && running == 0 {
            Some(
                "No Running or Queued job; paused, starting, admitted or retry-wait state exists"
                    .into(),
            )
        } else if ready > 0 && running == 0 {
            Some("Queued; scheduler placement reason unavailable".into())
        } else {
            None
        };
        rows.push(json!({"name":text(name),"ready":ready,"running":running,"oldest_age_seconds":oldest,"idle_reason":idle}));
    }
    if rows.is_empty() {
        rows.push(json!({"name":"Scoped active native jobs","ready":0,"running":0,"oldest_age_seconds":null,"idle_reason":"No active job in scoped scheduler census"}));
    }
    Ok(rows)
}
pub async fn observe(config: &Config) -> Result<Snapshot> {
    ensure!(
        !config.scope.is_empty() && text(&config.scope) == config.scope,
        "scope must be bounded printable text"
    );
    let mut snapshot = Snapshot { schema:"jain.jope_native_status.v1", observed_at_unix_ms:now()?, scope:config.scope.clone(), resources:vec![], queues:vec![], continuations:vec![], corpus:None, checkpoints:vec![], comparisons:vec![], winners:vec![], notes:vec![
        text(format!("Scheduler tenant={} project={}; active state census, not an atomic transaction", config.tenant, config.project)),
        "Ready denotes SmartCluster Queued; starting, paused and retrying jobs are not numerical completions.".into(),
        "CPU is aggregate cgroup usage / effective CPU set, not useful-work occupancy; GPU is device-wide NVML utilization.".into(),
        "No independently verified scientific receipt feed configured: continuation, corpus, checkpoint, matched comparison and winner evidence unavailable.".into(),
    ], source_errors:vec![], scientific_sources:vec![], scientific_verified_at_unix_ms:None };
    let client = Client::new(&config.socket);
    let mut reason = None;
    match tokio::time::timeout(Duration::from_secs(2), client.status()).await {
        Ok(Ok(status)) => {
            snapshot.notes.push(text(format!(
                "Daemon {} paused={} admission_clamped={} pressure cpu={:?} memory={:?} io={:?}",
                status.daemon_version,
                status.paused,
                status.pressure.admissions_clamped,
                status.pressure.cpu,
                status.pressure.memory,
                status.pressure.io
            )));
            if status.paused {
                reason = Some("Scheduler globally paused");
            } else if status.pressure.admissions_clamped {
                reason = Some("Scheduler pressure clamps new admission");
            }
        }
        result => snapshot
            .source_errors
            .push(text(format!("Scheduler status unavailable: {result:?}"))),
    }
    match tokio::time::timeout(Duration::from_secs(4), jobs(&client, config)).await {
        Ok(Ok(jobs)) => match queue_rows(&jobs, snapshot.observed_at_unix_ms, reason) {
            Ok(rows) => snapshot.queues = rows,
            Err(error) => snapshot.source_errors.push(text(error.to_string())),
        },
        result => snapshot
            .source_errors
            .push(text(format!("Scoped active jobs unavailable: {result:?}"))),
    }
    match kernel(&config.cgroup).await {
        Ok(row) => snapshot.resources.push(row),
        Err(error) => snapshot
            .source_errors
            .push(text(format!("Kernel cgroup sample unavailable: {error:#}"))),
    }
    let terminal = ListRequest {
        filter: ListFilter {
            tenant: Some(config.tenant.clone()),
            project: Some(config.project.clone()),
            states: [JobState::Failed, JobState::Cancelled, JobState::Completed]
                .into_iter()
                .collect(),
            ..Default::default()
        },
        cursor: None,
        limit: 128,
    };
    match tokio::time::timeout(Duration::from_secs(2), client.list(terminal)).await {
        Ok(Ok(mut page)) => {
            if page.next_cursor.is_some() {
                snapshot.notes.push("Terminal job sample truncated to first 128 records; displayed records are not necessarily the latest globally.".into());
            }
            page.jobs
                .sort_by_key(|job| std::cmp::Reverse(job.updated_at));
            for job in page.jobs.into_iter().take(8) {
                snapshot.notes.push(text(format!(
                    "Native process terminal record {} at {}: {:?}; numerical completion independently unverified",
                    job.id, job.updated_at, job.terminal
                )));
            }
        }
        result => snapshot.source_errors.push(text(format!(
            "Terminal process records unavailable: {result:?}"
        ))),
    }
    match gpu().await {
        Ok(rows) => snapshot.resources.extend(rows),
        Err(error) => snapshot
            .source_errors
            .push(text(format!("NVML sample unavailable: {error:#}"))),
    }
    Ok(snapshot)
}
pub fn publish(config: &Config, snapshot: &Snapshot) -> Result<()> {
    let parent = config.output.parent().context("snapshot parent missing")?;
    physical(parent)?;
    ensure!(
        config.output.is_absolute() && config.output.file_name().is_some(),
        "absolute snapshot file required"
    );
    if config.output.try_exists()? {
        physical(&config.output)?;
        ensure!(config.output.is_file(), "snapshot must be a regular file");
    }
    let bytes = serde_json::to_vec(snapshot)?;
    ensure!(bytes.len() <= 4 * 1024 * 1024, "snapshot exceeds 4 MiB");
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(&bytes)?;
    temporary.as_file().sync_all()?;
    temporary.persist(&config.output)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cpu_basis_and_reset_are_explicit() {
        assert_eq!(cpu_count("8-11,72-75\n").unwrap(), 8);
        assert_eq!(cpu_count("1-3,2-4").unwrap(), 4);
        assert!(cpu_count("4-2").is_err());
        assert_eq!(
            cpu_percent(100, 4_000_100, Duration::from_secs(1), 8).unwrap(),
            50.0
        );
        assert!(cpu_percent(100, 0, Duration::from_secs(1), 8).is_err());
    }
    #[test]
    fn nvml_unknown_is_not_zero() {
        assert!(gpu_rows("GPU-test, N/A, 1").is_err());
        assert!(gpu_rows("GPU-test, 101, 1").is_err());
        let rows = gpu_rows("GPU-test, 0, 1").unwrap();
        assert_eq!(rows[0]["gpu_percent"], 0.0);
        assert_eq!(rows[0]["gpu_memory_bytes"], 1048576);
    }
    #[test]
    fn control_text_cannot_reach_terminal() {
        assert_eq!(text("hello\n\u{1b}world"), "helloworld");
        assert_eq!(text("x".repeat(600)).len(), 512);
    }
    #[test]
    fn publication_is_complete_and_refuses_symlink_targets() {
        let temp = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(temp.path()).unwrap();
        let config = Config {
            socket: root.join("unused.sock"),
            cgroup: root.clone(),
            output: root.join("status.json"),
            tenant: TenantId::new("test").unwrap(),
            project: ProjectId::new("test").unwrap(),
            scope: "test fixture only".into(),
        };
        let snapshot = Snapshot {
            schema: "jain.jope_native_status.v1",
            observed_at_unix_ms: 1000,
            scope: config.scope.clone(),
            resources: vec![],
            queues: vec![],
            continuations: vec![],
            corpus: None,
            checkpoints: vec![],
            comparisons: vec![],
            winners: vec![],
            notes: vec![],
            source_errors: vec!["source unavailable".into()],
            scientific_sources: vec![],
            scientific_verified_at_unix_ms: None,
        };
        publish(&config, &snapshot).unwrap();
        let first = fs::read(&config.output).unwrap();
        let decoded: Value = serde_json::from_slice(&first).unwrap();
        assert!(decoded["corpus"].is_null());
        assert_eq!(decoded["source_errors"][0], "source unavailable");
        let mut replacement = snapshot;
        replacement.observed_at_unix_ms = 2000;
        publish(&config, &replacement).unwrap();
        let decoded: Value = serde_json::from_slice(&fs::read(&config.output).unwrap()).unwrap();
        assert_eq!(decoded["observed_at_unix_ms"], 2000);
        fs::rename(&config.output, root.join("original.json")).unwrap();
        std::os::unix::fs::symlink(root.join("original.json"), &config.output).unwrap();
        assert!(publish(&config, &replacement).is_err());
        assert_eq!(
            serde_json::from_slice::<Value>(&fs::read(root.join("original.json")).unwrap())
                .unwrap(),
            decoded
        );
    }
}
