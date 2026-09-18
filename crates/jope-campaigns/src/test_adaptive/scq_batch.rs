use super::*;
use jope_dime_regressor::paths;
use serde_json::json;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

pub(super) const SCQ_BATCH_SCHEMA: &str = "JopeGolden42ScqBatchManifest.v1";
pub(super) const SCQ_BATCH_RECEIPT_SCHEMA: &str = "JopeGolden42ScqBatchReceipt.v1";
pub(super) const SCQ_WORKER_RECEIPT_SCHEMA: &str = "JopeGolden42ScqBatchWorkerReceipt.v1";
pub(super) const SCQ_STAGING_SCHEMA: &str = "JopeGolden42ScqStagingReceipt.v1";
pub(super) const AGGRESSIVE_BATCH_SCHEMA: &str = "JopeGolden42AggressiveBatchManifest.v1";
pub(super) const AGGRESSIVE_BATCH_RECEIPT_SCHEMA: &str = "JopeGolden42AggressiveBatchReceipt.v1";
pub(super) const AGGRESSIVE_WORKER_RECEIPT_SCHEMA: &str =
    "JopeGolden42AggressiveBatchWorkerReceipt.v1";
pub(super) const SCQ_PROJECT: &str = "jope";
pub(super) const SCQ_CPUS: usize = 90;
pub(super) const AGGRESSIVE_CPUS: usize = 120;
pub(super) const AGGRESSIVE_WORKERS: usize = 240;
pub(super) const AGGRESSIVE_DATASET_LANES: usize = 42;
pub(super) const AGGRESSIVE_CANDIDATES_PER_NODE: usize = 12;
const AGGRESSIVE_DATASET_POPULATION: usize = AGGRESSIVE_CANDIDATES_PER_NODE * NODES.len();
const WAVE_MAX_LANE_POPULATION: usize = 36;
const WAVE_MAX_CANDIDATES_PER_NODE: usize = 12;
const AGGRESSIVE_MAX_REQUESTS_PER_NODE: usize =
    AGGRESSIVE_DATASET_LANES * AGGRESSIVE_CANDIDATES_PER_NODE;
pub(super) const CANDIDATE_TIMEOUT_SECONDS: usize = 30;
pub(super) const AGGRESSIVE_CANDIDATE_TIMEOUT_SECONDS: usize = 120;
pub(super) const NODES: [&str; 3] = ["xbabe1", "xbabe2", "xbabe3"];
const GOLDEN46_RUN_ID: &str = "golden46-new3-20260723-v4";
const GOLDEN46_TRANSFER_RUN_ID: &str = "golden46-new3-20260723-v4-target-family-informed";
const GOLDEN46_EFFECTIVELY_UNBOUNDED_RUNTIME_HOURS: usize = 876_000;
jope_paths::once_str!(
    fn scq_binary_str => paths::scq_binary()
);
jope_paths::once_str!(
    fn remote_root_str => paths::legacy_tree("jope-golden42")
);
jope_paths::once_str!(
    fn node_lease_root_str => paths::legacy_tree("jope-golden42-controller-node-leases")
);
const NODE_LEASE_SLOTS: usize = 5;
const ISLAND_START_GENERATION: usize = 20;
const ISLAND_END_GENERATION: usize = 30;

#[derive(Clone, Copy)]
struct BatchFlavor {
    manifest_schema: &'static str,
    receipt_schema: &'static str,
    worker_receipt_schema: &'static str,
    worker_command: &'static str,
    cpus: usize,
    candidate_timeout_seconds: usize,
    max_runtime_hours: usize,
}

const FROZEN_FLAVOR: BatchFlavor = BatchFlavor {
    manifest_schema: SCQ_BATCH_SCHEMA,
    receipt_schema: SCQ_BATCH_RECEIPT_SCHEMA,
    worker_receipt_schema: SCQ_WORKER_RECEIPT_SCHEMA,
    worker_command: "test-adaptive-island-batch-worker",
    cpus: SCQ_CPUS,
    candidate_timeout_seconds: CANDIDATE_TIMEOUT_SECONDS,
    max_runtime_hours: 1,
};

const AGGRESSIVE_FLAVOR: BatchFlavor = BatchFlavor {
    manifest_schema: AGGRESSIVE_BATCH_SCHEMA,
    receipt_schema: AGGRESSIVE_BATCH_RECEIPT_SCHEMA,
    worker_receipt_schema: AGGRESSIVE_WORKER_RECEIPT_SCHEMA,
    worker_command: "golden42-aggressive-batch-worker",
    cpus: AGGRESSIVE_CPUS,
    candidate_timeout_seconds: AGGRESSIVE_CANDIDATE_TIMEOUT_SECONDS,
    max_runtime_hours: 1,
};

fn is_golden46_run(run_id: &str) -> bool {
    matches!(run_id, GOLDEN46_RUN_ID | GOLDEN46_TRANSFER_RUN_ID)
}

fn aggressive_flavor(run_id: &str) -> BatchFlavor {
    if is_golden46_run(run_id) {
        BatchFlavor {
            candidate_timeout_seconds: 0,
            max_runtime_hours: GOLDEN46_EFFECTIVELY_UNBOUNDED_RUNTIME_HOURS,
            ..AGGRESSIVE_FLAVOR
        }
    } else {
        AGGRESSIVE_FLAVOR
    }
}

fn maximum_batch_requests(flavor: BatchFlavor) -> usize {
    if flavor.manifest_schema == AGGRESSIVE_BATCH_SCHEMA {
        AGGRESSIVE_MAX_REQUESTS_PER_NODE
    } else {
        flavor.cpus
    }
}

fn reserved_batch_cpus(flavor: BatchFlavor, request_count: usize) -> Result<usize> {
    if request_count == 0 || request_count > maximum_batch_requests(flavor) {
        bail!("SCQ batch request count is outside its resource contract");
    }
    Ok(if flavor.manifest_schema == AGGRESSIVE_BATCH_SCHEMA {
        request_count.min(AGGRESSIVE_CPUS)
    } else {
        flavor.cpus
    })
}

struct NodeLease {
    file: File,
}

impl NodeLease {
    fn acquire(node: &str) -> Result<Self> {
        Self::acquire_at(Path::new(node_lease_root_str()), node)
    }

    fn acquire_at(lock_dir: &Path, node: &str) -> Result<Self> {
        if !NODES.contains(&node) {
            bail!("refuse SCQ lease for unknown node {node}");
        }
        fs::create_dir_all(&lock_dir)?;
        loop {
            for slot in 0..NODE_LEASE_SLOTS {
                let file = OpenOptions::new()
                    .create(true)
                    .read(true)
                    .write(true)
                    .open(lock_dir.join(format!("{node}.slot-{slot}.lock")))?;
                let result =
                    unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
                if result == 0 {
                    return Ok(Self { file });
                }
                let error = std::io::Error::last_os_error();
                if !matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                ) {
                    return Err(error).context("acquire SCQ node lease slot");
                }
            }
            thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for NodeLease {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ScqPolicy {
    pub nodes: Vec<String>,
    pub project: String,
    pub core_cap_per_node: usize,
    pub candidate_threads: usize,
    pub candidate_timeout_seconds: usize,
    pub batch_max_runtime_hours: usize,
    pub infrastructure_retries: usize,
    pub preemptible: bool,
    pub routing: String,
    pub failover: String,
    pub concurrent_datasets: String,
    pub batch_schema: String,
    pub receipt_schema: String,
}

impl ScqPolicy {
    pub(super) fn fixed() -> Self {
        Self {
            nodes: NODES.iter().map(|node| (*node).to_string()).collect(),
            project: SCQ_PROJECT.to_string(),
            core_cap_per_node: SCQ_CPUS,
            candidate_threads: 1,
            candidate_timeout_seconds: CANDIDATE_TIMEOUT_SECONDS,
            batch_max_runtime_hours: 1,
            infrastructure_retries: 1,
            preemptible: false,
            routing: "primary ring by (ordered dataset index + generation + batch ordinal) mod 3"
                .to_string(),
            failover: "one infrastructure retry on the next node in the ring".to_string(),
            concurrent_datasets:
                "all dataset controllers may advance concurrently; barriers are dataset-local; five controller lease slots per node defer final CPU admission to SCQ"
                    .to_string(),
            batch_schema: SCQ_BATCH_SCHEMA.to_string(),
            receipt_schema: SCQ_BATCH_RECEIPT_SCHEMA.to_string(),
        }
    }

    pub(super) fn aggressive_wave() -> Self {
        Self {
            nodes: NODES.iter().map(|node| (*node).to_string()).collect(),
            project: SCQ_PROJECT.to_string(),
            core_cap_per_node: AGGRESSIVE_CPUS,
            candidate_threads: 1,
            candidate_timeout_seconds: AGGRESSIVE_CANDIDATE_TIMEOUT_SECONDS,
            batch_max_runtime_hours: 1,
            infrastructure_retries: 3,
            preemptible: false,
            routing: "candidate slot plus generation modulo the three-node ring".to_string(),
            failover: "missing candidates retry on each subsequent node in the ring".to_string(),
            concurrent_datasets:
                "all active dataset generations share over-decomposed batches through five controller lease slots per node; SCQ enforces final CPU admission"
                    .to_string(),
            batch_schema: AGGRESSIVE_BATCH_SCHEMA.to_string(),
            receipt_schema: AGGRESSIVE_BATCH_RECEIPT_SCHEMA.to_string(),
        }
    }

    pub(super) fn golden46_wave() -> Self {
        let mut policy = Self::aggressive_wave();
        policy.candidate_timeout_seconds = 0;
        policy.batch_max_runtime_hours = GOLDEN46_EFFECTIVELY_UNBOUNDED_RUNTIME_HOURS;
        policy.infrastructure_retries = 1;
        policy.failover = "one infrastructure retry on a different node".to_string();
        policy
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct StagedDataset {
    pub dataset_hash: String,
    pub train_sha256: String,
    pub test_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct StagedNode {
    pub node: String,
    pub hostname: String,
    pub bundle_root: String,
    pub release_binary_sha256: String,
    pub datasets: Vec<StagedDataset>,
    pub true_job_id: String,
    pub true_job_show_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ScqStagingReceipt {
    pub schema: String,
    pub benchmark_valid: bool,
    pub scq_binary_sha256: String,
    pub doctor_output_sha256: String,
    pub release_binary_sha256: String,
    pub nodes: Vec<StagedNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct ScqBatchManifest {
    pub schema: String,
    pub benchmark_valid: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_adaptive: Option<bool>,
    pub batch_id: String,
    pub run_id: String,
    pub node: String,
    pub attempt: usize,
    pub project: String,
    pub cpus: usize,
    pub candidate_threads: usize,
    pub candidate_timeout_seconds: usize,
    pub max_runtime_hours: usize,
    pub preemptible: bool,
    pub scq_binary_sha256: String,
    pub release_binary_sha256: String,
    pub bundle_root: String,
    pub requests: Vec<CandidateRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct BatchCandidateReceipt {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_adaptive: Option<bool>,
    pub dataset_hash: String,
    pub generation: usize,
    pub slot: usize,
    pub attempt: usize,
    pub semantic_sha256: String,
    pub file: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ScqWorkerReceipt {
    schema: String,
    benchmark_valid: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    test_adaptive: Option<bool>,
    batch_id: String,
    node: String,
    hostname: String,
    manifest_sha256: String,
    started_at_unix_ms: u128,
    finished_at_unix_ms: u128,
    candidates: Vec<BatchCandidateReceipt>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct ScqBatchReceipt {
    pub schema: String,
    pub benchmark_valid: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_adaptive: Option<bool>,
    pub batch_id: String,
    pub scq_job_id: String,
    pub node: String,
    pub hostname: Option<String>,
    pub attempt: usize,
    pub manifest_file: String,
    pub manifest_sha256: String,
    pub scq_spec_file: String,
    pub scq_spec_sha256: String,
    pub worker_receipt_sha256: Option<String>,
    pub submitted_at_unix_ms: u128,
    pub started_at_unix_ms: Option<u128>,
    pub worker_finished_at_unix_ms: Option<u128>,
    pub finished_at_unix_ms: u128,
    pub scq_state: String,
    pub exit_code: Option<i64>,
    pub scq_show_file: String,
    pub scq_show_sha256: String,
    pub candidates: Vec<BatchCandidateReceipt>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct ExpectedFresh {
    pub dataset_hash: String,
    pub generation: usize,
    pub slot: usize,
    pub attempt: usize,
    pub semantic_sha256: String,
    pub node: String,
    pub scq_job_id: String,
}

pub(super) fn command_output(mut command: Command, label: &str) -> Result<std::process::Output> {
    let output = command
        .output()
        .with_context(|| format!("launch {label}"))?;
    if !output.status.success() {
        bail!(
            "{label} failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output)
}

pub(super) fn scq_json(args: &[&str]) -> Result<Value> {
    let mut command = Command::new(scq_binary_str());
    command.arg("--json").args(args);
    let output = command_output(command, "scq")?;
    serde_json::from_slice(&output.stdout).context("parse scq JSON")
}

pub(super) fn scq_job<'a>(
    value: &'a Value,
    id: &str,
) -> Result<&'a serde_json::Map<String, Value>> {
    value
        .pointer("/Job/job")
        .and_then(Value::as_object)
        .filter(|job| job.get("id").and_then(Value::as_str) == Some(id))
        .context("scq show returned the wrong job")
}

fn active_jobs_on_node(queue: &Value, node: &str) -> Result<Vec<String>> {
    queue
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .context("SCQ queue response has no jobs array")?
        .iter()
        .filter_map(|job| {
            let id = job.get("id").and_then(Value::as_str);
            let assigned = job.get("node").and_then(Value::as_str);
            let state = job.get("state").and_then(Value::as_str);
            match (id, assigned, state) {
                (Some(_), _, Some("completed" | "failed" | "cancelled")) => None,
                (Some(id), Some(assigned), Some(_)) if assigned == node => Some(Ok(id.to_string())),
                (Some(_), Some(_), Some(_)) => None,
                (Some(_), None, Some(_)) => None,
                _ => Some(Err(anyhow::anyhow!(
                    "SCQ queue job has no exact id/node/state binding"
                ))),
            }
        })
        .collect()
}

fn is_campaign_job_id(id: &str) -> bool {
    id.starts_with("jope-g42-")
}

fn competing_node_processes(
    process_table: &str,
    node: &str,
    local_hostname: &str,
    controller_pid: &str,
) -> Vec<String> {
    process_table
        .lines()
        .filter(|line| {
            let lower = line.to_ascii_lowercase();
            let comm = line.split_whitespace().nth(1).unwrap_or_default();
            let is_jope_executable = comm.starts_with("jope-dime-regre");
            let is_scq_executable = comm == "scq";
            let is_this_controller =
                node == local_hostname && line.split_whitespace().next() == Some(controller_pid);
            let is_campaign_controller = lower.contains("run-golden42-aggressive")
                || lower.contains("run-golden46")
                || lower.contains("run-tabarena-lite-islands")
                || lower.contains("golden42-aggressive-batch-worker")
                || lower.contains("test-adaptive-worker --request");
            (is_jope_executable || is_scq_executable)
                && !lower.contains("scqd")
                && !lower.contains("scq controller serve")
                && !lower.contains("scq agent status")
                && !is_campaign_controller
                && !is_this_controller
        })
        .map(str::trim)
        .map(str::to_string)
        .collect()
}

fn ensure_node_available_after_lease(node: &str) -> Result<()> {
    let competing_jobs = active_jobs_on_node(&scq_json(&["ls"])?, node)?
        .into_iter()
        .filter(|id| !is_campaign_job_id(id))
        .collect::<Vec<_>>();
    if !competing_jobs.is_empty() {
        bail!(
            "SCQ node {node} became occupied while waiting for its lease: {}",
            competing_jobs.join(", ")
        );
    }
    let local_hostname = String::from_utf8(
        command_output(
            {
                let mut command = Command::new("hostname");
                command.arg("-s");
                command
            },
            "controller hostname",
        )?
        .stdout,
    )?
    .trim()
    .to_string();
    let process_table = String::from_utf8(
        ssh_output(
            node,
            &["ps", "-eo", "pid=,comm=,args="],
            "recheck node process occupancy",
        )?
        .stdout,
    )?;
    let competing = competing_node_processes(
        &process_table,
        node,
        &local_hostname,
        &std::process::id().to_string(),
    );
    if !competing.is_empty() {
        bail!("SCQ node {node} became occupied after preflight");
    }
    Ok(())
}

pub(super) fn wait_for_job(id: &str) -> Result<Value> {
    loop {
        let show = scq_json(&["show", id])?;
        let state = scq_job(&show, id)?
            .get("state")
            .and_then(Value::as_str)
            .context("scq job has no state")?;
        if matches!(state, "completed" | "failed" | "cancelled") {
            return Ok(show);
        }
        thread::sleep(Duration::from_secs(1));
    }
}

fn submit_true(node: &str, suffix: &str) -> Result<(String, String)> {
    let id = format!("jope-g42-preflight-{node}-{suffix}");
    let spec = format!(
        "[job]\nid = {id:?}\nproject = {project:?}\ncommand = \"/bin/true\"\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = 1\nomp_threads = 1\nram_gb = 0\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = \"Golden42 C++17 compiler available\"\ncommand = \"/usr/bin/test\"\nargs = [\"-x\", \"/usr/bin/g++\"]\ntimeout_secs = 5\n\n[preempt]\npreemptible = false\nsignal = \"SIGTERM\"\ngrace_seconds = 60\n\n[limits]\nmax_runtime_hours = 1\nmax_restarts = 0\n",
        project = SCQ_PROJECT,
    );
    let spec_path = std::env::temp_dir().join(format!("{id}.toml"));
    fs::write(&spec_path, spec.as_bytes())?;
    let mut command = Command::new(scq_binary_str());
    command.arg("submit").arg(&spec_path);
    let submit = command_output(command, "pinned SCQ preflight");
    fs::remove_file(&spec_path)?;
    submit?;
    let show = wait_for_job(&id)?;
    let job = scq_job(&show, &id)?;
    if job.get("state").and_then(Value::as_str) != Some("completed")
        || job.get("exit_code").and_then(Value::as_i64) != Some(0)
    {
        bail!("pinned SCQ preflight {id} did not complete successfully on {node}");
    }
    Ok((id, sha256_bytes(&canonical_json(&show)?)))
}

pub(super) fn ssh_output(
    node: &str,
    remote_args: &[&str],
    label: &str,
) -> Result<std::process::Output> {
    let output = ssh_raw(node, remote_args, label)?;
    if !output.status.success() {
        bail!(
            "{label} failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output)
}

fn ssh_raw(node: &str, remote_args: &[&str], label: &str) -> Result<std::process::Output> {
    let mut command = Command::new("ssh");
    command.arg(node).arg("--").args(remote_args);
    command.output().with_context(|| format!("launch {label}"))
}

pub(super) fn remote_sha256(node: &str, path: &str) -> Result<String> {
    let output = ssh_output(node, &["sha256sum", path], "remote sha256sum")?;
    String::from_utf8(output.stdout)?
        .split_whitespace()
        .next()
        .map(str::to_string)
        .context("remote sha256sum returned no digest")
}

fn remote_file_mode(node: &str, path: &str) -> Result<Option<u32>> {
    let stat = ssh_raw(node, &["stat", "-c", "%f", "--", path], "remote stat")?;
    if stat.status.success() {
        let raw = String::from_utf8(stat.stdout)?;
        let mode = u32::from_str_radix(raw.trim(), 16).context("parse remote stat mode")?;
        return Ok(Some(mode));
    }
    let symlink = ssh_raw(node, &["test", "-L", path], "probe remote symlink")?;
    if symlink.status.success() {
        bail!("remote immutable path is a symbolic link: {node}:{path}");
    }
    if symlink.status.code() != Some(1) {
        bail!(
            "remote symlink probe failed on {node}:{path} ({}): {}",
            symlink.status,
            String::from_utf8_lossy(&symlink.stderr).trim()
        );
    }
    let exists = ssh_raw(node, &["test", "-e", path], "probe remote path")?;
    if exists.status.success() {
        bail!("remote stat failed for existing path {node}:{path}");
    }
    if exists.status.code() != Some(1) {
        bail!(
            "remote existence probe failed on {node}:{path} ({}): {}",
            exists.status,
            String::from_utf8_lossy(&exists.stderr).trim()
        );
    }
    Ok(None)
}

pub(super) fn validate_remote_immutable(
    node: &str,
    path: &str,
    expected_sha256: &str,
    expected_permissions: u32,
) -> Result<()> {
    let mode = remote_file_mode(node, path)?
        .with_context(|| format!("remote immutable path is absent: {node}:{path}"))?;
    validate_immutable_mode(mode, expected_permissions)
        .with_context(|| format!("remote immutable mode mismatch on {node}:{path}"))?;
    let actual_sha256 = remote_sha256(node, path)?;
    if actual_sha256 != expected_sha256 {
        bail!(
            "remote immutable hash mismatch on {node}:{path}: got {actual_sha256}, expected {expected_sha256}"
        );
    }
    Ok(())
}

fn validate_immutable_mode(mode: u32, expected_permissions: u32) -> Result<()> {
    if mode & 0o170000 != 0o100000 || mode & 0o7777 != expected_permissions {
        bail!("got {mode:#o}, expected regular file with permissions {expected_permissions:#o}");
    }
    Ok(())
}

fn remove_remote_temporary(node: &str, temporary: &str) -> Result<()> {
    ssh_output(
        node,
        &["rm", "-f", "--", temporary],
        "remove remote staging temporary",
    )?;
    Ok(())
}

pub(super) fn copy_atomic(
    node: &str,
    source: &Path,
    destination: &str,
    executable: bool,
) -> Result<()> {
    let parent = Path::new(destination)
        .parent()
        .context("remote destination has no parent")?
        .to_string_lossy()
        .into_owned();
    let expected_sha256 = sha256_file(source)?;
    let expected_permissions = if executable { 0o555 } else { 0o444 };
    if remote_file_mode(node, destination)?.is_some() {
        return validate_remote_immutable(
            node,
            destination,
            &expected_sha256,
            expected_permissions,
        );
    }
    ssh_output(
        node,
        &["mkdir", "-p", &parent],
        "create remote staging directory",
    )?;
    let temporary = String::from_utf8(
        ssh_output(
            node,
            &["mktemp", "-p", &parent, ".jope-stage.XXXXXXXXXX"],
            "create exclusive remote staging temporary",
        )?
        .stdout,
    )?
    .trim()
    .to_string();
    if Path::new(&temporary).parent() != Some(Path::new(&parent)) {
        bail!("remote mktemp returned a path outside the staging directory");
    }
    let mut copy = Command::new("scp");
    copy.arg("--")
        .arg(source)
        .arg(format!("{node}:{temporary}"));
    if let Err(error) = command_output(copy, "copy immutable staging file") {
        let _ = remove_remote_temporary(node, &temporary);
        return Err(error);
    }
    if executable {
        ssh_output(
            node,
            &["chmod", "0555", "--", &temporary],
            "chmod staged binary",
        )?;
    } else {
        ssh_output(
            node,
            &["chmod", "0444", "--", &temporary],
            "chmod staged dataset",
        )?;
    }
    validate_remote_immutable(node, &temporary, &expected_sha256, expected_permissions)?;
    let publish = ssh_raw(
        node,
        &["ln", "--", &temporary, destination],
        "publish immutable staged file",
    )?;
    remove_remote_temporary(node, &temporary)?;
    if !publish.status.success() {
        validate_remote_immutable(node, destination, &expected_sha256, expected_permissions)
            .with_context(|| {
                format!(
                    "exclusive publish failed ({}): {}",
                    publish.status,
                    String::from_utf8_lossy(&publish.stderr).trim()
                )
            })?;
        return Ok(());
    }
    validate_remote_immutable(node, destination, &expected_sha256, expected_permissions)
}

pub(super) fn scq_binary_sha256() -> Result<String> {
    sha256_file(Path::new(scq_binary_str()))
}

pub(super) fn remote_bundle_root(release_binary_sha256: &str) -> String {
    format!(
        "{remote_root_str}/bundles/{release_binary_sha256}",
        remote_root_str = remote_root_str()
    )
}

pub(super) fn remote_dataset_root(release_binary_sha256: &str, dataset_hash: &str) -> String {
    format!(
        "{}/datasets/{dataset_hash}",
        remote_bundle_root(release_binary_sha256)
    )
}

pub(super) fn preflight_and_stage(
    registry: &Path,
    release_binary: &Path,
    release_binary_sha256: &str,
    specs: &[DatasetSpec],
) -> Result<ScqStagingReceipt> {
    if sha256_file(release_binary)? != release_binary_sha256 {
        bail!("release binary changed before SCQ staging");
    }
    let scq_hash = scq_binary_sha256()?;
    let doctor = command_output(
        {
            let mut command = Command::new(scq_binary_str());
            command.arg("doctor");
            command
        },
        "scq doctor",
    )?;
    let doctor_output_sha256 = sha256_bytes(&[doctor.stdout, doctor.stderr].concat());
    let suffix = format!("{}-{}", &release_binary_sha256[..8], unix_millis());
    let mut nodes = Vec::new();
    for node in NODES {
        let mut pinned = None;
        for attempt in 0..=2 {
            match submit_true(node, &format!("{suffix}-a{attempt}")) {
                Ok(receipt) => {
                    pinned = Some(receipt);
                    break;
                }
                Err(error) => {
                    eprintln!("pinned SCQ preflight attempt {attempt} failed on {node}: {error:#}")
                }
            }
        }
        let (true_job_id, true_job_show_sha256) =
            pinned.with_context(|| format!("pinned SCQ preflight exhausted retries on {node}"))?;
        let hostname =
            String::from_utf8(ssh_output(node, &["hostname", "-s"], "remote hostname")?.stdout)?
                .trim()
                .to_string();
        if hostname != node {
            bail!("SSH target {node} reports hostname {hostname}");
        }
        let bundle_root = remote_bundle_root(release_binary_sha256);
        let binary_destination = format!("{bundle_root}/jope-dime-regressor");
        copy_atomic(node, release_binary, &binary_destination, true)?;
        if remote_sha256(node, &binary_destination)? != release_binary_sha256 {
            bail!("staged release binary hash mismatch on {node}");
        }
        let mut datasets = Vec::new();
        for spec in specs {
            let dataset_root = registry.join(&spec.dataset_hash);
            let remote_root = remote_dataset_root(release_binary_sha256, &spec.dataset_hash);
            let remote_train = format!("{remote_root}/train.csv");
            let remote_test = format!("{remote_root}/test.csv");
            copy_atomic(node, &dataset_root.join("train.csv"), &remote_train, false)?;
            copy_atomic(node, &dataset_root.join("test.csv"), &remote_test, false)?;
            if remote_sha256(node, &remote_train)? != spec.train_sha256
                || remote_sha256(node, &remote_test)? != spec.test_sha256
            {
                bail!(
                    "staged dataset {} hash mismatch on {node}",
                    spec.dataset_hash
                );
            }
            datasets.push(StagedDataset {
                dataset_hash: spec.dataset_hash.clone(),
                train_sha256: spec.train_sha256.clone(),
                test_sha256: spec.test_sha256.clone(),
            });
        }
        nodes.push(StagedNode {
            node: node.to_string(),
            hostname,
            bundle_root,
            release_binary_sha256: release_binary_sha256.to_string(),
            datasets,
            true_job_id,
            true_job_show_sha256,
        });
    }
    Ok(ScqStagingReceipt {
        schema: SCQ_STAGING_SCHEMA.to_string(),
        benchmark_valid: false,
        scq_binary_sha256: scq_hash,
        doctor_output_sha256,
        release_binary_sha256: release_binary_sha256.to_string(),
        nodes,
    })
}

pub(super) fn validate_staging(
    staging: &ScqStagingReceipt,
    release_binary_sha256: &str,
    specs: &[DatasetSpec],
) -> Result<()> {
    if staging.schema != SCQ_STAGING_SCHEMA
        || staging.benchmark_valid
        || staging.release_binary_sha256 != release_binary_sha256
        || staging.nodes.len() != NODES.len()
    {
        bail!("invalid Golden42 SCQ staging receipt");
    }
    for (index, node) in staging.nodes.iter().enumerate() {
        if node.node != NODES[index]
            || node.hostname != NODES[index]
            || node.release_binary_sha256 != release_binary_sha256
            || node.bundle_root != remote_bundle_root(release_binary_sha256)
            || node.datasets.len() != specs.len()
        {
            bail!("Golden42 staging node binding mismatch");
        }
        for (recorded, expected) in node.datasets.iter().zip(specs) {
            if recorded.dataset_hash != expected.dataset_hash
                || recorded.train_sha256 != expected.train_sha256
                || recorded.test_sha256 != expected.test_sha256
            {
                bail!("Golden42 staging dataset binding mismatch");
            }
        }
    }
    Ok(())
}

pub(super) fn verify_remote_staging(staging: &ScqStagingReceipt) -> Result<()> {
    for node in &staging.nodes {
        let binary = format!("{}/jope-dime-regressor", node.bundle_root);
        validate_remote_immutable(&node.node, &binary, &node.release_binary_sha256, 0o555)?;
        for dataset in &node.datasets {
            let root = format!("{}/datasets/{}", node.bundle_root, dataset.dataset_hash);
            validate_remote_immutable(
                &node.node,
                &format!("{root}/train.csv"),
                &dataset.train_sha256,
                0o444,
            )?;
            validate_remote_immutable(
                &node.node,
                &format!("{root}/test.csv"),
                &dataset.test_sha256,
                0o444,
            )?;
        }
    }
    Ok(())
}

pub(super) fn primary_node(dataset_index: usize, generation: usize, batch_ordinal: usize) -> usize {
    (dataset_index + generation.saturating_sub(ISLAND_START_GENERATION) + batch_ordinal)
        % NODES.len()
}

pub(super) fn retry_node(primary: usize) -> usize {
    (primary + 1) % NODES.len()
}

fn request_relative(request: &CandidateRequest) -> PathBuf {
    PathBuf::from("datasets")
        .join(&request.dataset_hash)
        .join("requests")
        .join(&request.purpose)
        .join(format!("generation-{:03}", request.generation))
        .join(format!(
            "slot-{:03}-attempt-{}.v1.json",
            request.slot, request.attempt
        ))
}

fn outcome_relative(request: &CandidateRequest) -> PathBuf {
    PathBuf::from("datasets")
        .join(&request.dataset_hash)
        .join("receipts")
        .join(&request.purpose)
        .join(format!("generation-{:03}", request.generation))
        .join(format!(
            "slot-{:03}-attempt-{}.v1.json",
            request.slot, request.attempt
        ))
}

fn candidate_output_name(request: &CandidateRequest) -> String {
    format!(
        "{}-g{:03}-s{:03}-a{}.v1.json",
        request.dataset_hash, request.generation, request.slot, request.attempt
    )
}

fn batch_id(
    run_id: &str,
    node: &str,
    attempt: usize,
    release_binary_sha256: &str,
    requests: &[CandidateRequest],
) -> Result<String> {
    let binding = json!({
        "run_id": run_id,
        "node": node,
        "attempt": attempt,
        "release_binary_sha256": release_binary_sha256,
        "requests": requests.iter().map(|request| json!({
            "dataset": request.dataset_hash,
            "purpose": request.purpose,
            "generation": request.generation,
            "slot": request.slot,
            "semantic": request.semantic_sha256,
        })).collect::<Vec<_>>(),
    });
    Ok(format!(
        "jope-g42-{}",
        &sha256_bytes(&canonical_json(&binding)?)[..24]
    ))
}

fn immutable_or_equal(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        if fs::read(path)? != bytes {
            bail!("immutable artifact collision at {}", path.display());
        }
    } else {
        immutable_write(path, bytes)?;
    }
    Ok(())
}

fn copy_remote_outputs(node: &str, remote: &str, local: &Path) -> Result<()> {
    fs::create_dir_all(local)?;
    let mut command = Command::new("scp");
    command
        .arg("-r")
        .arg("--")
        .arg(format!("{node}:{remote}/."))
        .arg(local);
    command_output(command, "pull SCQ batch outputs")?;
    Ok(())
}

fn submit_batch_job(manifest: &ScqBatchManifest, spec_path: &Path) -> Result<Value> {
    let mut command = Command::new(scq_binary_str());
    command.arg("submit").arg(spec_path);
    let output = command
        .output()
        .context("launch submit Golden42 SCQ batch")?;
    if !output.status.success() {
        let show = scq_json(&["show", &manifest.batch_id]).with_context(|| {
            format!(
                "submit Golden42 SCQ batch failed ({}): {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )
        })?;
        scq_job(&show, &manifest.batch_id)?;
    }
    wait_for_job(&manifest.batch_id)
}

fn run_batch(
    campaign_root: &Path,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    node: &str,
    attempt: usize,
    requests: Vec<CandidateRequest>,
) -> Result<(ScqBatchReceipt, Vec<CandidateReceipt>)> {
    run_batch_flavored(
        FROZEN_FLAVOR,
        campaign_root,
        release_binary_sha256,
        scq_binary_sha256,
        node,
        attempt,
        requests,
    )
}

fn run_batch_flavored(
    flavor: BatchFlavor,
    campaign_root: &Path,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    node: &str,
    attempt: usize,
    mut requests: Vec<CandidateRequest>,
) -> Result<(ScqBatchReceipt, Vec<CandidateReceipt>)> {
    let test_adaptive = (flavor.manifest_schema == AGGRESSIVE_BATCH_SCHEMA).then_some(true);
    let maximum_requests = maximum_batch_requests(flavor);
    if requests.is_empty() || requests.len() > maximum_requests {
        bail!("SCQ batch must contain 1..={maximum_requests} candidates");
    }
    let id = batch_id(
        &requests[0].run_id,
        node,
        attempt,
        release_binary_sha256,
        &requests,
    )?;
    for request in &mut requests {
        request.node = node.to_string();
        request.execution_host = "scq".to_string();
        request.scq_job_id = id.clone();
        request.attempt = attempt;
        request.dataset_root = remote_dataset_root(release_binary_sha256, &request.dataset_hash);
    }
    let bundle_root = remote_bundle_root(release_binary_sha256);
    let reserved_cpus = reserved_batch_cpus(flavor, requests.len())?;
    let manifest = ScqBatchManifest {
        schema: flavor.manifest_schema.to_string(),
        benchmark_valid: false,
        test_adaptive,
        batch_id: id.clone(),
        run_id: requests[0].run_id.clone(),
        node: node.to_string(),
        attempt,
        project: SCQ_PROJECT.to_string(),
        cpus: reserved_cpus,
        candidate_threads: 1,
        candidate_timeout_seconds: flavor.candidate_timeout_seconds,
        max_runtime_hours: flavor.max_runtime_hours,
        preemptible: false,
        scq_binary_sha256: scq_binary_sha256.to_string(),
        release_binary_sha256: release_binary_sha256.to_string(),
        bundle_root: bundle_root.clone(),
        requests,
    };
    let manifest_bytes = canonical_json(&manifest)?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let manifest_relative = PathBuf::from("batches/manifests").join(format!("{id}.v1.json"));
    let manifest_path = campaign_root.join(&manifest_relative);
    immutable_or_equal(&manifest_path, &manifest_bytes)?;
    for request in &manifest.requests {
        immutable_or_equal(
            &campaign_root.join(request_relative(request)),
            &canonical_json(request)?,
        )?;
    }
    let receipt_path = campaign_root
        .join("batches/receipts")
        .join(format!("{id}.v1.json"));
    if receipt_path.is_file() {
        let bytes = fs::read(&receipt_path)?;
        let receipt: ScqBatchReceipt = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&receipt)?
            || receipt.schema != flavor.receipt_schema
            || receipt.test_adaptive != test_adaptive
            || receipt.batch_id != id
            || receipt.scq_job_id != id
            || receipt.node != node
            || receipt.attempt != attempt
            || receipt.manifest_file != manifest_relative.to_string_lossy()
            || receipt.manifest_sha256 != manifest_sha256
        {
            bail!("immutable SCQ batch receipt changed on resume");
        }
        let mut receipts = Vec::new();
        for candidate in &receipt.candidates {
            let request = manifest.requests.iter().find(|request| {
                request.dataset_hash == candidate.dataset_hash
                    && request.generation == candidate.generation
                    && request.slot == candidate.slot
                    && request.attempt == candidate.attempt
                    && request.semantic_sha256 == candidate.semantic_sha256
            });
            let Some(request) = request else {
                bail!("resumed batch contains an unrequested candidate");
            };
            let path = campaign_root.join(outcome_relative(request));
            let candidate_bytes = fs::read(&path)?;
            if sha256_bytes(&candidate_bytes) != candidate.sha256 {
                bail!("resumed batch candidate receipt hash changed");
            }
            receipts.push(serde_json::from_slice(&candidate_bytes)?);
        }
        return Ok((receipt, receipts));
    }
    let _node_lease = NodeLease::acquire(node)?;
    if receipt_path.is_file() {
        bail!("SCQ batch receipt appeared while waiting for its node lease");
    }
    ensure_node_available_after_lease(node)?;
    let remote_batch_root = format!(
        "{remote_root_str}/batches/{id}",
        remote_root_str = remote_root_str()
    );
    let remote_manifest = format!("{remote_batch_root}/manifest.v1.json");
    let remote_output = format!("{remote_batch_root}/outputs");
    let binary = format!("{bundle_root}/jope-dime-regressor");
    let args = [
        "RAYON_NUM_THREADS=1".to_string(),
        "RUST_NUM_THREADS=1".to_string(),
        "OMP_NUM_THREADS=1".to_string(),
        "OPENBLAS_NUM_THREADS=1".to_string(),
        "MKL_NUM_THREADS=1".to_string(),
        "VECLIB_MAXIMUM_THREADS=1".to_string(),
        "NUMEXPR_NUM_THREADS=1".to_string(),
        binary,
        flavor.worker_command.to_string(),
        "--manifest".to_string(),
        remote_manifest.clone(),
        "--output-dir".to_string(),
        remote_output.clone(),
    ];
    let rendered_args = args
        .iter()
        .map(|arg| serde_json::to_string(arg))
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let spec = format!(
        "[job]\nid = {id:?}\nproject = {project:?}\ncommand = \"/usr/bin/env\"\nargs = [{rendered_args}]\nworkdir = {workdir:?}\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = {cpus}\nomp_threads = 1\nram_gb = 0\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = \"Golden42 immutable worker manifest staged\"\ncommand = \"/usr/bin/test\"\nargs = [\"-f\", {remote_manifest:?}]\ntimeout_secs = 5\n\n[preempt]\npreemptible = false\nsignal = \"SIGTERM\"\ngrace_seconds = 60\n\n[limits]\nmax_runtime_hours = {max_runtime_hours}\nmax_restarts = 0\n",
        project = SCQ_PROJECT,
        workdir = bundle_root,
        cpus = manifest.cpus,
        max_runtime_hours = manifest.max_runtime_hours,
    );
    let spec_relative = PathBuf::from("batches/scq-specs").join(format!("{id}.toml"));
    immutable_or_equal(&campaign_root.join(&spec_relative), spec.as_bytes())?;
    copy_atomic(node, &manifest_path, &remote_manifest, false)?;
    ssh_output(
        node,
        &["mkdir", "-p", &remote_output],
        "create remote batch output",
    )?;
    let submitted_at_unix_ms = unix_millis();
    let show = submit_batch_job(&manifest, &campaign_root.join(&spec_relative))?;
    let show_bytes = canonical_json(&show)?;
    let show_relative = PathBuf::from("batches/scq-show").join(format!("{id}.v1.json"));
    immutable_or_equal(&campaign_root.join(&show_relative), &show_bytes)?;
    let job = scq_job(&show, &id)?;
    let state = job
        .get("state")
        .and_then(Value::as_str)
        .context("completed SCQ job has no state")?
        .to_string();
    let exit_code = job.get("exit_code").and_then(Value::as_i64);
    let local_pull =
        std::env::temp_dir().join(format!("jope-g42-pull-{}-{id}", std::process::id()));
    if local_pull.exists() {
        fs::remove_dir_all(&local_pull)?;
    }
    let pull_result = copy_remote_outputs(node, &remote_output, &local_pull);
    let worker_path = local_pull.join("batch-worker-receipt.v1.json");
    let worker_failure_path = local_pull.join("batch-worker-failures.v1.json");
    if pull_result.is_ok() && worker_failure_path.is_file() {
        let bytes = fs::read(&worker_failure_path)?;
        let value: Value = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&value)?
            || value.get("schema").and_then(Value::as_str)
                != Some("JopeGolden42AggressiveBatchWorkerFailures.v1")
            || value.get("batch_id").and_then(Value::as_str) != Some(id.as_str())
            || value.get("node").and_then(Value::as_str) != Some(node)
        {
            bail!("SCQ batch worker failure artifact binding mismatch");
        }
        immutable_or_equal(
            &campaign_root
                .join("batches/worker-failures")
                .join(format!("{id}.v1.json")),
            &bytes,
        )?;
    }
    let (hostname, worker_receipt_sha256, started_at, worker_finished_at, mut candidates) =
        if pull_result.is_ok() && worker_path.is_file() {
            let worker_bytes = fs::read(&worker_path)?;
            let worker: ScqWorkerReceipt = serde_json::from_slice(&worker_bytes)?;
            if worker.schema != flavor.worker_receipt_schema
                || worker.benchmark_valid
                || worker.test_adaptive != test_adaptive
                || worker.batch_id != id
                || worker.node != node
                || worker.hostname != node
                || worker.manifest_sha256 != manifest_sha256
            {
                bail!("SCQ worker receipt binding mismatch");
            }
            (
                Some(worker.hostname),
                Some(sha256_bytes(&worker_bytes)),
                Some(worker.started_at_unix_ms),
                Some(worker.finished_at_unix_ms),
                worker.candidates,
            )
        } else {
            (None, None, None, None, Vec::new())
        };
    if pull_result.is_ok() && !worker_path.is_file() {
        for request in &manifest.requests {
            let file = candidate_output_name(request);
            let path = local_pull.join(&file);
            if !path.is_file() {
                continue;
            }
            let bytes = fs::read(&path)?;
            let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
            if bytes != canonical_json(&receipt)? {
                bail!("partial SCQ candidate receipt is not canonical");
            }
            candidates.push(BatchCandidateReceipt {
                test_adaptive: request.test_adaptive,
                dataset_hash: request.dataset_hash.clone(),
                generation: request.generation,
                slot: request.slot,
                attempt: request.attempt,
                semantic_sha256: request.semantic_sha256.clone(),
                file,
                sha256: sha256_bytes(&bytes),
            });
        }
    }
    let mut receipts = Vec::new();
    let mut accepted = Vec::new();
    for candidate in candidates {
        let request = manifest.requests.iter().find(|request| {
            request.dataset_hash == candidate.dataset_hash
                && request.generation == candidate.generation
                && request.slot == candidate.slot
                && request.attempt == candidate.attempt
                && request.semantic_sha256 == candidate.semantic_sha256
        });
        let Some(request) = request else {
            bail!("batch worker returned an unrequested candidate");
        };
        if candidate.test_adaptive != request.test_adaptive {
            bail!("batch worker candidate adaptation flag mismatch");
        }
        let path = local_pull.join(&candidate.file);
        let bytes = fs::read(&path)?;
        if sha256_bytes(&bytes) != candidate.sha256 {
            bail!("pulled candidate receipt hash mismatch");
        }
        let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
        if receipt.dataset_hash != request.dataset_hash
            || receipt.generation != request.generation
            || receipt.slot != request.slot
            || receipt.attempt != request.attempt
            || receipt.node != node
            || receipt.scq_job_id != id
        {
            bail!("pulled candidate receipt request mismatch");
        }
        let destination = campaign_root.join(outcome_relative(request));
        immutable_or_equal(&destination, &bytes)?;
        accepted.push(candidate);
        receipts.push(receipt);
    }
    if local_pull.exists() {
        fs::remove_dir_all(&local_pull)?;
    }
    let finished_at_unix_ms = unix_millis();
    let receipt = ScqBatchReceipt {
        schema: flavor.receipt_schema.to_string(),
        benchmark_valid: false,
        test_adaptive: manifest.test_adaptive,
        batch_id: id.clone(),
        scq_job_id: id.clone(),
        node: node.to_string(),
        hostname,
        attempt,
        manifest_file: manifest_relative.to_string_lossy().into_owned(),
        manifest_sha256,
        scq_spec_file: spec_relative.to_string_lossy().into_owned(),
        scq_spec_sha256: sha256_bytes(spec.as_bytes()),
        worker_receipt_sha256,
        submitted_at_unix_ms,
        started_at_unix_ms: Some(started_at.unwrap_or(submitted_at_unix_ms)),
        worker_finished_at_unix_ms: Some(worker_finished_at.unwrap_or(finished_at_unix_ms)),
        finished_at_unix_ms,
        scq_state: state,
        exit_code,
        scq_show_file: show_relative.to_string_lossy().into_owned(),
        scq_show_sha256: sha256_bytes(&show_bytes),
        candidates: accepted,
    };
    immutable_or_equal(&receipt_path, &canonical_json(&receipt)?)?;
    Ok((receipt, receipts))
}

fn candidate_key(receipt: &CandidateReceipt) -> (String, usize, usize) {
    (
        receipt.dataset_hash.clone(),
        receipt.generation,
        receipt.slot,
    )
}

pub(super) fn evaluate_batches(
    campaign_root: &Path,
    identity: &CampaignIdentity,
    registry: &Path,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    dataset_index: usize,
    purpose: &str,
    generation: usize,
    members: &[(usize, Member)],
    test: &RegressionData,
) -> Result<Vec<CachedEvaluation>> {
    let mut requests = members
        .iter()
        .map(|(slot, member)| {
            make_request(
                identity, registry, purpose, generation, *slot, 0, member, NODES[0],
            )
        })
        .collect::<Vec<_>>();
    requests.sort_by_key(|request| request.slot);
    let mut all_receipts = Vec::new();
    for (batch_ordinal, chunk) in requests.chunks(SCQ_CPUS).enumerate() {
        let primary = primary_node(dataset_index, generation, batch_ordinal);
        let (_, mut receipts) = run_batch(
            campaign_root,
            release_binary_sha256,
            scq_binary_sha256,
            NODES[primary],
            0,
            chunk.to_vec(),
        )?;
        let completed = receipts.iter().map(candidate_key).collect::<HashSet<_>>();
        let missing = chunk
            .iter()
            .filter(|request| {
                !completed.contains(&(
                    request.dataset_hash.clone(),
                    request.generation,
                    request.slot,
                ))
            })
            .cloned()
            .collect::<Vec<_>>();
        all_receipts.append(&mut receipts);
        if !missing.is_empty() {
            let (_, mut retry_receipts) = run_batch(
                campaign_root,
                release_binary_sha256,
                scq_binary_sha256,
                NODES[retry_node(primary)],
                1,
                missing,
            )?;
            all_receipts.append(&mut retry_receipts);
        }
    }
    let by_key = all_receipts
        .into_iter()
        .map(|receipt| (candidate_key(&receipt), receipt))
        .collect::<HashMap<_, _>>();
    members
        .iter()
        .map(|(slot, _)| {
            let key = (identity.dataset_hash.clone(), generation, *slot);
            let receipt = by_key
                .get(&key)
                .with_context(|| format!("candidate {key:?} missing after deterministic retry"))?
                .clone();
            let request_path = campaign_root.join(request_relative_from_receipt(&receipt));
            let request: CandidateRequest = serde_json::from_slice(&fs::read(&request_path)?)?;
            validate_receipt_against_request(&receipt, &request, test)?;
            let relative = outcome_relative_from_receipt(&receipt);
            let bytes = fs::read(campaign_root.join(&relative))?;
            Ok(CachedEvaluation {
                source: CacheSource {
                    generation,
                    slot: *slot,
                },
                receipt_file: relative
                    .strip_prefix(PathBuf::from("datasets").join(&identity.dataset_hash))
                    .unwrap_or(&relative)
                    .to_string_lossy()
                    .into_owned(),
                receipt_sha256: sha256_bytes(&bytes),
                receipt,
            })
        })
        .collect()
}

/// Evaluate fresh members from up to 42 independent 36-slot dataset generations
/// as one wave. Full generations contribute twelve requests to each node; cached
/// generations may contribute fewer. A fixed pool of 120 pinned workers drains
/// each over-decomposed node queue.
pub(super) fn evaluate_aggressive_wave(
    campaign_root: &Path,
    registry: &Path,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    lanes: &[(
        &CampaignIdentity,
        usize,
        &[(usize, Member)],
        &RegressionData,
    )],
) -> Result<Vec<Vec<CachedEvaluation>>> {
    if lanes.is_empty() || lanes.len() > AGGRESSIVE_DATASET_LANES {
        bail!("aggressive wave must contain 1..={AGGRESSIVE_DATASET_LANES} datasets");
    }
    let run_id = &lanes[0].0.run_id;
    if lanes
        .iter()
        .any(|(identity, _, _, _)| identity.run_id != *run_id)
    {
        bail!("aggressive wave cannot mix run identities");
    }
    let flavor = aggressive_flavor(run_id);
    let maximum_retry_attempt = if is_golden46_run(run_id) { 1 } else { 3 };
    let mut groups = vec![Vec::<CandidateRequest>::new(); NODES.len()];
    for (identity, generation, members, _) in lanes {
        if members.is_empty() || members.len() > WAVE_MAX_LANE_POPULATION {
            bail!(
                "aggressive wave lane must contain 1..={WAVE_MAX_LANE_POPULATION} fresh candidates"
            );
        }
        let mut routed = [0usize; 3];
        for (slot, member) in *members {
            if *slot >= AGGRESSIVE_DATASET_POPULATION {
                bail!("aggressive logical slot {slot} is out of range");
            }
            let node_index = (slot + *generation) % NODES.len();
            let mut request = make_request(
                identity,
                registry,
                "aggressive-candidate",
                *generation,
                *slot,
                0,
                member,
                NODES[node_index],
            );
            request.test_adaptive = Some(true);
            groups[node_index].push(request);
            routed[node_index] += 1;
        }
        if routed
            .iter()
            .any(|count| *count > WAVE_MAX_CANDIDATES_PER_NODE)
        {
            bail!("wave lane routing exceeds twelve candidates on one node");
        }
    }
    for group in &mut groups {
        group.sort_by(|left, right| {
            left.dataset_hash
                .cmp(&right.dataset_hash)
                .then_with(|| left.generation.cmp(&right.generation))
                .then_with(|| left.slot.cmp(&right.slot))
        });
        let maximum = lanes.len() * WAVE_MAX_CANDIDATES_PER_NODE;
        if group.len() > maximum {
            bail!("aggressive wave routing exceeds {maximum} requests per node");
        }
    }

    let primary_results = thread::scope(|scope| -> Result<Vec<_>> {
        let handles = groups
            .iter()
            .enumerate()
            .filter(|(_, requests)| !requests.is_empty())
            .map(|(node_index, requests)| {
                scope.spawn(move || {
                    (
                        node_index,
                        run_batch_flavored(
                            flavor,
                            campaign_root,
                            release_binary_sha256,
                            scq_binary_sha256,
                            NODES[node_index],
                            0,
                            requests.clone(),
                        ),
                    )
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("aggressive node controller panicked"))
            })
            .collect::<Result<Vec<_>>>()
    })?;

    let mut all_receipts = Vec::new();
    for (node_index, primary_result) in primary_results {
        let mut receipts = match primary_result {
            Ok((_, receipts)) => receipts,
            Err(error) => {
                eprintln!(
                    "aggressive primary batch on {} failed before a complete receipt: {error:#}",
                    NODES[node_index]
                );
                Vec::new()
            }
        };
        let completed = receipts.iter().map(candidate_key).collect::<HashSet<_>>();
        let mut missing = groups[node_index]
            .iter()
            .filter(|request| {
                !completed.contains(&(
                    request.dataset_hash.clone(),
                    request.generation,
                    request.slot,
                ))
            })
            .cloned()
            .collect::<Vec<_>>();
        all_receipts.append(&mut receipts);
        for attempt in 1..=maximum_retry_attempt {
            if missing.is_empty() {
                break;
            }
            let retry_node_index = (node_index + attempt) % NODES.len();
            let mut retried = match run_batch_flavored(
                flavor,
                campaign_root,
                release_binary_sha256,
                scq_binary_sha256,
                NODES[retry_node_index],
                attempt,
                missing.clone(),
            ) {
                Ok((_, receipts)) => receipts,
                Err(error) => {
                    eprintln!(
                        "aggressive retry {attempt} on {} failed before a complete receipt: {error:#}",
                        NODES[retry_node_index]
                    );
                    Vec::new()
                }
            };
            let completed = retried.iter().map(candidate_key).collect::<HashSet<_>>();
            missing.retain(|request| {
                !completed.contains(&(
                    request.dataset_hash.clone(),
                    request.generation,
                    request.slot,
                ))
            });
            all_receipts.append(&mut retried);
        }
    }

    let by_key = all_receipts
        .into_iter()
        .map(|receipt| (candidate_key(&receipt), receipt))
        .collect::<HashMap<_, _>>();
    lanes
        .iter()
        .map(|(identity, generation, members, test)| {
            let mut evaluations = members
                .iter()
                .map(|(slot, _)| {
                    let key = (identity.dataset_hash.clone(), *generation, *slot);
                    let receipt = by_key
                        .get(&key)
                        .with_context(|| {
                            format!("candidate {key:?} missing after three deterministic retries")
                        })?
                        .clone();
                    let request_path = campaign_root.join(request_relative_from_receipt(&receipt));
                    let request: CandidateRequest =
                        serde_json::from_slice(&fs::read(&request_path)?)?;
                    validate_receipt_against_request(&receipt, &request, test)?;
                    let relative = outcome_relative_from_receipt(&receipt);
                    let bytes = fs::read(campaign_root.join(&relative))?;
                    Ok(CachedEvaluation {
                        source: CacheSource {
                            generation: *generation,
                            slot: *slot,
                        },
                        receipt_file: relative
                            .strip_prefix(PathBuf::from("datasets").join(&identity.dataset_hash))
                            .unwrap_or(&relative)
                            .to_string_lossy()
                            .into_owned(),
                        receipt_sha256: sha256_bytes(&bytes),
                        receipt,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            evaluations.sort_by_key(|evaluation| evaluation.source.slot);
            Ok(evaluations)
        })
        .collect()
}

pub(super) fn evaluate_aggressive_batches(
    campaign_root: &Path,
    identity: &CampaignIdentity,
    registry: &Path,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    generation: usize,
    members: &[(usize, Member)],
    test: &RegressionData,
) -> Result<Vec<CachedEvaluation>> {
    evaluate_aggressive_wave(
        campaign_root,
        registry,
        release_binary_sha256,
        scq_binary_sha256,
        &[(identity, generation, members, test)],
    )?
    .pop()
    .context("single-dataset aggressive wave returned no evaluations")
}

pub(super) fn replay_across_nodes(
    campaign_root: &Path,
    identity: &CampaignIdentity,
    registry: &Path,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    generation: usize,
    champion: &Member,
    test: &RegressionData,
) -> Result<Vec<CachedEvaluation>> {
    let mut evaluations = Vec::with_capacity(NODES.len());
    for (slot, node) in NODES.iter().enumerate() {
        let request = make_request(
            identity,
            registry,
            "replay-final",
            generation,
            slot,
            0,
            champion,
            node,
        );
        let (_, receipts) = run_batch(
            campaign_root,
            release_binary_sha256,
            scq_binary_sha256,
            node,
            0,
            vec![request],
        )?;
        let receipt = receipts
            .into_iter()
            .next()
            .with_context(|| format!("cross-node replay missing on {node}"))?;
        validate_receipt_against_request(
            &receipt,
            &serde_json::from_slice(&fs::read(
                campaign_root.join(request_relative_from_receipt(&receipt)),
            )?)?,
            test,
        )?;
        let relative = outcome_relative_from_receipt(&receipt);
        let bytes = fs::read(campaign_root.join(&relative))?;
        evaluations.push(CachedEvaluation {
            source: CacheSource { generation, slot },
            receipt_file: relative
                .strip_prefix(PathBuf::from("datasets").join(&identity.dataset_hash))
                .unwrap_or(&relative)
                .to_string_lossy()
                .into_owned(),
            receipt_sha256: sha256_bytes(&bytes),
            receipt,
        });
    }
    Ok(evaluations)
}

pub(super) fn replay_aggressive_across_nodes(
    campaign_root: &Path,
    identity: &CampaignIdentity,
    registry: &Path,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    generation: usize,
    champion: &Member,
    test: &RegressionData,
) -> Result<Vec<CachedEvaluation>> {
    let flavor = aggressive_flavor(&identity.run_id);
    let mut evaluations = Vec::with_capacity(NODES.len());
    for (slot, node) in NODES.iter().enumerate() {
        let mut selected = None;
        for attempt in 0..=1 {
            let mut request = make_request(
                identity,
                registry,
                "aggressive-confirmation",
                generation,
                slot,
                attempt,
                champion,
                node,
            );
            request.test_adaptive = Some(true);
            let (_, receipts) = run_batch_flavored(
                flavor,
                campaign_root,
                release_binary_sha256,
                scq_binary_sha256,
                node,
                attempt,
                vec![request],
            )?;
            if let Some(receipt) = receipts.into_iter().next() {
                selected = Some(receipt);
                break;
            }
        }
        let receipt = selected
            .with_context(|| format!("aggressive confirmation missing on {node} after retry"))?;
        validate_receipt_against_request(
            &receipt,
            &serde_json::from_slice(&fs::read(
                campaign_root.join(request_relative_from_receipt(&receipt)),
            )?)?,
            test,
        )?;
        let relative = outcome_relative_from_receipt(&receipt);
        let bytes = fs::read(campaign_root.join(&relative))?;
        evaluations.push(CachedEvaluation {
            source: CacheSource { generation, slot },
            receipt_file: relative
                .strip_prefix(PathBuf::from("datasets").join(&identity.dataset_hash))
                .unwrap_or(&relative)
                .to_string_lossy()
                .into_owned(),
            receipt_sha256: sha256_bytes(&bytes),
            receipt,
        });
    }
    Ok(evaluations)
}

fn request_relative_from_receipt(receipt: &CandidateReceipt) -> PathBuf {
    PathBuf::from("datasets")
        .join(&receipt.dataset_hash)
        .join("requests")
        .join(&receipt.purpose)
        .join(format!("generation-{:03}", receipt.generation))
        .join(format!(
            "slot-{:03}-attempt-{}.v1.json",
            receipt.slot, receipt.attempt
        ))
}

fn outcome_relative_from_receipt(receipt: &CandidateReceipt) -> PathBuf {
    PathBuf::from("datasets")
        .join(&receipt.dataset_hash)
        .join(receipt_relative(receipt))
}

fn validate_batch_manifest(manifest: &ScqBatchManifest, flavor: BatchFlavor) -> Result<()> {
    let hostname = String::from_utf8(
        command_output(
            {
                let mut command = Command::new("hostname");
                command.arg("-s");
                command
            },
            "hostname",
        )?
        .stdout,
    )?
    .trim()
    .to_string();
    let maximum_requests = maximum_batch_requests(flavor);
    if manifest.schema != flavor.manifest_schema
        || manifest.benchmark_valid
        || manifest.test_adaptive
            != (flavor.manifest_schema == AGGRESSIVE_BATCH_SCHEMA).then_some(true)
        || manifest.node != hostname
        || !NODES.contains(&manifest.node.as_str())
        || manifest.project != SCQ_PROJECT
        || manifest.cpus != reserved_batch_cpus(flavor, manifest.requests.len())?
        || manifest.candidate_threads != 1
        || manifest.candidate_timeout_seconds != flavor.candidate_timeout_seconds
        || manifest.max_runtime_hours != flavor.max_runtime_hours
        || manifest.preemptible
        || manifest.requests.is_empty()
        || manifest.requests.len() > maximum_requests
        || current_binary_sha256()? != manifest.release_binary_sha256
    {
        bail!("SCQ batch manifest identity or resource limit mismatch");
    }
    let mut keys = HashSet::new();
    for request in &manifest.requests {
        if request.run_id != manifest.run_id
            || request.node != manifest.node
            || request.execution_host != "scq"
            || request.scq_job_id != manifest.batch_id
            || request.attempt != manifest.attempt
            || request.binary_sha256 != manifest.release_binary_sha256
            || request.thread_counts != ThreadCounts::single()
            || !keys.insert((
                request.dataset_hash.clone(),
                request.generation,
                request.slot,
            ))
        {
            bail!("invalid or duplicate request in SCQ batch");
        }
    }
    Ok(())
}

fn validate_request_contract(request: &CandidateRequest) -> Result<()> {
    if request.schema != REQUEST_SCHEMA
        || request.benchmark_valid
        || (request.test_adaptive.is_some() && request.test_adaptive != Some(true))
        || request.thread_counts != ThreadCounts::single()
    {
        bail!("candidate request identity mismatch");
    }
    Ok(())
}

pub(super) fn evaluate_request(request: &CandidateRequest) -> Result<CandidateReceipt> {
    validate_request_contract(request)?;
    if current_binary_sha256()? != request.binary_sha256 {
        bail!("candidate request binary mismatch");
    }
    let hostname = String::from_utf8(
        command_output(
            {
                let mut command = Command::new("hostname");
                command.arg("-s");
                command
            },
            "hostname",
        )?
        .stdout,
    )?
    .trim()
    .to_string();
    match request.execution_host.as_str() {
        "scq" if hostname == request.node && NODES.contains(&request.node.as_str()) => {}
        EXECUTION_HOST if request.node == EXECUTION_NODE => {}
        _ => bail!("candidate hostname does not match its execution binding"),
    }
    let root = PathBuf::from(&request.dataset_root);
    let train_path = root.join("train.csv");
    let test_path = root.join("test.csv");
    if sha256_file(&train_path)? != request.train_sha256
        || sha256_file(&test_path)? != request.test_sha256
    {
        bail!("worker dataset hash mismatch");
    }
    let train = read_regression_csv(&train_path)?;
    let test = read_regression_csv(&test_path)?;
    if train.x.len() != request.train_rows
        || test.x.len() != request.test_rows
        || train.features != request.features
        || test.features != request.features
    {
        bail!("worker dataset shape mismatch");
    }
    evaluate_request_with_data(request, &train, &test)
}

fn evaluate_request_with_data(
    request: &CandidateRequest,
    train: &RegressionData,
    test: &RegressionData,
) -> Result<CandidateReceipt> {
    validate_request_contract(request)?;
    if train.x.len() != request.train_rows
        || test.x.len() != request.test_rows
        || train.features != request.features
        || test.features != request.features
    {
        bail!("loaded worker dataset shape mismatch");
    }
    let genome = GenomeIr::parse(&request.canonical_genome)?;
    let started = Instant::now();
    let result = feat_math::model::fit_predict(&genome, &train.x, &train.y, &test.x);
    let (predictions, metrics, failure) = match result {
        Ok(predictions) => match metrics(&test.y, &predictions) {
            Ok(metrics) => (predictions, Some(metrics), None),
            Err(error) => (
                Vec::new(),
                None,
                Some(FailureState {
                    class: "nonfinite".to_string(),
                    message: error.to_string(),
                    retryable: false,
                }),
            ),
        },
        Err(error) => (
            Vec::new(),
            None,
            Some(FailureState {
                class: "model".to_string(),
                message: error.to_string(),
                retryable: false,
            }),
        ),
    };
    let predictions_sha256 = sha256_bytes(&prediction_bytes(&predictions));
    Ok(CandidateReceipt {
        schema: RECEIPT_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: request.test_adaptive,
        purpose: request.purpose.clone(),
        run_id: request.run_id.clone(),
        generation: request.generation,
        slot: request.slot,
        attempt: request.attempt,
        semantic_sha256: request.semantic_sha256.clone(),
        canonical_genome: request.canonical_genome.clone(),
        binary_sha256: request.binary_sha256.clone(),
        dataset_hash: request.dataset_hash.clone(),
        train_sha256: request.train_sha256.clone(),
        test_sha256: request.test_sha256.clone(),
        train_rows: request.train_rows,
        test_rows: request.test_rows,
        features: request.features,
        execution_host: request.execution_host.clone(),
        node: request.node.clone(),
        scq_job_id: request.scq_job_id.clone(),
        runtime_seconds: started.elapsed().as_secs_f64(),
        thread_counts: ThreadCounts::single(),
        predictions,
        predictions_sha256,
        metrics,
        failure,
    })
}

fn parse_cpu_list(value: &str) -> Result<Vec<usize>> {
    let mut cpus = BTreeSet::new();
    for part in value.trim().split(',') {
        if part.is_empty() {
            bail!("empty CPU-list component");
        }
        let (start, end) = if let Some((start, end)) = part.split_once('-') {
            if end.contains('-') {
                bail!("invalid CPU range {part}");
            }
            let start = start.parse::<usize>().context("parse CPU range start")?;
            let end = end.parse::<usize>().context("parse CPU range end")?;
            (start, end)
        } else {
            let cpu = part.parse::<usize>().context("parse CPU")?;
            (cpu, cpu)
        };
        if start > end || end >= libc::CPU_SETSIZE as usize {
            bail!("invalid or unsupported CPU range {part}");
        }
        for cpu in start..=end {
            if !cpus.insert(cpu) {
                bail!("duplicate CPU {cpu} in allowed list");
            }
        }
    }
    if cpus.is_empty() {
        bail!("job allowed CPU list is empty");
    }
    Ok(cpus.into_iter().collect())
}

pub(super) fn job_allowed_cpus() -> Result<Vec<usize>> {
    let status = fs::read_to_string("/proc/self/status").context("read process CPU allowance")?;
    let encoded = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .context("process status has no Cpus_allowed_list")?;
    let parsed = parse_cpu_list(encoded)?;
    let mut kernel_set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    if unsafe {
        libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut kernel_set)
    } != 0
    {
        return Err(std::io::Error::last_os_error()).context("read job CPU affinity");
    }
    let kernel = (0..libc::CPU_SETSIZE as usize)
        .filter(|cpu| unsafe { libc::CPU_ISSET(*cpu, &kernel_set) })
        .collect::<Vec<_>>();
    if parsed != kernel {
        bail!("Cpus_allowed_list disagrees with sched_getaffinity");
    }
    Ok(parsed)
}

fn select_worker_cpus(allowed: &[usize], workers: usize) -> Result<Vec<usize>> {
    if allowed.len() < workers {
        bail!(
            "SCQ job allows {} CPUs but the batch requires {workers}",
            allowed.len()
        );
    }
    Ok(allowed[..workers].to_vec())
}

fn select_aggressive_worker_cpus(allowed: &[usize], workers: usize) -> Result<Vec<usize>> {
    if allowed.is_empty()
        || allowed.len() > AGGRESSIVE_CPUS
        || workers > AGGRESSIVE_WORKERS
        || workers > allowed.len() * 2
    {
        bail!(
            "aggressive worker pool requires 1..={AGGRESSIVE_CPUS} allowed CPUs and at most two workers per CPU"
        );
    }
    Ok((0..workers)
        .map(|worker| allowed[worker % allowed.len()])
        .collect())
}

fn constrain_job_cpus(allowed: &[usize], reserved: usize) -> Result<Vec<usize>> {
    if reserved == 0 || reserved > AGGRESSIVE_CPUS || allowed.len() < reserved {
        bail!(
            "SCQ job exposes {} CPUs but the batch reserves {reserved}",
            allowed.len()
        );
    }
    Ok(allowed[..reserved].to_vec())
}

pub(super) fn pin_current_thread(cpu: usize) -> Result<()> {
    if cpu >= libc::CPU_SETSIZE as usize {
        bail!("CPU {cpu} exceeds cpu_set_t capacity");
    }
    let mut requested: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::CPU_ZERO(&mut requested);
        libc::CPU_SET(cpu, &mut requested);
    }
    if unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &requested) }
        != 0
    {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("pin candidate thread to CPU {cpu}"));
    }
    let mut actual: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut actual) }
        != 0
    {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("verify candidate thread affinity on CPU {cpu}"));
    }
    let pinned = (0..libc::CPU_SETSIZE as usize)
        .filter(|candidate| unsafe { libc::CPU_ISSET(*candidate, &actual) })
        .collect::<Vec<_>>();
    if pinned != [cpu] {
        bail!("candidate thread affinity mismatch: expected CPU {cpu}, got {pinned:?}");
    }
    Ok(())
}

fn validate_aggressive_request_count(count: usize) -> Result<usize> {
    if count == 0 || count > AGGRESSIVE_MAX_REQUESTS_PER_NODE {
        bail!("aggressive SCQ batch requires 1..={AGGRESSIVE_MAX_REQUESTS_PER_NODE} candidates");
    }
    Ok(count.min(AGGRESSIVE_WORKERS))
}

fn load_aggressive_batch_datasets(
    manifest: &ScqBatchManifest,
) -> Result<HashMap<String, (RegressionData, RegressionData)>> {
    validate_aggressive_request_count(manifest.requests.len())?;
    let mut bindings = BTreeMap::<String, &CandidateRequest>::new();
    for request in &manifest.requests {
        validate_request_contract(request)?;
        if request.test_adaptive != Some(true) {
            bail!("aggressive SCQ request is not test-adaptive");
        }
        if let Some(first) = bindings.get(&request.dataset_hash) {
            if request.dataset_root != first.dataset_root
                || request.train_sha256 != first.train_sha256
                || request.test_sha256 != first.test_sha256
                || request.train_rows != first.train_rows
                || request.test_rows != first.test_rows
                || request.features != first.features
            {
                bail!("aggressive SCQ requests disagree on one dataset snapshot");
            }
        } else {
            bindings.insert(request.dataset_hash.clone(), request);
        }
    }
    let mut datasets = HashMap::with_capacity(bindings.len());
    for (dataset_hash, binding) in bindings {
        let root = PathBuf::from(&binding.dataset_root);
        let train_path = root.join("train.csv");
        let test_path = root.join("test.csv");
        let train_bytes = fs::read(&train_path)?;
        let test_bytes = fs::read(&test_path)?;
        if sha256_bytes(&train_bytes) != binding.train_sha256
            || sha256_bytes(&test_bytes) != binding.test_sha256
        {
            bail!("aggressive worker dataset hash mismatch");
        }
        let train = crate::data::read_regression_csv_bytes(
            &train_bytes,
            train_path.to_string_lossy().as_ref(),
        )?;
        let test = crate::data::read_regression_csv_bytes(
            &test_bytes,
            test_path.to_string_lossy().as_ref(),
        )?;
        if train.x.len() != binding.train_rows
            || test.x.len() != binding.test_rows
            || train.features != binding.features
            || test.features != binding.features
        {
            bail!("aggressive worker dataset shape mismatch");
        }
        datasets.insert(dataset_hash, (train, test));
    }
    Ok(datasets)
}

fn batch_worker(flavor: BatchFlavor, manifest_path: &Path, output_dir: &Path) -> Result<()> {
    let manifest_bytes = fs::read(manifest_path)?;
    let manifest: ScqBatchManifest = serde_json::from_slice(&manifest_bytes)?;
    validate_batch_manifest(&manifest, flavor)?;
    fs::create_dir_all(output_dir)?;
    let started_at_unix_ms = unix_millis();
    let candidates = Arc::new(Mutex::new(Vec::new()));
    thread::scope(|scope| {
        for request in &manifest.requests {
            let candidates = Arc::clone(&candidates);
            scope.spawn(move || {
                let request_dir = output_dir.join("requests");
                if fs::create_dir_all(&request_dir).is_err() {
                    return;
                }
                let request_path = request_dir.join(candidate_output_name(request));
                let Ok(request_bytes) = canonical_json(request) else {
                    return;
                };
                if immutable_write(&request_path, &request_bytes).is_err() {
                    return;
                }
                let Ok(binary) = std::env::current_exe() else {
                    return;
                };
                let mut command = Command::new("timeout");
                command
                    .arg(manifest.candidate_timeout_seconds.to_string())
                    .arg(binary)
                    .arg("test-adaptive-worker")
                    .arg("--request")
                    .arg(&request_path);
                let Ok(output) = command.output() else {
                    return;
                };
                if !output.status.success() {
                    return;
                }
                let Ok(receipt) = parse_worker_receipt(&output.stdout) else {
                    return;
                };
                let Ok(bytes) = canonical_json(&receipt) else {
                    return;
                };
                let file = candidate_output_name(request);
                let path = output_dir.join(&file);
                if immutable_write(&path, &bytes).is_err() {
                    return;
                }
                candidates.lock().unwrap().push(BatchCandidateReceipt {
                    test_adaptive: request.test_adaptive,
                    dataset_hash: request.dataset_hash.clone(),
                    generation: request.generation,
                    slot: request.slot,
                    attempt: request.attempt,
                    semantic_sha256: request.semantic_sha256.clone(),
                    file,
                    sha256: sha256_bytes(&bytes),
                });
            });
        }
    });
    let mut candidates = Arc::into_inner(candidates).unwrap().into_inner().unwrap();
    candidates.sort_by_key(|candidate| {
        (
            candidate.dataset_hash.clone(),
            candidate.generation,
            candidate.slot,
        )
    });
    let receipt = ScqWorkerReceipt {
        schema: flavor.worker_receipt_schema.to_string(),
        benchmark_valid: false,
        test_adaptive: manifest.test_adaptive,
        batch_id: manifest.batch_id.clone(),
        node: manifest.node.clone(),
        hostname: manifest.node.clone(),
        manifest_sha256: sha256_bytes(&manifest_bytes),
        started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
        candidates,
    };
    immutable_write(
        &output_dir.join("batch-worker-receipt.v1.json"),
        &canonical_json(&receipt)?,
    )?;
    Ok(())
}

fn batch_watchdog_expired(control: std::sync::mpsc::Receiver<()>, timeout: Duration) -> bool {
    if control.recv().is_err() {
        return true;
    }
    !matches!(control.recv_timeout(timeout), Ok(()))
}

fn timeout_candidate_receipt(
    request: &CandidateRequest,
    runtime_seconds: f64,
    timeout_seconds: usize,
) -> CandidateReceipt {
    let predictions = Vec::new();
    CandidateReceipt {
        schema: RECEIPT_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: request.test_adaptive,
        purpose: request.purpose.clone(),
        run_id: request.run_id.clone(),
        generation: request.generation,
        slot: request.slot,
        attempt: request.attempt,
        semantic_sha256: request.semantic_sha256.clone(),
        canonical_genome: request.canonical_genome.clone(),
        binary_sha256: request.binary_sha256.clone(),
        dataset_hash: request.dataset_hash.clone(),
        train_sha256: request.train_sha256.clone(),
        test_sha256: request.test_sha256.clone(),
        train_rows: request.train_rows,
        test_rows: request.test_rows,
        features: request.features,
        execution_host: request.execution_host.clone(),
        node: request.node.clone(),
        scq_job_id: request.scq_job_id.clone(),
        runtime_seconds,
        thread_counts: ThreadCounts::single(),
        predictions_sha256: sha256_bytes(&prediction_bytes(&predictions)),
        predictions,
        metrics: None,
        failure: Some(FailureState {
            class: "timeout".to_string(),
            message: format!(
                "candidate exceeded the immutable {} second wall-time budget",
                timeout_seconds
            ),
            retryable: false,
        }),
    }
}

fn evaluate_aggressive_request_process_once(
    request: &CandidateRequest,
    request_path: &Path,
    timeout_seconds: usize,
) -> Result<CandidateReceipt> {
    let binary = std::env::current_exe()?;
    let started = Instant::now();
    let mut command = if timeout_seconds == 0 {
        Command::new(&binary)
    } else {
        let mut command = Command::new("timeout");
        command
            .arg("--kill-after=1")
            .arg(timeout_seconds.to_string())
            .arg(&binary);
        command
    };
    let output = command
        .arg("test-adaptive-worker")
        .arg("--request")
        .arg(request_path)
        .output()
        .context("spawn aggressive candidate process")?;
    if output.status.success() {
        return parse_worker_receipt(&output.stdout);
    }
    if timeout_seconds != 0 && output.status.code() == Some(124) {
        return Ok(timeout_candidate_receipt(
            request,
            started.elapsed().as_secs_f64(),
            timeout_seconds,
        ));
    }
    bail!(
        "aggressive candidate process exited {:?}: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr).trim()
    )
}

fn evaluate_aggressive_request_process(
    request: &CandidateRequest,
    request_path: &Path,
    timeout_seconds: usize,
) -> Result<CandidateReceipt> {
    let mut failures = Vec::new();
    for local_attempt in 1..=3 {
        match evaluate_aggressive_request_process_once(request, request_path, timeout_seconds) {
            Ok(receipt) => return Ok(receipt),
            Err(error) => failures.push(format!("local attempt {local_attempt}: {error:#}")),
        }
    }
    bail!(
        "aggressive candidate failed three local process attempts: {}",
        failures.join("; ")
    )
}

fn aggressive_batch_worker_direct(manifest_path: &Path, output_dir: &Path) -> Result<()> {
    let manifest_bytes = fs::read(manifest_path)?;
    let manifest: ScqBatchManifest = serde_json::from_slice(&manifest_bytes)?;
    let flavor = aggressive_flavor(&manifest.run_id);
    validate_batch_manifest(&manifest, flavor)?;
    let _datasets = load_aggressive_batch_datasets(&manifest)?;
    let worker_count = validate_aggressive_request_count(manifest.requests.len())?;
    let allowed_cpus = job_allowed_cpus()?;
    // The SCQ agent on xbabe2 can transiently expose the host-wide affinity
    // set even though it recorded a smaller reservation. Fail closed when the
    // reservation is unavailable, but pin workers to exactly the reserved
    // prefix when the kernel exposes a harmless superset.
    let reserved_cpus = constrain_job_cpus(&allowed_cpus, manifest.cpus)?;
    let cpus = select_aggressive_worker_cpus(&reserved_cpus, worker_count)?;

    fs::create_dir_all(output_dir)?;
    let request_dir = output_dir.join("requests");
    fs::create_dir_all(&request_dir)?;
    for request in &manifest.requests {
        immutable_write(
            &request_dir.join(candidate_output_name(request)),
            &canonical_json(request)?,
        )?;
    }

    let started_at_unix_ms = unix_millis();
    let start_barrier = Arc::new(Barrier::new(worker_count + 1));
    let pin_failed = Arc::new(AtomicBool::new(false));
    let next_request = Arc::new(AtomicUsize::new(0));
    let candidates = Arc::new(Mutex::new(Vec::<BatchCandidateReceipt>::new()));
    let failures = Arc::new(Mutex::new(Vec::<String>::new()));
    let watchdog = if manifest.candidate_timeout_seconds == 0 {
        None
    } else {
        let (watchdog_control, watchdog_receiver) = std::sync::mpsc::sync_channel(2);
        let queue_waves = manifest.requests.len().div_ceil(worker_count);
        let watchdog_timeout = Duration::from_secs(
            manifest.candidate_timeout_seconds as u64 * queue_waves as u64 + 60,
        );
        let watchdog = thread::Builder::new()
            .name("jope-aggressive-watchdog".to_string())
            .spawn(move || {
                if batch_watchdog_expired(watchdog_receiver, watchdog_timeout) {
                    std::process::exit(124);
                }
            })
            .context("spawn aggressive batch watchdog")?;
        watchdog_control
            .send(())
            .context("start aggressive batch watchdog")?;
        Some((watchdog_control, watchdog))
    };
    thread::scope(|scope| -> Result<()> {
        for (worker, cpu) in cpus.iter().enumerate() {
            let start_barrier = Arc::clone(&start_barrier);
            let pin_failed = Arc::clone(&pin_failed);
            let next_request = Arc::clone(&next_request);
            let candidates = Arc::clone(&candidates);
            let failures = Arc::clone(&failures);
            let requests = &manifest.requests;
            let request_dir = &request_dir;
            scope.spawn(move || {
                let pin_result = pin_current_thread(*cpu);
                if pin_result.is_err() {
                    pin_failed.store(true, AtomicOrdering::SeqCst);
                }
                start_barrier.wait();
                if let Err(error) = pin_result {
                    failures
                        .lock()
                        .unwrap()
                        .push(format!("worker {worker} CPU pinning: {error:#}"));
                    return;
                }
                if pin_failed.load(AtomicOrdering::SeqCst) {
                    failures
                        .lock()
                        .unwrap()
                        .push(format!("worker {worker}: peer CPU pinning failed"));
                    return;
                }
                loop {
                    let index = next_request.fetch_add(1, AtomicOrdering::Relaxed);
                    let Some(request) = requests.get(index) else {
                        break;
                    };
                    let result = (|| -> Result<BatchCandidateReceipt> {
                        let request_path = request_dir.join(candidate_output_name(request));
                        let receipt = evaluate_aggressive_request_process(
                            request,
                            &request_path,
                            manifest.candidate_timeout_seconds,
                        )?;
                        let bytes = canonical_json(&receipt)?;
                        let file = candidate_output_name(request);
                        immutable_write(&output_dir.join(&file), &bytes)?;
                        Ok(BatchCandidateReceipt {
                            test_adaptive: request.test_adaptive,
                            dataset_hash: request.dataset_hash.clone(),
                            generation: request.generation,
                            slot: request.slot,
                            attempt: request.attempt,
                            semantic_sha256: request.semantic_sha256.clone(),
                            file,
                            sha256: sha256_bytes(&bytes),
                        })
                    })();
                    match result {
                        Ok(candidate) => candidates.lock().unwrap().push(candidate),
                        Err(error) => failures.lock().unwrap().push(format!(
                            "{} generation {} slot {}: {error:#}",
                            request.dataset_hash, request.generation, request.slot
                        )),
                    }
                }
            });
        }
        start_barrier.wait();
        Ok(())
    })?;
    if let Some((watchdog_control, watchdog)) = watchdog {
        watchdog_control
            .send(())
            .context("cancel aggressive batch watchdog")?;
        watchdog
            .join()
            .map_err(|_| anyhow::anyhow!("aggressive batch watchdog panicked"))?;
    }

    let failures = Arc::into_inner(failures)
        .context("aggressive worker failure references remain")?
        .into_inner()
        .map_err(|_| anyhow::anyhow!("aggressive worker failure mutex poisoned"))?;
    if !failures.is_empty() {
        let failure_artifact = serde_json::json!({
            "schema": "JopeGolden42AggressiveBatchWorkerFailures.v1",
            "batch_id": manifest.batch_id.clone(),
            "node": manifest.node.clone(),
            "failures": failures.clone(),
        });
        immutable_write(
            &output_dir.join("batch-worker-failures.v1.json"),
            &canonical_json(&failure_artifact)?,
        )?;
        bail!("aggressive worker failed: {}", failures.join("; "));
    }

    let mut candidates = Arc::into_inner(candidates)
        .context("aggressive worker candidate references remain")?
        .into_inner()
        .map_err(|_| anyhow::anyhow!("aggressive worker candidate mutex poisoned"))?;
    if candidates.len() != manifest.requests.len() {
        bail!(
            "aggressive worker returned {} results for {} candidates",
            candidates.len(),
            manifest.requests.len(),
        );
    }
    candidates.sort_by_key(|candidate| {
        (
            candidate.dataset_hash.clone(),
            candidate.generation,
            candidate.slot,
        )
    });
    let receipt = ScqWorkerReceipt {
        schema: AGGRESSIVE_FLAVOR.worker_receipt_schema.to_string(),
        benchmark_valid: false,
        test_adaptive: manifest.test_adaptive,
        batch_id: manifest.batch_id.clone(),
        node: manifest.node.clone(),
        hostname: manifest.node.clone(),
        manifest_sha256: sha256_bytes(&manifest_bytes),
        started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
        candidates,
    };
    immutable_write(
        &output_dir.join("batch-worker-receipt.v1.json"),
        &canonical_json(&receipt)?,
    )?;
    Ok(())
}

pub(crate) fn island_batch_worker(manifest_path: &Path, output_dir: &Path) -> Result<()> {
    batch_worker(FROZEN_FLAVOR, manifest_path, output_dir)
}

pub(crate) fn aggressive_batch_worker(manifest_path: &Path, output_dir: &Path) -> Result<()> {
    aggressive_batch_worker_direct(manifest_path, output_dir)
}

pub(super) fn validate_batch_records(
    campaign_root: &Path,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    ordered_datasets: &[String],
    expected_fresh: &[ExpectedFresh],
) -> Result<()> {
    let receipt_dir = campaign_root.join("batches/receipts");
    let mut receipt_paths = fs::read_dir(&receipt_dir)
        .with_context(|| format!("read {}", receipt_dir.display()))?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    receipt_paths.sort();
    let mut records = Vec::new();
    let mut candidate_membership = HashSet::new();
    for receipt_path in receipt_paths {
        let receipt_bytes = fs::read(&receipt_path)?;
        let receipt: ScqBatchReceipt = serde_json::from_slice(&receipt_bytes)?;
        if receipt_bytes != canonical_json(&receipt)?
            || receipt.schema != SCQ_BATCH_RECEIPT_SCHEMA
            || receipt.benchmark_valid
            || receipt.batch_id != receipt.scq_job_id
            || !NODES.contains(&receipt.node.as_str())
            || receipt
                .hostname
                .as_deref()
                .is_some_and(|host| host != receipt.node)
            || receipt.finished_at_unix_ms < receipt.submitted_at_unix_ms
        {
            bail!("invalid immutable SCQ batch receipt");
        }
        let show_bytes = fs::read(campaign_root.join(&receipt.scq_show_file))?;
        if sha256_bytes(&show_bytes) != receipt.scq_show_sha256 {
            bail!("immutable SCQ show record changed");
        }
        let show: Value = serde_json::from_slice(&show_bytes)?;
        let job = scq_job(&show, &receipt.scq_job_id)?;
        if job.get("state").and_then(Value::as_str) != Some(receipt.scq_state.as_str())
            || job
                .get("node")
                .and_then(Value::as_str)
                .is_some_and(|node| node != receipt.node)
            || job.get("exit_code").and_then(Value::as_i64) != receipt.exit_code
            || job.get("cpus").and_then(Value::as_u64) != Some(SCQ_CPUS as u64)
            || job.get("project").and_then(Value::as_str) != Some(SCQ_PROJECT)
        {
            bail!("SCQ show record does not bind its batch job and resources");
        }
        let manifest_path = campaign_root.join(&receipt.manifest_file);
        let manifest_bytes = fs::read(&manifest_path)?;
        if sha256_bytes(&manifest_bytes) != receipt.manifest_sha256 {
            bail!("SCQ batch manifest hash mismatch");
        }
        let manifest: ScqBatchManifest = serde_json::from_slice(&manifest_bytes)?;
        if manifest_bytes != canonical_json(&manifest)?
            || manifest.schema != SCQ_BATCH_SCHEMA
            || manifest.benchmark_valid
            || manifest.batch_id != receipt.batch_id
            || manifest.node != receipt.node
            || manifest.attempt != receipt.attempt
            || manifest.project != SCQ_PROJECT
            || manifest.cpus != SCQ_CPUS
            || manifest.candidate_threads != 1
            || manifest.candidate_timeout_seconds != CANDIDATE_TIMEOUT_SECONDS
            || manifest.max_runtime_hours != 1
            || manifest.preemptible
            || manifest.scq_binary_sha256 != scq_binary_sha256
            || manifest.release_binary_sha256 != release_binary_sha256
            || manifest.bundle_root != remote_bundle_root(release_binary_sha256)
            || manifest.requests.is_empty()
            || manifest.requests.len() > SCQ_CPUS
            || manifest.batch_id
                != batch_id(
                    &manifest.run_id,
                    &manifest.node,
                    manifest.attempt,
                    &manifest.release_binary_sha256,
                    &manifest.requests,
                )?
        {
            bail!("SCQ batch manifest identity, hash, or resource mismatch");
        }
        let scq_spec_bytes = fs::read(campaign_root.join(&receipt.scq_spec_file))?;
        if sha256_bytes(&scq_spec_bytes) != receipt.scq_spec_sha256 {
            bail!("immutable SCQ submission spec changed");
        }
        let request_keys = manifest
            .requests
            .iter()
            .map(|request| {
                if request.run_id != manifest.run_id
                    || request.node != manifest.node
                    || request.execution_host != "scq"
                    || request.scq_job_id != manifest.batch_id
                    || request.attempt != manifest.attempt
                    || request.binary_sha256 != release_binary_sha256
                    || request.thread_counts != ThreadCounts::single()
                {
                    bail!("SCQ batch request binding mismatch");
                }
                let request_path = campaign_root.join(request_relative(request));
                if fs::read(&request_path)? != canonical_json(request)? {
                    bail!("immutable SCQ candidate request changed");
                }
                Ok((
                    request.dataset_hash.clone(),
                    request.generation,
                    request.slot,
                    request.attempt,
                    request.semantic_sha256.clone(),
                ))
            })
            .collect::<Result<HashSet<_>>>()?;
        if request_keys.len() != manifest.requests.len() {
            bail!("SCQ batch contains duplicate requests");
        }
        for candidate in &receipt.candidates {
            let key = (
                candidate.dataset_hash.clone(),
                candidate.generation,
                candidate.slot,
                candidate.attempt,
                candidate.semantic_sha256.clone(),
            );
            if !request_keys.contains(&key) || !candidate_membership.insert(key.clone()) {
                bail!("missing, duplicate, or unrequested fresh candidate receipt");
            }
            let request = manifest
                .requests
                .iter()
                .find(|request| {
                    request.dataset_hash == candidate.dataset_hash
                        && request.generation == candidate.generation
                        && request.slot == candidate.slot
                        && request.attempt == candidate.attempt
                        && request.semantic_sha256 == candidate.semantic_sha256
                })
                .unwrap();
            let candidate_path = campaign_root.join(outcome_relative(request));
            let bytes = fs::read(&candidate_path)?;
            if sha256_bytes(&bytes) != candidate.sha256 {
                bail!("SCQ candidate receipt changed after batch completion");
            }
            let candidate_receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
            if candidate_receipt.node != receipt.node
                || candidate_receipt.scq_job_id != receipt.scq_job_id
                || candidate_receipt.dataset_hash != candidate.dataset_hash
                || candidate_receipt.generation != candidate.generation
                || candidate_receipt.slot != candidate.slot
                || candidate_receipt.attempt != candidate.attempt
                || candidate_receipt.semantic_sha256 != candidate.semantic_sha256
            {
                bail!("SCQ hostname/job/candidate binding mismatch");
            }
        }
        records.push((manifest, receipt));
    }
    let expected = expected_fresh
        .iter()
        .map(|fresh| {
            (
                fresh.dataset_hash.clone(),
                fresh.generation,
                fresh.slot,
                fresh.attempt,
                fresh.semantic_sha256.clone(),
            )
        })
        .collect::<HashSet<_>>();
    let candidate_only = records
        .iter()
        .flat_map(|(manifest, receipt)| {
            receipt.candidates.iter().filter_map(move |candidate| {
                manifest
                    .requests
                    .iter()
                    .find(|request| {
                        request.dataset_hash == candidate.dataset_hash
                            && request.generation == candidate.generation
                            && request.slot == candidate.slot
                            && request.attempt == candidate.attempt
                    })
                    .filter(|request| request.purpose == "candidate")
                    .map(|_| {
                        (
                            candidate.dataset_hash.clone(),
                            candidate.generation,
                            candidate.slot,
                            candidate.attempt,
                            candidate.semantic_sha256.clone(),
                        )
                    })
            })
        })
        .collect::<HashSet<_>>();
    if expected != candidate_only || expected.len() != expected_fresh.len() {
        bail!("fresh SCQ request membership does not reconstruct checkpoints");
    }
    for fresh in expected_fresh {
        let (_, receipt) = records
            .iter()
            .find(|(_, receipt)| receipt.batch_id == fresh.scq_job_id)
            .context("fresh checkpoint receipt is absent from its SCQ batch")?;
        if receipt.node != fresh.node {
            bail!("fresh checkpoint node differs from SCQ batch node");
        }
    }
    let mut primary_groups = BTreeMap::<(usize, usize), Vec<&ScqBatchManifest>>::new();
    for (manifest, _) in &records {
        if manifest.attempt == 0
            && manifest
                .requests
                .first()
                .is_some_and(|request| request.purpose == "candidate")
        {
            let dataset = &manifest.requests[0].dataset_hash;
            let dataset_index = ordered_datasets
                .iter()
                .position(|value| value == dataset)
                .context("batch dataset is absent from ordered island cohort")?;
            primary_groups
                .entry((dataset_index, manifest.requests[0].generation))
                .or_default()
                .push(manifest);
        }
    }
    for ((dataset_index, generation), mut batches) in primary_groups {
        batches.sort_by_key(|manifest| manifest.requests.iter().map(|request| request.slot).min());
        for (ordinal, manifest) in batches.into_iter().enumerate() {
            if manifest.node != NODES[primary_node(dataset_index, generation, ordinal)] {
                bail!("incorrect deterministic primary SCQ node routing");
            }
        }
    }
    for (retry, _) in records.iter().filter(|(manifest, _)| manifest.attempt == 1) {
        for request in &retry.requests {
            let primary = records
                .iter()
                .find(|(manifest, _)| {
                    manifest.attempt == 0
                        && manifest.requests.iter().any(|candidate| {
                            candidate.dataset_hash == request.dataset_hash
                                && candidate.generation == request.generation
                                && candidate.slot == request.slot
                                && candidate.semantic_sha256 == request.semantic_sha256
                        })
                })
                .context("retry request has no primary attempt")?;
            let primary_node_index = NODES
                .iter()
                .position(|node| *node == primary.0.node)
                .context("primary batch has an unknown node")?;
            if retry.node != NODES[retry_node(primary_node_index)]
                || primary.1.candidates.iter().any(|candidate| {
                    candidate.dataset_hash == request.dataset_hash
                        && candidate.generation == request.generation
                        && candidate.slot == request.slot
                })
            {
                bail!("incorrect retry node or retry of a completed candidate");
            }
        }
    }
    for node in NODES {
        let mut intervals = records
            .iter()
            .filter(|(manifest, _)| manifest.node == node)
            .filter_map(|(_, receipt)| {
                Some((
                    receipt.started_at_unix_ms?,
                    receipt.worker_finished_at_unix_ms?,
                ))
            })
            .collect::<Vec<_>>();
        intervals.sort();
        if intervals.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            bail!("overlapping SCQ batches exceed the 90-worker node cap");
        }
    }
    for dataset in ordered_datasets {
        let mut generations = BTreeMap::<usize, (u128, u128)>::new();
        for (manifest, receipt) in &records {
            if manifest.requests.first().is_some_and(|request| {
                request.purpose == "candidate" && request.dataset_hash == *dataset
            }) {
                let (Some(start), Some(end)) = (
                    receipt.started_at_unix_ms,
                    receipt.worker_finished_at_unix_ms,
                ) else {
                    continue;
                };
                let generation = manifest.requests[0].generation;
                generations
                    .entry(generation)
                    .and_modify(|interval| {
                        interval.0 = interval.0.min(start);
                        interval.1 = interval.1.max(end);
                    })
                    .or_insert((start, end));
            }
        }
        for pair in generations.iter().collect::<Vec<_>>().windows(2) {
            let (left_generation, left_interval) = pair[0];
            let (right_generation, right_interval) = pair[1];
            if *left_generation + 1 != *right_generation || left_interval.1 > right_interval.0 {
                bail!("dataset-local generation barrier was violated");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn deterministic_routing_and_one_hop_failover_cover_the_ring() {
        for dataset in 0..7 {
            for generation in ISLAND_START_GENERATION..ISLAND_END_GENERATION {
                let primary = primary_node(dataset, generation, 0);
                assert!(primary < NODES.len());
                assert_eq!(retry_node(primary), (primary + 1) % NODES.len());
                assert_ne!(primary, retry_node(primary));
            }
        }
    }

    #[test]
    fn fixed_policy_binds_the_90_worker_cap_and_cpu_only_resources() {
        let policy = ScqPolicy::fixed();
        assert_eq!(policy.nodes, vec!["xbabe1", "xbabe2", "xbabe3"]);
        assert_eq!(policy.core_cap_per_node, 90);
        assert_eq!(policy.candidate_threads, 1);
        assert_eq!(policy.candidate_timeout_seconds, CANDIDATE_TIMEOUT_SECONDS);
        assert_eq!(policy.batch_max_runtime_hours, 1);
        assert_eq!(policy.infrastructure_retries, 1);
        assert!(!policy.preemptible);
    }

    #[test]
    fn aggressive_policy_matches_the_live_wave_executor() {
        let policy = ScqPolicy::aggressive_wave();
        assert_eq!(policy.core_cap_per_node, AGGRESSIVE_CPUS);
        assert_eq!(
            policy.candidate_timeout_seconds,
            AGGRESSIVE_CANDIDATE_TIMEOUT_SECONDS
        );
        assert_eq!(policy.infrastructure_retries, 3);
        assert_eq!(policy.batch_schema, AGGRESSIVE_BATCH_SCHEMA);
        assert_eq!(policy.receipt_schema, AGGRESSIVE_BATCH_RECEIPT_SCHEMA);
    }

    #[test]
    fn golden46_policy_removes_candidate_timeout_and_keeps_one_failover() {
        let policy = ScqPolicy::golden46_wave();
        assert_eq!(policy.candidate_timeout_seconds, 0);
        assert_eq!(
            policy.batch_max_runtime_hours,
            GOLDEN46_EFFECTIVELY_UNBOUNDED_RUNTIME_HOURS
        );
        assert_eq!(policy.infrastructure_retries, 1);
        assert_eq!(
            aggressive_flavor(GOLDEN46_RUN_ID).candidate_timeout_seconds,
            0
        );
        assert_eq!(
            aggressive_flavor(GOLDEN46_RUN_ID).max_runtime_hours,
            GOLDEN46_EFFECTIVELY_UNBOUNDED_RUNTIME_HOURS
        );
        assert_eq!(
            aggressive_flavor("golden42-all-regression-v1").candidate_timeout_seconds,
            AGGRESSIVE_CANDIDATE_TIMEOUT_SECONDS
        );
    }

    #[test]
    fn global_node_lease_enforces_the_controller_slot_limit() {
        let root = std::env::temp_dir().join(format!(
            "jope-scq-node-lease-{}-{}",
            std::process::id(),
            unix_millis()
        ));
        let mut leases = (0..NODE_LEASE_SLOTS)
            .map(|_| NodeLease::acquire_at(&root, "xbabe1").unwrap())
            .collect::<Vec<_>>();
        let (waiting_ready_sender, waiting_ready_receiver) = mpsc::channel();
        let waiting_root = root.clone();
        let waiting_handle = thread::spawn(move || {
            let _waiting = NodeLease::acquire_at(&waiting_root, "xbabe1").unwrap();
            waiting_ready_sender.send(()).unwrap();
        });
        assert!(waiting_ready_receiver
            .recv_timeout(Duration::from_millis(100))
            .is_err());
        leases.pop();
        waiting_ready_receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        waiting_handle.join().unwrap();
        drop(leases);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cpu_list_parser_and_aggressive_selection_fail_closed() {
        assert_eq!(parse_cpu_list("0-2,7,9-10").unwrap(), [0, 1, 2, 7, 9, 10]);
        assert!(parse_cpu_list("").is_err());
        assert!(parse_cpu_list("3-2").is_err());
        assert!(parse_cpu_list("1,1").is_err());

        let allowed = (10..140).collect::<Vec<_>>();
        assert_eq!(
            select_worker_cpus(&allowed, AGGRESSIVE_CPUS).unwrap(),
            (10..130).collect::<Vec<_>>()
        );
        assert!(select_worker_cpus(&allowed[..AGGRESSIVE_CPUS - 1], AGGRESSIVE_CPUS).is_err());
        let aggressive_allowed = &allowed[..AGGRESSIVE_CPUS];
        let oversubscribed =
            select_aggressive_worker_cpus(aggressive_allowed, AGGRESSIVE_WORKERS).unwrap();
        assert_eq!(oversubscribed.len(), AGGRESSIVE_WORKERS);
        assert_eq!(oversubscribed[0], 10);
        assert_eq!(oversubscribed[AGGRESSIVE_CPUS], 10);
        assert_eq!(
            constrain_job_cpus(&allowed, 36).unwrap(),
            (10..46).collect::<Vec<_>>()
        );
        assert!(constrain_job_cpus(&allowed[..35], 36).is_err());
    }

    #[test]
    fn aggressive_worker_caps_threads_while_accepting_overdecomposed_queues() {
        assert_eq!(AGGRESSIVE_DATASET_POPULATION, 36);
        assert_eq!(AGGRESSIVE_MAX_REQUESTS_PER_NODE, 504);
        assert_eq!(maximum_batch_requests(AGGRESSIVE_FLAVOR), 504);
        assert_eq!(maximum_batch_requests(FROZEN_FLAVOR), SCQ_CPUS);
        assert_eq!(reserved_batch_cpus(AGGRESSIVE_FLAVOR, 17).unwrap(), 17);
        assert_eq!(
            reserved_batch_cpus(AGGRESSIVE_FLAVOR, AGGRESSIVE_MAX_REQUESTS_PER_NODE).unwrap(),
            AGGRESSIVE_CPUS
        );
        assert_eq!(reserved_batch_cpus(FROZEN_FLAVOR, 1).unwrap(), SCQ_CPUS);
        assert_eq!(validate_aggressive_request_count(1).unwrap(), 1);
        assert_eq!(
            validate_aggressive_request_count(AGGRESSIVE_CPUS).unwrap(),
            AGGRESSIVE_CPUS
        );
        assert_eq!(
            validate_aggressive_request_count(AGGRESSIVE_MAX_REQUESTS_PER_NODE).unwrap(),
            AGGRESSIVE_WORKERS
        );
        assert!(validate_aggressive_request_count(0).is_err());
        assert!(validate_aggressive_request_count(AGGRESSIVE_MAX_REQUESTS_PER_NODE + 1).is_err());
    }

    #[test]
    fn candidate_thread_pins_to_one_allowed_cpu() {
        let cpu = job_allowed_cpus().unwrap()[0];
        thread::spawn(move || pin_current_thread(cpu))
            .join()
            .unwrap()
            .unwrap();
    }

    #[test]
    fn immutable_staging_requires_regular_exact_mode() {
        validate_immutable_mode(0o100555, 0o555).unwrap();
        validate_immutable_mode(0o100444, 0o444).unwrap();
        assert!(validate_immutable_mode(0o100755, 0o555).is_err());
        assert!(validate_immutable_mode(0o120555, 0o555).is_err());
    }

    #[test]
    fn post_lease_occupancy_is_node_exact_and_preserves_service_exclusions() {
        let queue = json!({
            "Jobs": {"jobs": [
                {"id": "job-a", "node": "xbabe1", "state": "running"},
                {"id": "job-b", "node": "xbabe2", "state": "queued"},
                {"id": "old-a", "node": "xbabe1", "state": "completed"},
                {"id": "old-b", "node": "xbabe1", "state": "failed"},
                {"id": "old-c", "node": "xbabe1", "state": "cancelled"},
                {"id": "queued-unassigned", "node": null, "state": "queued"},
                {"id": "old-unassigned", "node": null, "state": "failed"}
            ]}
        });
        assert_eq!(active_jobs_on_node(&queue, "xbabe1").unwrap(), ["job-a"]);
        assert_eq!(
            active_jobs_on_node(&queue, "xbabe3").unwrap(),
            Vec::<String>::new()
        );
        let services = format!(
            "1 scqd {scqd}\n2 scq {scq} controller serve\n3 scq {scq} agent status --node xbabe1\n",
            scqd = paths::scqd_binary().display(),
            scq = scq_binary_str(),
        );
        assert!(competing_node_processes(&services, "xbabe1", "xbabe2", "42").is_empty());
        assert!(competing_node_processes(
            "2365996 jope-dime-regre /opt/jope-dime-regressor run-golden42-aggressive --resume",
            "xbabe2",
            "xbabe2",
            "different-pid",
        )
        .is_empty());
        assert!(competing_node_processes(
            "2365997 jope-dime-regre /opt/jope-dime-regressor golden42-aggressive-batch-worker --manifest x",
            "xbabe1",
            "xbabe2",
            "different-pid",
        )
        .is_empty());
        assert!(competing_node_processes(
            "2365998 jope-dime-regre /opt/jope-dime-regressor test-adaptive-worker --request x",
            "xbabe1",
            "xbabe2",
            "different-pid",
        )
        .is_empty());
        assert!(is_campaign_job_id("jope-g42-abc"));
        assert!(!is_campaign_job_id("unrelated-job"));
        assert!(
            active_jobs_on_node(&json!({"Jobs": {"jobs": [{"id": "unbound"}]}}), "xbabe1").is_err()
        );

        let processes = "10 scqd /opt/scqd\n11 scq /opt/scq controller serve\n12 jope-dime-regressor /opt/jope-dime-regressor\n13 cargo cargo test\n";
        assert!(competing_node_processes(processes, "xbabe1", "xbabe1", "12").is_empty());
        assert_eq!(
            competing_node_processes(
                "14 jope-dime-regre /opt/jope-dime-regressor scq-aggressive-worker",
                "xbabe1",
                "xbabe2",
                "different-pid",
            ),
            ["14 jope-dime-regre /opt/jope-dime-regressor scq-aggressive-worker"]
        );
    }

    #[test]
    fn batch_watchdog_cancels_or_expires_fail_closed() {
        let (cancelled_tx, cancelled_rx) = std::sync::mpsc::sync_channel(2);
        cancelled_tx.send(()).unwrap();
        cancelled_tx.send(()).unwrap();
        assert!(!batch_watchdog_expired(
            cancelled_rx,
            Duration::from_secs(1)
        ));

        let (expired_tx, expired_rx) = std::sync::mpsc::sync_channel(1);
        expired_tx.send(()).unwrap();
        drop(expired_tx);
        assert!(batch_watchdog_expired(expired_rx, Duration::from_secs(1)));
    }

    #[test]
    fn batch_id_is_bound_to_the_frozen_release_binary() {
        let first = batch_id("run", "xbabe1", 0, "binary-a", &[]).unwrap();
        let second = batch_id("run", "xbabe1", 0, "binary-b", &[]).unwrap();
        assert_ne!(first, second);
    }
}
