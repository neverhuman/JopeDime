//! Native SmartCluster execution. Legacy funding gates remain unchanged.
use crate::{manifest::*, resident_protocol as resident};
use anyhow::{ensure, Context, Result};
use jain_smartcluster::{
    jope::{TaskKind, TaskManifest},
    Client, ControlMessage, FencedAttempt, JobRecord, JobState, ProtocolVersion,
    StructuredProgress, Timestamp, Validate, WorkerControlFrame, WorkerReport, WorkerReportFrame,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            fs::{MetadataExt, OpenOptionsExt},
            net::UnixStream,
        },
    },
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    time::Duration,
};

pub const POLICY_PATH: &str = "/etc/jope-smartcluster/worker.json";
pub const CONTROL_NAME: &str = "jope-subtask-v1";
const FRAME_LIMIT: usize = 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerPolicy {
    pub root: PathBuf,
    pub state_root: PathBuf,
    pub daemon_socket: PathBuf,
    pub cgroup_root: PathBuf,
    pub daemon_uid: u32,
    pub daemon_sha256: String,
    pub worker_sha256: String,
    pub allowed_runtime_sha256: std::collections::BTreeSet<String>,
    pub cpu_evaluator: Option<PathBuf>,
    #[serde(default)]
    pub datasets_executable: Option<PathBuf>,
    #[serde(default)]
    pub datasets_sha256: Option<String>,
    #[serde(default)]
    pub approved_dataset_packers: BTreeMap<PathBuf, String>,
    #[serde(default)]
    pub dataset_verification_timeout_ms: Option<u64>,
    pub fit_authority: Option<PathBuf>,
    pub fit_service_socket: Option<PathBuf>,
    pub fit_scope: Option<String>,
    pub fit_witness: Option<PathBuf>,
    pub minimum_available_memory_bytes: u64,
    pub storage_reserves: BTreeMap<PathBuf, u64>,
    pub storage_devices: BTreeMap<PathBuf, PathBuf>,
    pub scratch_root: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputManifest {
    pub schema: String,
    pub root: PathBuf,
    pub operation: Operation,
    #[serde(default)]
    pub input_artifacts: BTreeMap<PathBuf, String>,
    #[serde(default)]
    pub closure: Option<ClosureManifest>,
    pub output_receipt: PathBuf,
}

/// Every input frozen before continuation declarations existed. It never
/// carries a declaration.
pub const INPUT_SCHEMA_V1: &str = "jope.smartcluster_input.v1";
/// A training input that declares its lineage continuation mode
/// (W6 follow-up). Only `Operation::Train` uses it, and it always declares.
pub const INPUT_SCHEMA_V2: &str = "jope.smartcluster_input.v2";

impl InputManifest {
    /// The schema/declaration pairing:
    /// - v1 carries no declaration;
    /// - v2 is a training input declaring a lineage mode, never `exact_resume`,
    ///   which is an attempt property.
    pub fn validate_version(&self) -> Result<()> {
        let declared = match &self.operation {
            Operation::Train {
                continuation_mode, ..
            } => *continuation_mode,
            _ => None,
        };
        match self.schema.as_str() {
            INPUT_SCHEMA_V1 => ensure!(
                declared.is_none(),
                "a continuation declaration requires {INPUT_SCHEMA_V2}"
            ),
            INPUT_SCHEMA_V2 => {
                ensure!(
                    matches!(self.operation, Operation::Train { .. }),
                    "{INPUT_SCHEMA_V2} is a training input"
                );
                let mode = declared.with_context(|| {
                    format!("{INPUT_SCHEMA_V2} must declare its continuation mode")
                })?;
                ensure!(
                    mode != crate::task_contract::ContinuationMode::ExactResume,
                    "exact resume is an attempt property; declare the lineage mode"
                );
            }
            other => anyhow::bail!("unknown input manifest schema {other}"),
        }
        Ok(())
    }

    /// The training input to freeze for a request, under the version it was
    /// first frozen with. `existing` holds the bytes already frozen at this
    /// input's path, if any:
    /// - absent: a new input, frozen as v2 with the declared `mode`;
    /// - a v1 input: re-materialized as v1, with no declaration, so its bytes
    ///   are unchanged;
    /// - a v2 input: re-materialized as v2 with the same declaration.
    ///
    /// `self` is the undeclared v1 form the materializer always built.
    pub fn declare_training(
        mut self,
        existing: Option<&[u8]>,
        mode: crate::task_contract::ContinuationMode,
    ) -> Result<Self> {
        ensure!(
            self.schema == INPUT_SCHEMA_V1,
            "a training input is declared from its undeclared form"
        );
        let Operation::Train {
            continuation_mode, ..
        } = &mut self.operation
        else {
            anyhow::bail!("only a training input declares a continuation mode")
        };
        ensure!(
            continuation_mode.is_none(),
            "a training input is declared from its undeclared form"
        );
        let version = match existing {
            None => INPUT_SCHEMA_V2.to_string(),
            Some(bytes) => {
                #[derive(Deserialize)]
                struct Version {
                    schema: String,
                }
                serde_json::from_slice::<Version>(bytes)
                    .context("frozen input manifest unreadable")?
                    .schema
            }
        };
        match version.as_str() {
            INPUT_SCHEMA_V1 => {}
            INPUT_SCHEMA_V2 => {
                *continuation_mode = Some(mode);
                self.schema = INPUT_SCHEMA_V2.into();
            }
            other => anyhow::bail!("unknown frozen input manifest schema {other}"),
        }
        self.validate_version()?;
        Ok(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClosureManifest {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClosureEntry {
    path: PathBuf,
    sha256: String,
}

fn read_closure(root: &Path, closure: &ClosureManifest) -> Result<BTreeMap<PathBuf, String>> {
    use std::io::BufRead;
    let path = resolve(root, &closure.path)?;
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() <= 256 * 1024 * 1024,
        "closure size/type invalid"
    );
    let mut reader = std::io::BufReader::new(file);
    let mut digest = sha2::Sha256::new();
    use sha2::Digest;
    let mut entries = BTreeMap::new();
    let mut total = 0usize;
    loop {
        let mut line = Vec::new();
        let bytes = reader.by_ref().take(8193).read_until(b'\n', &mut line)?;
        if bytes == 0 {
            break;
        }
        total += bytes;
        ensure!(
            bytes <= 8192 && total <= 256 * 1024 * 1024 && entries.len() < 1_000_000,
            "closure record/count/byte limit exceeded"
        );
        digest.update(&line);
        let entry: ClosureEntry = serde_json::from_slice(&line)?;
        relative(&entry.path)?;
        ensure!(
            entry.sha256.len() == 64
                && entry
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "closure digest invalid"
        );
        ensure!(
            entries.insert(entry.path, entry.sha256).is_none(),
            "duplicate closure path"
        );
    }
    ensure!(
        format!("{:x}", digest.finalize()) == closure.sha256,
        "closure bytes changed"
    );
    Ok(entries)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum Operation {
    Resident,
    DirectEvaluation {
        manifest: PathBuf,
        sha256: String,
    },
    SeededContinuation {
        manifest: PathBuf,
        sha256: String,
    },
    Search {
        prepared: PathBuf,
        cut: usize,
        output: PathBuf,
        evaluations: usize,
        seconds: u64,
        qualification: bool,
        seeds: Option<PathBuf>,
        vocabulary: Option<PathBuf>,
        rng_seed: u64,
    },
    Pack {
        contexts: PathBuf,
        vocabulary: PathBuf,
        observations: Vec<PathBuf>,
        output: PathBuf,
        qualification: bool,
    },
    DatasetPublication {
        manifest: PathBuf,
        sha256: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logical_attempt: Option<u64>,
    },
    PackAndPublish {
        manifest: PathBuf,
        sha256: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logical_attempt: Option<u64>,
    },
    Verify,
    Train {
        corpus: PathBuf,
        run: PathBuf,
        launch: PathBuf,
        micro: usize,
        checkpoint_steps: usize,
        initialize: PathBuf,
        replay_policy: PathBuf,
        initial_optimizer_step: u64,
        requested_step_increment: u64,
        target_optimizer_step: u64,
        /// Explicit lineage-mode declaration; absent on requests frozen before
        /// it, which keep their implicit meaning (W6 follow-up).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        continuation_mode: Option<crate::task_contract::ContinuationMode>,
    },
    Proposal {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        preparation_overlay: Option<crate::native_graph::Artifact>,
        champion_snapshot: PathBuf,
        corpus: PathBuf,
        checkpoint: PathBuf,
        output: PathBuf,
        cuts: usize,
        cut_start: usize,
        shuffled: bool,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubtaskCommand {
    pub manifest: PathBuf,
    pub manifest_sha256: String,
}

impl Operation {
    pub(crate) fn dataset_logical_attempt(&self) -> Option<u64> {
        match self {
            Self::DatasetPublication {
                logical_attempt, ..
            }
            | Self::PackAndPublish {
                logical_attempt, ..
            } => *logical_attempt,
            _ => None,
        }
    }
    pub(crate) fn dataset_attempt(&self, native_number: u64) -> Result<u64> {
        let attempt = self.dataset_logical_attempt().unwrap_or(native_number);
        ensure!(attempt > 0, "dataset logical attempt must be positive");
        Ok(attempt)
    }
}

/// Opaque process-local proof; only verified native allocation dispatch constructs it.
pub struct VerifiedTask {
    #[cfg(feature = "neural")]
    attempt: FencedAttempt,
    #[cfg(feature = "neural")]
    manifest: TaskManifest,
    #[cfg(feature = "neural")]
    root: PathBuf,
    operation: Operation,
    stop: Arc<AtomicBool>,
    #[cfg(feature = "neural")]
    recovering: bool,
}
impl VerifiedTask {
    #[cfg(feature = "neural")]
    pub(crate) fn attempt(&self) -> FencedAttempt {
        self.attempt
    }
    #[cfg(feature = "neural")]
    pub(crate) fn manifest(&self) -> &TaskManifest {
        &self.manifest
    }
    #[cfg(feature = "neural")]
    pub(crate) fn root(&self) -> &Path {
        &self.root
    }
    pub(crate) fn operation(&self) -> &Operation {
        &self.operation
    }
    #[cfg(feature = "neural")]
    pub(crate) fn stop(&self) -> &AtomicBool {
        &self.stop
    }
    #[cfg(feature = "neural")]
    pub(crate) fn recovering(&self) -> bool {
        self.recovering
    }
}

fn read_bounded(path: &Path, expected: Option<&str>) -> Result<Vec<u8>> {
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    ensure!(file.metadata()?.is_file(), "expected regular manifest");
    let mut bytes = Vec::new();
    file.take(FRAME_LIMIT as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= FRAME_LIMIT, "manifest exceeds 1 MiB");
    if let Some(expected) = expected {
        ensure!(hash(&bytes) == expected, "immutable manifest changed");
    }
    Ok(bytes)
}

fn load_policy() -> Result<WorkerPolicy> {
    let path = Path::new(POLICY_PATH);
    for ancestor in path.ancestors() {
        let meta = fs::symlink_metadata(ancestor)?;
        ensure!(
            !meta.file_type().is_symlink() && meta.uid() == 0 && meta.mode() & 0o022 == 0,
            "worker policy path is not root-controlled"
        );
    }
    let policy: WorkerPolicy = serde_json::from_slice(&read_bounded(path, None)?)?;
    for path in [
        &policy.root,
        &policy.state_root,
        &policy.daemon_socket,
        &policy.cgroup_root,
        &policy.scratch_root,
    ] {
        ensure!(path.is_absolute(), "worker policy paths must be absolute");
    }
    ensure!(
        policy.root.canonicalize()? == policy.root,
        "worker root must be canonical"
    );
    ensure!(
        file_hash(Path::new("/proc/self/exe"))? == policy.worker_sha256,
        "loaded worker binary is not qualified"
    );
    verify_loaded_runtime(&policy)?;
    Ok(policy)
}

pub fn runtime_observation() -> Result<Value> {
    crate::runtime_custody::loaded_runtime_descriptor(&crate::eval_registry::ProcessIdentity::read(
        std::process::id(),
    )?)
}
fn verify_loaded_runtime(policy: &WorkerPolicy) -> Result<()> {
    let observed = runtime_observation()?;
    let digest = observed["runtime_sha256"]
        .as_str()
        .context("loaded runtime digest absent")?;
    ensure!(
        policy.allowed_runtime_sha256.contains(digest),
        "loaded runtime is not in qualified policy"
    );
    Ok(())
}

struct Allocation {
    policy: WorkerPolicy,
    attempt: FencedAttempt,
    stream: UnixStream,
    bootstrap: TaskManifest,
    daemon_identity: crate::eval_registry::ProcessIdentity,
}

fn current_job(
    policy: &WorkerPolicy,
    id: jain_smartcluster::JobId,
    expected: &crate::eval_registry::ProcessIdentity,
) -> Result<JobRecord> {
    match crate::eval_registry::native_scq::authenticated_exchange(
        &policy.daemon_socket,
        policy.daemon_uid,
        &policy.daemon_sha256,
        Some(expected),
        jain_smartcluster::Request::Get(jain_smartcluster::GetRequest { job_id: id }),
    )? {
        jain_smartcluster::Response::Job(job) => {
            let job = job.context("native allocation absent")?;
            let attempt = job.active_attempt.context("native active attempt absent")?;
            let leader = crate::eval_registry::native_scq::active_process(
                &policy.daemon_socket,
                policy.daemon_uid,
                &policy.daemon_sha256,
                Some(expected),
                attempt,
            )?;
            ensure!(
                leader == crate::eval_registry::ProcessIdentity::current()?,
                "worker is not its original scheduler leader"
            );
            Ok(job)
        }
        _ => anyhow::bail!("native allocation response differs"),
    }
}

fn retain_control_descriptor(descriptor: i32) -> Result<UnixStream> {
    // SAFETY: F_DUPFD_CLOEXEC takes an integer minimum and touches no memory; an invalid
    // `descriptor` only yields -1 (EBADF), checked below.
    let duplicate = unsafe { libc::fcntl(descriptor, libc::F_DUPFD_CLOEXEC, 3) };
    ensure!(duplicate >= 0, "cannot retain control descriptor");
    ensure!(
        // SAFETY: F_SETFD takes an integer flag and touches no memory; an invalid
        // `descriptor` only yields -1, which is checked.
        unsafe { libc::fcntl(descriptor, libc::F_SETFD, libc::FD_CLOEXEC) } == 0,
        "cannot protect inherited control descriptor"
    );
    // SAFETY: F_DUPFD_CLOEXEC just returned this fresh, nonnegative duplicate that nothing else
    // owns; UnixStream becomes its sole owner and closes it once. `descriptor` stays separate.
    Ok(unsafe { UnixStream::from_raw_fd(duplicate) })
}

fn authenticate(policy: WorkerPolicy, path: &Path, sha: &str) -> Result<Allocation> {
    let descriptor: i32 = std::env::var("SCQ_CONTROL_FD")
        .context("inherited native control socket required")?
        .parse()?;
    ensure!(descriptor >= 3, "invalid control descriptor");
    let stream = retain_control_descriptor(descriptor)?;
    std::env::remove_var("SCQ_CONTROL_FD");
    let peer = resident::peer_credentials(&stream)?;
    let birth = resident::process_identity(peer.pid as u32)?;
    ensure!(
        peer.uid == policy.daemon_uid
            && file_hash(Path::new(&format!("/proc/{}/exe", peer.pid)))? == policy.daemon_sha256,
        "unqualified daemon control peer"
    );
    let daemon_identity = crate::eval_registry::ProcessIdentity::read(peer.pid as u32)?;
    let id = std::env::var("SCQ_JOB_ID")?.parse()?;
    let job = current_job(&policy, id, &daemon_identity)?;
    let attempt = job
        .active_attempt
        .context("allocation has no active fenced attempt")?;
    ensure!(
        attempt.number.to_string() == std::env::var("SCQ_ATTEMPT_NUMBER")?
            && attempt.fence.to_string() == std::env::var("SCQ_ATTEMPT_FENCE")?,
        "native attempt fence changed"
    );
    ensure!(
        matches!(job.state, JobState::Starting | JobState::Running),
        "native allocation is not live"
    );
    let expected = policy.cgroup_root.join(format!(
        "scq-{}-attempt-{}-fence-{}",
        attempt.job_id, attempt.number, attempt.fence
    ));
    let own = fs::read_to_string("/proc/self/cgroup")?;
    let actual = own
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .context("unified cgroup membership absent")?;
    ensure!(
        Path::new("/sys/fs/cgroup").join(actual.trim_start_matches('/')) == expected,
        "worker is outside its native fenced cgroup"
    );
    let spec = jain_smartcluster::jope::job_spec(
        path,
        sha,
        job.spec.tenant.clone(),
        job.spec.project.clone(),
        Timestamp::from(resident::now_ms()?),
    )?;
    ensure!(job.spec == spec, "manifest does not match native JobSpec");
    ensure!(
        resident::process_identity(peer.pid as u32)? == birth,
        "daemon peer changed during authentication"
    );
    let bootstrap = serde_json::from_slice(&read_bounded(path, Some(sha))?)?;
    Ok(Allocation {
        policy,
        attempt,
        stream,
        bootstrap,
        daemon_identity,
    })
}

fn load_task(
    policy: &WorkerPolicy,
    path: &Path,
    sha: &str,
) -> Result<(TaskManifest, InputManifest)> {
    verify_task_inputs(&policy.root, path, sha)
}

pub(crate) fn verify_task_inputs(
    root: &Path,
    path: &Path,
    sha: &str,
) -> Result<(TaskManifest, InputManifest)> {
    let task: TaskManifest = serde_json::from_slice(&read_bounded(path, Some(sha))?)?;
    task.validate_at(Timestamp::from(resident::now_ms()?))?;
    let input_path = path.with_file_name(format!(
        "{}.input.json",
        path.file_stem()
            .context("manifest stem absent")?
            .to_string_lossy()
    ));
    let mut input: InputManifest = serde_json::from_slice(&read_bounded(
        &input_path,
        Some(&task.input_manifest_sha256),
    )?)?;
    ensure!(input.root == root, "input root/schema changed");
    input
        .validate_version()
        .context("input root/schema changed")?;
    let dependencies: BTreeMap<String, String> = if let Some(closure) = &input.closure {
        ensure!(
            input.input_artifacts.is_empty(),
            "closure and inline artifacts cannot be mixed"
        );
        BTreeMap::from([(
            closure.path.to_string_lossy().into_owned(),
            closure.sha256.clone(),
        )])
    } else {
        input
            .input_artifacts
            .iter()
            .map(|(path, sha)| (path.to_string_lossy().into_owned(), sha.clone()))
            .collect()
    };
    ensure!(
        dependencies == task.dependencies,
        "input closure differs from task dependencies"
    );
    if let Some(closure) = &input.closure {
        input.input_artifacts = read_closure(root, closure)?;
    }
    verify_training_closure(root, &input)?;
    let _verified_input_scope = crate::native_input_cache::enter(root, &input.input_artifacts)?;
    for (path, sha) in &input.input_artifacts {
        ensure!(
            file_hash(&resolve(root, path)?)? == *sha,
            "dependency changed: {}",
            path.display()
        );
    }
    relative(&input.output_receipt)?;
    validate_operation(&task, &input)?;
    match &input.operation {
        Operation::Train {
            corpus, initialize, ..
        }
        | Operation::Proposal {
            corpus,
            checkpoint: initialize,
            ..
        } => {
            ensure!(
                task.corpus_sha256.as_deref()
                    == Some(file_hash(&resolve(root, corpus)?.join("index.json"))?.as_str()),
                "task corpus identity differs from numerical input"
            );
            ensure!(
                task.parent_checkpoint_sha256.as_deref()
                    == Some(file_hash(&resolve(root, initialize)?.join("state.json"))?.as_str()),
                "task parent identity differs from numerical input"
            );
        }
        _ => {}
    }
    verify_required_inputs(&input)?;
    Ok((task, input))
}

fn verify_training_closure(root: &Path, input: &InputManifest) -> Result<()> {
    if let Operation::Train { corpus, .. } = &input.operation {
        // Historical Train manifests over unprojected corpora fail closed here by
        // design: re-verifying them would open every excluded context they bind.
        relative(corpus)?;
        let path = corpus.join("index.json");
        let sha256 = input
            .input_artifacts
            .get(&path)
            .cloned()
            .context("training corpus index is not in dependency closure")?;
        let index = crate::native_genome::load_corpus_index(
            root,
            &crate::native_graph::Artifact { path, sha256 },
        )?;
        crate::corpus_projection::verify_admitted_context_closure(&index)?;
    }
    Ok(())
}

fn validate_operation(task: &TaskManifest, input: &InputManifest) -> Result<()> {
    ensure!(
        input.operation.dataset_logical_attempt() != Some(0),
        "dataset logical attempt must be positive"
    );
    match (&task.kind, &input.operation) {
        (
            TaskKind::ResidentGpu,
            Operation::Resident | Operation::Train { .. } | Operation::Proposal { .. },
        )
        | (TaskKind::GpSearch, Operation::DirectEvaluation { .. })
        | (TaskKind::Continuation, Operation::SeededContinuation { .. })
        | (TaskKind::GpSearch, Operation::Search { seeds: None, .. })
        | (TaskKind::Continuation, Operation::Search { seeds: Some(_), .. })
        | (TaskKind::Packing, Operation::Pack { .. } | Operation::PackAndPublish { .. })
        | (TaskKind::Publication, Operation::DatasetPublication { .. })
        | (TaskKind::Verification, Operation::Verify) => {}
        _ => anyhow::bail!(
            "operation is not implemented by this fixed service; publication belongs to dataset authority"
        ),
    }
    if !matches!(input.operation, Operation::Resident | Operation::Verify) {
        ensure!(
            task.resources.scratch_bytes > 0,
            "numerical operation requires explicit scratch reservation"
        );
    }
    if let Operation::Train {
        micro,
        checkpoint_steps,
        initial_optimizer_step,
        requested_step_increment,
        target_optimizer_step,
        ..
    } = &input.operation
    {
        ensure!(
            [64, 128, 256].contains(micro)
                && *checkpoint_steps > 0
                && *requested_step_increment > 0
                && initial_optimizer_step.checked_add(*requested_step_increment)
                    == Some(*target_optimizer_step),
            "original optimizer window invalid"
        );
    }
    for path in operation_inputs(&input.operation) {
        relative(path)?;
        ensure!(
            input
                .input_artifacts
                .keys()
                .any(|bound| bound == path || bound.starts_with(path)),
            "operation input is not in dependency closure: {}",
            path.display()
        );
    }
    for path in operation_outputs(&input.operation) {
        relative(path)?;
    }
    Ok(())
}

fn verify_required_inputs(input: &InputManifest) -> Result<()> {
    let mut required = BTreeMap::new();
    match &input.operation {
        Operation::DirectEvaluation { manifest, sha256 }
        | Operation::SeededContinuation { manifest, sha256 } => {
            let reference = crate::native_graph::Artifact {
                path: manifest.clone(),
                sha256: sha256.clone(),
            };
            required = crate::native_genome::required_inputs(&input.root, &reference)?;
            let task: crate::native_genome::GenomeTask =
                crate::native_genome::load(&input.root, &reference)?;
            ensure!(
                (task.stage == crate::native_genome::Stage::Direct)
                    == matches!(input.operation, Operation::DirectEvaluation { .. }),
                "native genome operation stage changed"
            );
        }
        Operation::DatasetPublication {
            manifest, sha256, ..
        }
        | Operation::PackAndPublish {
            manifest, sha256, ..
        } => {
            ensure!(
                file_hash(&resolve(&input.root, manifest)?)? == *sha256,
                "dataset manifest changed"
            );
            required.insert(manifest.clone(), sha256.clone());
        }
        Operation::Train {
            corpus,
            run,
            launch,
            micro,
            checkpoint_steps,
            initialize,
            replay_policy,
            requested_step_increment,
            ..
        } => {
            required = resident::input_artifacts(
                &input.root,
                &json!({"kind":"Train","corpus":corpus,"run":run,"launch":launch,"world":1,"micro":micro,"steps":requested_step_increment,"checkpoint_steps":checkpoint_steps,"resume":false,"initialize":initialize,"replay_policy":replay_policy}),
            )?;
        }
        Operation::Proposal {
            corpus,
            checkpoint,
            output,
            cuts,
            cut_start,
            shuffled,
            champion_snapshot,
            preparation_overlay,
        } => {
            required = resident::request_artifacts(
                &input.root,
                &json!({"kind":"Proposal","corpus":corpus,"checkpoint":checkpoint,"output":output,"world":1,"cuts":cuts,"cut_start":cut_start,"shuffled":shuffled}),
                &json!({"JOPE_CHAMPION_SNAPSHOT":champion_snapshot}),
            )?;
            if let Some(a) = preparation_overlay {
                required.extend(crate::preparation_overlay::dependencies(&input.root, a)?);
            }
        }
        Operation::Search { prepared, .. } => {
            let prepared_path = resolve(&input.root, prepared)?;
            let metadata: crate::prepare::Prepared =
                read_json(&prepared_path.join("prepared.json"))?;
            ensure!(
                metadata.meta_partition != "meta_test",
                "sealed-test search admission refused"
            );
            for entry in walkdir::WalkDir::new(&prepared_path).follow_links(false) {
                let entry = entry?;
                ensure!(
                    !entry.file_type().is_symlink(),
                    "prepared input symlink refused"
                );
                if entry.file_type().is_file() {
                    required.insert(
                        entry.path().strip_prefix(&input.root)?.to_owned(),
                        file_hash(entry.path())?,
                    );
                }
            }
            if let Some(storage) = metadata.storage {
                let source = if storage.source.is_absolute() {
                    storage.source
                } else {
                    input.root.join(storage.source)
                };
                let relative = source.strip_prefix(&input.root)?.to_owned();
                ensure!(
                    file_hash(&resolve(&input.root, &relative)?)? == storage.source_sha256,
                    "prepared observation source changed"
                );
                required.insert(relative, storage.source_sha256);
            }
        }
        _ => {}
    }
    ensure!(
        required
            .iter()
            .all(|(path, sha)| input.input_artifacts.get(path) == Some(sha)),
        "required numerical dependency closure incomplete"
    );
    Ok(())
}

fn operation_inputs(op: &Operation) -> Vec<&Path> {
    match op {
        Operation::DirectEvaluation { manifest, .. }
        | Operation::SeededContinuation { manifest, .. } => vec![manifest],
        Operation::DatasetPublication { manifest, .. }
        | Operation::PackAndPublish { manifest, .. } => vec![manifest],
        Operation::Search {
            prepared,
            seeds,
            vocabulary,
            ..
        } => std::iter::once(prepared.as_path())
            .chain(seeds.iter().map(PathBuf::as_path))
            .chain(vocabulary.iter().map(PathBuf::as_path))
            .collect(),
        Operation::Pack {
            contexts,
            vocabulary,
            observations,
            ..
        } => std::iter::once(contexts.as_path())
            .chain(std::iter::once(vocabulary.as_path()))
            .chain(observations.iter().map(PathBuf::as_path))
            .collect(),
        Operation::Train {
            corpus,
            initialize,
            replay_policy,
            ..
        } => vec![corpus, initialize, replay_policy],
        Operation::Proposal {
            corpus,
            checkpoint,
            champion_snapshot,
            ..
        } => vec![corpus, checkpoint, champion_snapshot],
        _ => vec![],
    }
}
fn operation_outputs(op: &Operation) -> Vec<&Path> {
    match op {
        Operation::Search { output, .. }
        | Operation::Pack { output, .. }
        | Operation::Proposal { output, .. } => vec![output],
        Operation::Train { run, launch, .. } => vec![run, launch],
        _ => vec![],
    }
}

fn admission_guards(policy: &WorkerPolicy) -> Result<()> {
    for (mountpoint, device) in &policy.storage_devices {
        let mounts = fs::read_to_string("/proc/self/mountinfo")?;
        let source = mounts
            .lines()
            .find_map(|line| {
                let (left, right) = line.split_once(" - ")?;
                (left.split_whitespace().nth(4)? == mountpoint.to_str()?)
                    .then(|| right.split_whitespace().nth(1))
                    .flatten()
            })
            .context("required storage mount absent")?;
        let actual = fs::metadata(source)?;
        let expected = fs::metadata(device)?;
        use std::os::unix::fs::FileTypeExt;
        ensure!(
            actual.file_type().is_block_device()
                && expected.file_type().is_block_device()
                && actual.rdev() == expected.rdev(),
            "storage mount device changed"
        );
    }

    let memory = fs::read_to_string("/proc/meminfo")?;
    let available: u64 = memory
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .context("MemAvailable absent")?
        .split_whitespace()
        .next()
        .context("MemAvailable value absent")?
        .parse()?;
    ensure!(
        available.saturating_mul(1024) >= policy.minimum_available_memory_bytes,
        "RAM reserve blocks subtask"
    );
    for (path, reserve) in &policy.storage_reserves {
        let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
        let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        ensure!(
            // SAFETY: `path` is a CString that outlives the call (NUL-terminated) and `stats`
            // has room for the one statvfs that statvfs writes.
            unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } == 0,
            "storage measurement unavailable"
        );
        // SAFETY: statvfs returned 0 (checked above), so it filled every field of `stats`.
        let stats = unsafe { stats.assume_init() };
        ensure!(
            stats.f_bavail.saturating_mul(stats.f_frsize) >= *reserve,
            "storage reserve blocks subtask"
        );
    }
    Ok(())
}

fn read_frame(stream: &mut UnixStream) -> Result<WorkerControlFrame> {
    let mut header = [0; 4];
    stream.read_exact(&mut header)?;
    let length = u32::from_be_bytes(header) as usize;
    ensure!(length <= FRAME_LIMIT, "control frame too large");
    let mut bytes = vec![0; length];
    stream.read_exact(&mut bytes)?;
    let (frame, remaining): (WorkerControlFrame, &[u8]) = postcard::take_from_bytes(&bytes)?;
    ensure!(remaining.is_empty(), "trailing control bytes");
    Ok(frame)
}
fn report(
    stream: &mut UnixStream,
    attempt: FencedAttempt,
    sequence: &mut u64,
    report: WorkerReport,
) -> Result<()> {
    report.validate()?;
    *sequence = sequence
        .checked_add(1)
        .context("report sequence overflow")?;
    let bytes = postcard::to_stdvec(&WorkerReportFrame {
        version: ProtocolVersion::CURRENT,
        attempt,
        sequence: *sequence,
        report,
    })?;
    ensure!(bytes.len() <= FRAME_LIMIT, "report too large");
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    stream.write_all(&(bytes.len() as u32).to_be_bytes())?;
    stream.write_all(&bytes)?;
    Ok(())
}

fn within_allocation(child: &TaskManifest, parent: &TaskManifest) -> Result<()> {
    ensure!(
        child.kind == TaskKind::ResidentGpu
            && child.checkpoint_allowance_ms.unwrap_or(1000)
                <= parent.checkpoint_allowance_ms.unwrap_or(1000)
            && child.run == parent.run
            && child.deadline <= parent.deadline
            && child.resources.cpu.maximum_millicores <= parent.resources.cpu.maximum_millicores
            && child.resources.ram.hard_bytes <= parent.resources.ram.hard_bytes
            && child.resources.scratch_bytes <= parent.resources.scratch_bytes
            && child.resources.gpu == parent.resources.gpu,
        "resident subtask exceeds native allocation"
    );
    Ok(())
}

/// Persist the coordinator's dispatch intent before `Client::send_control` delivery.
pub fn freeze_dispatch_intent(path: &Path, command: &SubtaskCommand) -> Result<()> {
    let bytes = serde_json::to_vec(command)?;
    if path.exists() {
        ensure!(
            fs::read(path)? == bytes,
            "conflicting subtask dispatch intent"
        );
        return Ok(());
    }
    create_immutable(path, &bytes)
}
/// Delivery acknowledgement is not completion; verify the durable receipt separately.
pub async fn dispatch_subtask(
    client: &Client,
    job_id: jain_smartcluster::JobId,
    intent: &Path,
    command: &SubtaskCommand,
) -> Result<jain_smartcluster::OperationAck> {
    let payload = serde_json::to_vec(command)?;
    ensure!(payload.len() <= 64 * 1024, "subtask command too large");
    freeze_dispatch_intent(intent, command)?;
    Ok(client
        .send_control(jain_smartcluster::SendControlRequest {
            job_id,
            control: ControlMessage::Application {
                name: CONTROL_NAME.to_owned(),
                payload,
            },
        })
        .await?)
}
pub(crate) fn write_immutable_record(path: &Path, value: &Value) -> Result<()> {
    create_immutable(path, &serde_json::to_vec(value)?)
}
fn create_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure!(
        bytes.len() <= FRAME_LIMIT,
        "durable control record too large"
    );
    let parent = path.parent().context("record parent absent")?;
    fs::create_dir_all(parent)?;
    // Keep interrupted staging files: final names become visible only after a
    // complete fsynced payload, using an atomic no-replace rename publication.
    let temporary = tempfile::Builder::new()
        .prefix(".immutable-pending-")
        .tempfile_in(parent)?;
    let (mut file, temporary_path) = temporary.keep()?;
    file.write_all(bytes)?;
    file.sync_all()?;
    match crate::native_persistence::publish_staged(&temporary_path, path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            ensure!(fs::read(path)? == bytes, "conflicting immutable record");
        }
        Err(error) => return Err(error.into()),
    }
    fs::File::open(parent)?.sync_all()?;
    crate::native_persistence::discard_staging(&temporary_path)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

fn execute(
    allocation: &Allocation,
    path: &Path,
    sha: &str,
    stop: Arc<AtomicBool>,
) -> Result<Value> {
    admission_guards(&allocation.policy)?;
    verify_loaded_runtime(&allocation.policy)?;
    let current = current_job(
        &allocation.policy,
        allocation.attempt.job_id,
        &allocation.daemon_identity,
    )?;
    ensure!(
        current.active_attempt == Some(allocation.attempt)
            && matches!(current.state, JobState::Starting | JobState::Running),
        "superseded native allocation"
    );
    let (task, input) = load_task(&allocation.policy, path, sha)?;
    if allocation.bootstrap.kind == TaskKind::ResidentGpu {
        within_allocation(&task, &allocation.bootstrap)?;
    }
    let key = hash(&serde_json::to_vec(&(
        &task.run,
        task.generation,
        &task.task_id,
    ))?);
    let directory = allocation.policy.state_root.join("tasks").join(key);
    fs::create_dir_all(&directory)?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(directory.join("lock"))?;
    ensure!(lock.try_lock().is_ok(), "task is already executing");
    let intent = json!({"manifest_sha256":sha,"task":task,"input_manifest_sha256":task.input_manifest_sha256,"operation":input.operation,"output_receipt":input.output_receipt});
    let intent_path = directory.join("intent.json");
    if intent_path.exists() {
        ensure!(
            read_json::<Value>(&intent_path)? == intent,
            "conflicting duplicate task identity"
        );
    } else {
        create_immutable(&intent_path, &serde_json::to_vec(&intent)?)?;
    }
    let output = resolve(&allocation.policy.root, &input.output_receipt)?;
    let complete = directory.join("complete.json");
    if complete.exists() {
        let result: Value = read_json(&complete)?;
        ensure!(
            result["output_sha256"] == file_hash(&output)? && result["manifest_sha256"] == sha,
            "durable result changed"
        );
        return Ok(result);
    }
    let overlay = match &input.operation {
        Operation::Proposal {
            preparation_overlay,
            ..
        } => preparation_overlay.clone(),
        Operation::DirectEvaluation { manifest, sha256 }
        | Operation::SeededContinuation { manifest, sha256 } => {
            let g: crate::native_genome::GenomeTask = crate::native_genome::load(
                &input.root,
                &crate::native_graph::Artifact {
                    path: manifest.clone(),
                    sha256: sha256.clone(),
                },
            )?;
            g.preparation_overlay
        }
        _ => None,
    };
    if let Some(a) = overlay {
        crate::preparation_overlay::load(&input.root, &a, &task.run)?
            .require_live(resident::now_ms()?, task.deadline.as_unix_millis())?;
    }
    let active = directory.join("active.json");
    let recovering = active.exists();
    if recovering {
        let previous: Value = read_json(&active)?;
        ensure!(
            previous["job_id"] == allocation.attempt.job_id.to_string()
                && previous["fence"].as_u64().context("prior fence absent")?
                    < allocation.attempt.fence.get(),
            "duplicate or stale task attempt"
        );
        ensure!(
            matches!(input.operation, Operation::Train { .. }),
            "unfinished nontraining attempt requires explicit reconciliation"
        );
    }
    write_json(
        &active,
        &json!({"job_id":allocation.attempt.job_id.to_string(),"fence":allocation.attempt.fence.get(),"attempt":allocation.attempt.number.get()}),
    )?;
    let _deadline = DeadlineWatch::start(
        Arc::clone(&stop),
        task.deadline.as_unix_millis(),
        task.checkpoint_allowance_ms.unwrap_or(1000),
    );
    let verified = VerifiedTask {
        #[cfg(feature = "neural")]
        attempt: allocation.attempt,
        #[cfg(feature = "neural")]
        manifest: task.clone(),
        #[cfg(feature = "neural")]
        root: allocation.policy.root.clone(),
        operation: input.operation.clone(),
        stop,
        #[cfg(feature = "neural")]
        recovering,
    };
    let _scratch = ScratchGuard::start(allocation, &task, &directory)?;
    let outcome = match &input.operation {
        Operation::DatasetPublication {
            manifest, sha256, ..
        }
        | Operation::PackAndPublish {
            manifest, sha256, ..
        } => run_dataset_operation(allocation, &task, &input, manifest, sha256, &output),
        _ => run_operation(&allocation.policy, &verified, &output),
    };
    if let Err(error) = outcome {
        write_json(
            &directory.join("interrupted.json"),
            &json!({"manifest_sha256":sha,"error":error.to_string(),"checkpoint_requested":verified.stop.load(Ordering::Acquire)}),
        )?;
        return Err(error);
    }
    ensure!(
        output.is_file(),
        "numerical operation produced no output receipt"
    );
    fs::File::open(&output)?.sync_all()?;
    let result = json!({"schema":"jope.smartcluster_completion.v1","task_id":task.task_id,"receipt_id":task.expected_receipt_id,"manifest_sha256":sha,"native_attempt":allocation.attempt,"dataset_logical_attempt":input.operation.dataset_logical_attempt(),"output_path":input.output_receipt,"output_sha256":file_hash(&output)?});
    create_immutable(&complete, &serde_json::to_vec(&result)?)?;
    Ok(result)
}

fn run_dataset_operation(
    allocation: &Allocation,
    task: &TaskManifest,
    input: &InputManifest,
    manifest: &Path,
    sha256: &str,
    output: &Path,
) -> Result<()> {
    let policy = &allocation.policy;
    let executable = policy
        .datasets_executable
        .as_ref()
        .context("fixed dataset executable unbound")?;
    let executable_sha = policy
        .datasets_sha256
        .as_ref()
        .context("fixed dataset digest unbound")?;
    ensure!(
        executable.is_absolute() && fs::canonicalize(executable)? == *executable,
        "physical absolute dataset executable required"
    );
    let mut executable_file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(executable)?;
    let meta = executable_file.metadata()?;
    ensure!(
        meta.is_file() && meta.uid() == 0 && meta.mode() & 0o022 == 0,
        "root-owned immutable dataset executable required"
    );
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = executable_file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    ensure!(
        hex::encode(digest.finalize()) == *executable_sha,
        "dataset executable changed"
    );
    let manifest_path = resolve(&policy.root, manifest)?;
    let bytes = read_bounded(&manifest_path, Some(sha256))?;
    let request: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        request["schema"] == "jain.datasets_service_manifest.v1"
            && request["root"] == serde_json::to_value(&policy.root)?
            && request["task_id"] == task.task_id
            && request["attempt"]
                == input
                    .operation
                    .dataset_attempt(u64::from(allocation.attempt.number.get()))?
            && request["expected_receipt_id"] == task.expected_receipt_id
            && request["output_receipt"] == serde_json::to_value(&input.output_receipt)?,
        "dataset manifest differs from original native task/attempt/output"
    );
    let kind = request["operation"]["kind"]
        .as_str()
        .context("dataset operation absent")?;
    ensure!(
        match input.operation {
            Operation::PackAndPublish { .. } =>
                matches!(kind, "pack_and_publish" | "pack_continuation_and_publish"),
            Operation::DatasetPublication { .. } => matches!(
                kind,
                "winner_publish" | "corpus_publish" | "publish_pending"
            ),
            _ => false,
        },
        "dataset operation is not allowed by this service"
    );
    validate_dataset_packer(&policy.approved_dataset_packers, &request)?;
    if kind == "winner_publish" {
        crate::champions::recover_outbox(&policy.root)?;
    }
    // Fixed executable and fixed argument grammar. The original native cgroup
    // contains this child and drains it before any subsequent attempt.
    let status =
        std::process::Command::new(format!("/proc/self/fd/{}", executable_file.as_raw_fd()))
            .arg("service")
            .arg("--manifest")
            .arg(&manifest_path)
            .arg("--manifest-sha256")
            .arg(sha256)
            .stdin(std::process::Stdio::null())
            .status()?;
    ensure!(
        status.success(),
        "dataset authority refused operation: {status}"
    );
    let output_file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(output)?;
    ensure!(
        output_file.metadata()?.is_file() && output_file.metadata()?.len() <= 256 * 1024 * 1024,
        "dataset receipt exceeds 256 MiB bound or is not regular"
    );
    let mut output_bytes = Vec::new();
    output_file
        .take(256 * 1024 * 1024 + 1)
        .read_to_end(&mut output_bytes)?;
    ensure!(
        output_bytes.len() <= 256 * 1024 * 1024,
        "dataset receipt grew beyond bound"
    );
    let receipt: Value = serde_json::from_slice(&output_bytes)?;
    ensure!(
        receipt["schema"] == "jain.datasets_service_receipt.v1"
            && receipt["passed"] == true
            && receipt["task_id"] == task.task_id
            && receipt["attempt"] == request["attempt"]
            && receipt["receipt_id"] == task.expected_receipt_id
            && receipt["manifest_sha256"] == sha256
            && receipt["tool_sha256"] == *executable_sha,
        "dataset receipt identity or result differs"
    );
    let files = receipt["output_files"]
        .as_array()
        .context("dataset output closure missing")?;
    for file in files {
        let path = Path::new(
            file["path"]
                .as_str()
                .context("dataset output path absent")?,
        );
        ensure!(
            file_hash(&resolve(&policy.root, path)?)? == file["sha256"],
            "dataset output artifact changed"
        );
    }
    if matches!(kind, "pack_and_publish" | "pack_continuation_and_publish") {
        ensure!(
            !files.is_empty() && receipt["local_corpus_committed"] == true,
            "changed corpus has no durable local commit"
        );
    }
    Ok(())
}

fn validate_dataset_packer(approved: &BTreeMap<PathBuf, String>, request: &Value) -> Result<()> {
    let kind = request["operation"]["kind"]
        .as_str()
        .context("dataset operation absent")?;
    if matches!(kind, "pack_and_publish" | "pack_continuation_and_publish") {
        let packer = PathBuf::from(
            request["operation"]["packer"]["path"]
                .as_str()
                .context("fixed dataset CPU packer path absent")?,
        );
        let digest = request["operation"]["packer"]["sha256"]
            .as_str()
            .context("fixed dataset CPU packer digest absent")?;
        ensure!(
            approved.get(&packer).is_some_and(|sha| sha == digest),
            "dataset CPU packer is not independently allowlisted by worker policy"
        );
    }
    Ok(())
}

fn bind_native_fit_policy(policy: &WorkerPolicy) -> Result<()> {
    for (name, path) in [
        ("JOPE_CPU_EVALUATOR", &policy.cpu_evaluator),
        ("JOPE_FIT_AUTHORITY", &policy.fit_authority),
        ("JOPE_FIT_SERVICE_SOCKET", &policy.fit_service_socket),
        ("JOPE_FIT_WITNESS", &policy.fit_witness),
    ] {
        let path = path.as_ref().context("native fixed fit binding absent")?;
        ensure!(path.is_absolute(), "native fixed fit binding is relative");
        std::env::set_var(name, path);
    }
    let scope = policy
        .fit_scope
        .as_ref()
        .context("native fixed fit scope absent")?;
    ensure!(!scope.trim().is_empty(), "native fit scope empty");
    std::env::set_var("JOPE_FIT_SCOPE", scope);
    Ok(())
}

fn run_operation(policy: &WorkerPolicy, task: &VerifiedTask, receipt: &Path) -> Result<()> {
    ensure!(
        !task.stop.load(Ordering::Acquire),
        "native control already lost"
    );
    match task.operation() {
        Operation::DirectEvaluation { manifest, sha256 }
        | Operation::SeededContinuation { manifest, sha256 } => {
            bind_native_fit_policy(policy)?;
            crate::native_genome::execute(
                &policy.root,
                &crate::native_graph::Artifact {
                    path: manifest.clone(),
                    sha256: sha256.clone(),
                },
                if matches!(task.operation(), Operation::DirectEvaluation { .. }) {
                    crate::native_genome::Stage::Direct
                } else {
                    crate::native_genome::Stage::Continuation
                },
                receipt,
            )
        }
        Operation::Search {
            prepared,
            cut,
            output,
            evaluations,
            seconds,
            qualification,
            seeds,
            vocabulary,
            rng_seed,
        } => {
            ensure!(*evaluations > 0 && *seconds > 0, "empty search budget");
            // Search may only use the authority selected by the root-owned
            // policy. Never inherit an unrelated service or grant from the
            // daemon's environment when a policy binding is missing.
            let evaluator = policy
                .cpu_evaluator
                .as_ref()
                .context("native CPU evaluator unbound")?;
            let authority = policy
                .fit_authority
                .as_ref()
                .context("native fit authority unbound")?;
            let socket = policy
                .fit_service_socket
                .as_ref()
                .context("native fit socket unbound")?;
            let witness = policy
                .fit_witness
                .as_ref()
                .context("native fit witness unbound")?;
            let scope = policy
                .fit_scope
                .as_ref()
                .context("native fit scope unbound")?;
            ensure!(!scope.trim().is_empty(), "native fit scope empty");
            for path in [evaluator, authority, socket, witness] {
                ensure!(path.is_absolute(), "native fit binding must be absolute");
            }
            std::env::set_var("JOPE_CPU_EVALUATOR", evaluator);
            std::env::set_var("JOPE_FIT_AUTHORITY", authority);
            std::env::set_var("JOPE_FIT_SERVICE_SOCKET", socket);
            std::env::set_var("JOPE_FIT_WITNESS", witness);
            std::env::set_var("JOPE_FIT_SCOPE", scope);
            let seeds = seeds
                .as_ref()
                .map(|path| resolve(&policy.root, path))
                .transpose()?;
            let vocabulary = vocabulary
                .as_ref()
                .map(|path| resolve(&policy.root, path))
                .transpose()?;
            crate::gp::search_seeded_with_rng_in_root(
                &policy.root,
                &resolve(&policy.root, prepared)?,
                *cut,
                &resolve(&policy.root, output)?,
                crate::gp::SearchOptions {
                    evaluations: *evaluations,
                    seconds: *seconds,
                    qualification: *qualification,
                    seeds: seeds.as_deref(),
                    vocabulary: vocabulary.as_deref(),
                },
                *rng_seed,
            )?;
            write_json(
                receipt,
                &json!({"schema":"jope.native_gp_completion.v1","output":output,"evaluations_budget":evaluations,"seed":rng_seed,"artifacts":hash_directory(&resolve(&policy.root,output)?)?}),
            )
        }
        Operation::Pack {
            contexts,
            vocabulary,
            observations,
            output,
            qualification,
        } => {
            let observations = observations
                .iter()
                .map(|path| resolve(&policy.root, path))
                .collect::<Result<Vec<_>>>()?;
            crate::corpus_publication::build(
                &policy.root,
                crate::corpus_publication::PublicationSources {
                    contexts: &resolve(&policy.root, contexts)?,
                    vocab: &resolve(&policy.root, vocabulary)?,
                    observations: &observations,
                },
                &resolve(&policy.root, output)?,
                *qualification,
                None,
                None,
            )?;
            write_json(
                receipt,
                &json!({"schema":"jope.native_pack_completion.v1","artifacts":hash_directory(&resolve(&policy.root,output)?)?}),
            )
        }
        Operation::Verify => write_json(
            receipt,
            &json!({"schema":"jope.native_verification.v1","verified":true}),
        ),
        #[cfg(feature = "neural")]
        Operation::Train { .. } => crate::training::smartcluster_rank(task, receipt),
        #[cfg(feature = "neural")]
        Operation::Proposal { .. } => crate::proposals::smartcluster_rank(task, receipt),
        _ => anyhow::bail!("operation requires qualified neural runtime or is resident bootstrap"),
    }
}
pub(crate) fn hash_directory(path: &Path) -> Result<BTreeMap<PathBuf, String>> {
    let mut entries = BTreeMap::new();
    for entry in walkdir::WalkDir::new(path).follow_links(false) {
        let entry = entry?;
        ensure!(!entry.file_type().is_symlink(), "output symlink refused");
        if entry.file_type().is_file() {
            entries.insert(
                entry.path().strip_prefix(path)?.to_owned(),
                file_hash(entry.path())?,
            );
        }
    }
    ensure!(!entries.is_empty(), "numerical artifact directory is empty");
    Ok(entries)
}

/// Cooperative scratch accounting supplements the daemon's reservation. A quota
/// is not claimed: immutable numerical outputs remain on the guarded data root.
struct ScratchGuard {
    done: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl ScratchGuard {
    fn start(allocation: &Allocation, task: &TaskManifest, state: &Path) -> Result<Self> {
        let done = Arc::new(AtomicBool::new(false));
        if task.resources.scratch_bytes == 0 {
            return Ok(Self { done, thread: None });
        }
        let inherited_root = PathBuf::from(
            std::env::var_os("SCQ_SCRATCH_ROOT").context("native scratch root absent")?,
        );
        let inherited_budget: u64 = std::env::var("SCQ_SCRATCH_RESERVATION_BYTES")?.parse()?;
        ensure!(
            inherited_root == allocation.policy.scratch_root
                && inherited_budget == allocation.bootstrap.resources.scratch_bytes
                && task.resources.scratch_bytes <= inherited_budget,
            "native scratch reservation changed"
        );
        let allocation_root = inherited_root.join(allocation.attempt.job_id.to_string());
        let path = allocation_root.join(hash(&serde_json::to_vec(&(
            &task.run,
            task.generation,
            &task.task_id,
        ))?));
        fs::create_dir_all(&path)?;
        let within = |usage: ScratchUsage| {
            usage.task <= task.resources.scratch_bytes
                && usage.allocation <= inherited_budget
                && usage.root <= SCRATCH_ROOT_LIMIT
        };
        ensure!(
            within(scratch_usage(&inherited_root, &allocation_root, &path)?),
            "retained scratch exceeds task reservation"
        );
        std::env::set_var("TMPDIR", &path);
        let observed = Arc::clone(&done);
        let budget = task.resources.scratch_bytes;
        let state = state.to_owned();
        let thread = std::thread::spawn(move || {
            while !observed.load(Ordering::Acquire) {
                // One walk of the shared root per tick yields all three totals (the task and
                // allocation directories are inside it); walking each tree separately stat'ed
                // the task's files three times every 100 ms.
                let usage = scratch_usage(&inherited_root, &allocation_root, &path);
                if !usage.is_ok_and(|usage| {
                    usage.task <= budget
                        && usage.allocation <= inherited_budget
                        && usage.root <= SCRATCH_ROOT_LIMIT
                }) {
                    let _ = write_json(
                        &state.join("scratch-violation.json"),
                        &json!({"budget":budget,"path":path,"action":"terminate_and_preserve_attempt"}),
                    );
                    // SAFETY: _exit has no preconditions. Skipping destructors and flushes is
                    // intended: the over-budget task stops now, after a best-effort record.
                    unsafe {
                        libc::_exit(78);
                    }
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        Ok(Self {
            done,
            thread: Some(thread),
        })
    }
}
impl Drop for ScratchGuard {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
/// Byte cap on the whole inherited scratch root.
const SCRATCH_ROOT_LIMIT: u64 = 32 * 1024 * 1024 * 1024;

/// Regular-file bytes under the scratch root, its allocation directory and the task directory.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ScratchUsage {
    root: u64,
    allocation: u64,
    task: u64,
}

/// Sums [`ScratchUsage`] in one walk of `root` (`allocation` and `task` are inside it). A
/// symlink anywhere is refused. A file or directory that disappears between the directory read
/// and its `lstat` is a temporary file the task removed (TMPDIR points here), not a violation, so
/// it is skipped. The root, allocation and task directories themselves must exist (each
/// separate walk used to fail on a missing one): the root as a directory, possibly behind a link
/// as before (walkdir follows a root link), the other two as physical directories. Every other
/// error still fails closed. Entries renamed during a walk are counted on the next tick.
fn scratch_usage(root: &Path, allocation: &Path, task: &Path) -> Result<ScratchUsage> {
    let root_metadata =
        fs::metadata(root).with_context(|| format!("scratch root {}", root.display()))?;
    ensure!(root_metadata.is_dir(), "scratch root is not a directory");
    for directory in [allocation, task] {
        let metadata = fs::symlink_metadata(directory)
            .with_context(|| format!("scratch directory {}", directory.display()))?;
        ensure!(
            metadata.is_dir(),
            "scratch path is not a physical directory: {}",
            directory.display()
        );
    }
    let vanished = |error: &std::io::Error| error.kind() == std::io::ErrorKind::NotFound;
    let mut usage = ScratchUsage {
        root: 0,
        allocation: 0,
        task: 0,
    };
    for entry in walkdir::WalkDir::new(root).follow_links(false) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) if error.depth() > 0 && error.io_error().is_some_and(vanished) => continue,
            Err(error) => return Err(error.into()),
        };
        ensure!(!entry.file_type().is_symlink(), "scratch symlink rejected");
        if !entry.file_type().is_file() {
            continue;
        }
        let bytes = match entry.metadata() {
            Ok(metadata) => metadata.len(),
            Err(error) if error.io_error().is_some_and(vanished) => continue,
            Err(error) => return Err(error.into()),
        };
        let add = |total: u64| total.checked_add(bytes).context("scratch size overflow");
        usage.root = add(usage.root)?;
        if entry.path().starts_with(allocation) {
            usage.allocation = add(usage.allocation)?;
            if entry.path().starts_with(task) {
                usage.task = add(usage.task)?;
            }
        }
    }
    Ok(usage)
}

struct DeadlineWatch {
    done: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl DeadlineWatch {
    fn start(stop: Arc<AtomicBool>, deadline: u64, allowance: u64) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&done);
        let remaining =
            Duration::from_millis(deadline.saturating_sub(resident::now_ms().unwrap_or(u64::MAX)));
        let started = std::time::Instant::now();
        let thread = std::thread::spawn(move || {
            while !observed.load(Ordering::Acquire) {
                let now = resident::now_ms().unwrap_or(u64::MAX);
                if now >= deadline || started.elapsed() >= remaining {
                    stop.store(true, Ordering::Release);
                }
                if now >= deadline.saturating_add(allowance)
                    || started.elapsed()
                        >= remaining.saturating_add(Duration::from_millis(allowance))
                {
                    // SAFETY: _exit has no preconditions. Skipping destructors and flushes is
                    // intended: past deadline plus allowance the process is ended at once.
                    unsafe {
                        libc::_exit(76);
                    }
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        Self {
            done,
            thread: Some(thread),
        }
    }
}
impl Drop for DeadlineWatch {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum ControlStopReason {
    EndOfFile,
    FrameReadError,
    Checkpoint,
    MalformedNamedControl,
    UnknownControl,
    QueueFull,
    QueueDisconnected,
}

#[derive(Clone, Copy)]
struct ControlStopDeadline {
    instant: std::time::Instant,
    monotonic_ns: Option<u64>,
}

fn stop_control_with(
    stop: &AtomicBool,
    allowance: Duration,
    watchdog: impl FnOnce(std::time::Instant),
    persist: impl FnOnce(ControlStopDeadline) -> Result<()>,
) -> Result<()> {
    let now = std::time::Instant::now();
    let instant = now.checked_add(allowance).unwrap_or(now);
    stop.store(true, Ordering::Release);
    // Arm independently of all diagnostic filesystem I/O.
    watchdog(instant);
    let monotonic_ns = crate::host_os::monotonic_ns().ok().and_then(|n| {
        u64::try_from(
            instant
                .saturating_duration_since(std::time::Instant::now())
                .as_nanos(),
        )
        .ok()
        .and_then(|v| n.checked_add(v))
    });
    persist(ControlStopDeadline {
        instant,
        monotonic_ns,
    })
}

fn record_control_stop(
    state_root: &Path,
    attempt: FencedAttempt,
    sequence: u64,
    reason: ControlStopReason,
    deadline: ControlStopDeadline,
) -> Result<()> {
    let value = json!({"schema":"jope.smartcluster_control_stop.v1","attempt":attempt,
        "sequence":sequence,"reason":reason,"observed_at_unix_ms":resident::now_ms()?,
        "checkpoint_deadline_monotonic_ns":deadline.monotonic_ns,
        "checkpoint_remaining_ms":deadline.instant.saturating_duration_since(std::time::Instant::now()).as_millis()});
    ensure!(
        serde_json::to_vec(&value)?.len() <= 4096,
        "control stop receipt exceeds bound"
    );
    write_immutable_record(
        &state_root.join(format!(
            "control-stop-{}-{}.json",
            attempt.job_id, attempt.fence
        )),
        &value,
    )
}

pub fn serve(path: &Path, sha: &str) -> Result<()> {
    let policy = load_policy()?;
    let mut allocation = authenticate(policy, path, sha)?;
    let (_, input) = load_task(&allocation.policy, path, sha)?;
    let stop = Arc::new(AtomicBool::new(false));
    let mut reader = allocation.stream.try_clone()?;
    let observed = Arc::clone(&stop);
    let attempt = allocation.attempt;
    let control_state_root = allocation.policy.state_root.clone();
    let allowance = allocation.bootstrap.checkpoint_allowance_ms.unwrap_or(1000);
    let (send, receive) = mpsc::sync_channel::<SubtaskCommand>(1);
    std::thread::spawn(move || {
        let mut previous = 0;
        let reason = loop {
            let frame = match read_frame(&mut reader) {
                Ok(frame) => frame,
                Err(error) => {
                    break if error
                        .downcast_ref::<std::io::Error>()
                        .is_some_and(|e| e.kind() == std::io::ErrorKind::UnexpectedEof)
                    {
                        ControlStopReason::EndOfFile
                    } else {
                        ControlStopReason::FrameReadError
                    }
                }
            };
            if frame.attempt != attempt
                || frame.sequence <= previous
                || frame.version != ProtocolVersion::CURRENT
            {
                continue;
            }
            previous = frame.sequence;
            match frame.control {
                ControlMessage::Checkpoint { .. } => {
                    break ControlStopReason::Checkpoint;
                }
                ControlMessage::Application { name, payload } if name == CONTROL_NAME => {
                    match serde_json::from_slice(&payload) {
                        Ok(command) => match send.try_send(command) {
                            Ok(()) => {}
                            Err(mpsc::TrySendError::Full(_)) => break ControlStopReason::QueueFull,
                            Err(mpsc::TrySendError::Disconnected(_)) => {
                                break ControlStopReason::QueueDisconnected
                            }
                        },
                        Err(_) => {
                            break ControlStopReason::MalformedNamedControl;
                        }
                    }
                }
                _ => {
                    break ControlStopReason::UnknownControl;
                }
            }
        };
        if stop_control_with(
            &observed,
            Duration::from_millis(allowance),
            |deadline| {
                std::thread::spawn(move || {
                    std::thread::sleep(
                        deadline.saturating_duration_since(std::time::Instant::now()),
                    );
                    // SAFETY: _exit has no preconditions. Skipping destructors and flushes is
                    // intended: a worker still running past its checkpoint allowance is killed.
                    unsafe {
                        libc::_exit(76);
                    }
                });
            },
            |deadline| {
                record_control_stop(&control_state_root, attempt, previous, reason, deadline)
            },
        )
        .is_err()
        {
            eprintln!(
                "control stop diagnostic could not be persisted; numerical admission stopped"
            );
        }
    });
    let mut sequence = 0;
    report(
        &mut allocation.stream,
        attempt,
        &mut sequence,
        WorkerReport::Ready,
    )?;
    if !matches!(input.operation, Operation::Resident) {
        let result = execute(&allocation, path, sha, Arc::clone(&stop))?;
        return report(
            &mut allocation.stream,
            attempt,
            &mut sequence,
            result_progress(&result)?,
        );
    }
    while !stop.load(Ordering::Acquire) {
        let command = match receive.recv_timeout(Duration::from_millis(100)) {
            Ok(command) => command,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(_) => break,
        };
        match execute(
            &allocation,
            &command.manifest,
            &command.manifest_sha256,
            Arc::clone(&stop),
        ) {
            Ok(result) => report(
                &mut allocation.stream,
                attempt,
                &mut sequence,
                result_progress(&result)?,
            )?,
            Err(error) => {
                if stop.load(Ordering::Acquire) {
                    break;
                }
                report(
                    &mut allocation.stream,
                    attempt,
                    &mut sequence,
                    WorkerReport::Failure {
                        class: jain_smartcluster::FailureClass::Application,
                        message: format!("{error:#}").chars().take(4000).collect(),
                    },
                )?;
                return Err(error);
            }
        }
    }
    write_json(
        &allocation.policy.state_root.join(format!(
            "recovery-{}-{}.json",
            attempt.job_id, attempt.fence
        )),
        &json!({"schema":"jope.smartcluster_recovery.v1","attempt":attempt,"recovery_state_fsynced":true,"numerical_completion_requires_verified_task_receipt":true}),
    )?;
    Ok(())
}
fn result_progress(result: &Value) -> Result<WorkerReport> {
    Ok(WorkerReport::Progress(StructuredProgress {
        phase: "durable_subtask_result".into(),
        completed: 1,
        total: Some(1),
        message: None,
        fields: BTreeMap::from([("receipt".into(), serde_json::to_string(result)?)]),
    }))
}

#[cfg(test)]
mod tests {
    #[test]
    fn blocked_control_diagnostic_cannot_delay_stop_or_original_watchdog() {
        use super::*;
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let (began_send, began) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let (expired_send, expired) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            stop_control_with(
                &flag,
                Duration::from_millis(30),
                |deadline| {
                    std::thread::spawn(move || {
                        std::thread::sleep(
                            deadline.saturating_duration_since(std::time::Instant::now()),
                        );
                        expired_send.send(()).unwrap();
                    });
                },
                |_| {
                    began_send.send(()).unwrap();
                    wait.recv().unwrap();
                    anyhow::bail!("injected diagnostic storage failure")
                },
            )
        });
        began.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(stop.load(Ordering::Acquire));
        expired.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(
            !worker.is_finished(),
            "diagnostic remains blocked while watchdog has already fired"
        );
        release.send(()).unwrap();
        assert!(worker.join().unwrap().is_err());
    }

    #[test]
    fn control_stop_records_only_bounded_reason_and_preserves_original_attempt() {
        use super::*;
        for (reason, label) in [
            (ControlStopReason::EndOfFile, "end_of_file"),
            (ControlStopReason::FrameReadError, "frame_read_error"),
            (ControlStopReason::Checkpoint, "checkpoint"),
            (
                ControlStopReason::MalformedNamedControl,
                "malformed_named_control",
            ),
            (ControlStopReason::UnknownControl, "unknown_control"),
            (ControlStopReason::QueueFull, "queue_full"),
            (ControlStopReason::QueueDisconnected, "queue_disconnected"),
        ] {
            let root = tempfile::tempdir().unwrap();
            let attempt: FencedAttempt = serde_json::from_value(
                json!({"job_id":jain_smartcluster::JobId::new(),"number":1,"fence":9}),
            )
            .unwrap();
            let stop = AtomicBool::new(false);
            stop_control_with(
                &stop,
                Duration::from_secs(120),
                |_| {},
                |deadline| record_control_stop(root.path(), attempt, 17, reason, deadline),
            )
            .unwrap();
            assert!(stop.load(Ordering::Acquire));
            let path = root.path().join(format!(
                "control-stop-{}-{}.json",
                attempt.job_id, attempt.fence
            ));
            let bytes = fs::read(&path).unwrap();
            assert!(bytes.len() <= 4096);
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["reason"], label);
            assert_eq!(value["attempt"], serde_json::to_value(attempt).unwrap());
            assert_eq!(value["sequence"], 17);
            assert_eq!(value.as_object().unwrap().len(), 7);
            assert!(stop_control_with(
                &stop,
                Duration::from_secs(120),
                |_| {},
                |deadline| record_control_stop(
                    root.path(),
                    attempt,
                    18,
                    ControlStopReason::UnknownControl,
                    deadline
                )
            )
            .is_err());
            assert_eq!(
                fs::read(path).unwrap(),
                bytes,
                "original stop evidence must not be overwritten"
            );
        }
        let root = tempfile::tempdir().unwrap();
        let bad = root.path().join("regular-file");
        fs::write(&bad, b"preserved").unwrap();
        let attempt: FencedAttempt = serde_json::from_value(
            json!({"job_id":jain_smartcluster::JobId::new(),"number":1,"fence":1}),
        )
        .unwrap();
        let stop = AtomicBool::new(false);
        assert!(stop_control_with(
            &stop,
            Duration::from_secs(120),
            |_| {},
            |deadline| record_control_stop(
                &bad,
                attempt,
                0,
                ControlStopReason::EndOfFile,
                deadline
            )
        )
        .is_err());
        assert!(
            stop.load(Ordering::Acquire),
            "failed diagnostic storage must not keep numerical admission open"
        );
    }

    #[test]
    fn logical_dataset_retry_preserves_original_operation_and_legacy_binding() {
        let mut operation:super::Operation=serde_json::from_value(serde_json::json!({"kind":"PackAndPublish","manifest":"m.json","sha256":"a".repeat(64)})).unwrap();
        assert_eq!(operation.dataset_attempt(2).unwrap(), 2);
        let legacy = serde_json::to_vec(&operation).unwrap();
        if let super::Operation::PackAndPublish {
            logical_attempt, ..
        } = &mut operation
        {
            *logical_attempt = Some(1)
        }
        assert_eq!(operation.dataset_attempt(2).unwrap(), 1);
        assert_ne!(legacy, serde_json::to_vec(&operation).unwrap());
        if let super::Operation::PackAndPublish {
            logical_attempt, ..
        } = &mut operation
        {
            *logical_attempt = Some(0)
        }
        assert!(operation.dataset_attempt(2).is_err());
    }
    #[test]
    fn dataset_manifest_cannot_select_an_unapproved_cpu_packer() {
        let approved = std::collections::BTreeMap::from([(
            std::path::PathBuf::from("/fixed/packer"),
            "a".repeat(64),
        )]);
        let mut request = serde_json::json!({"operation":{"kind":"pack_continuation_and_publish","packer":{"path":"/fixed/packer","sha256":"a".repeat(64)}}});
        assert!(super::validate_dataset_packer(&approved, &request).is_ok());
        request["operation"]["packer"]["path"] = serde_json::json!("/tmp/unreviewed");
        assert!(super::validate_dataset_packer(&approved, &request).is_err());
        request["operation"]["packer"]["path"] = serde_json::json!("/fixed/packer");
        request["operation"]["packer"]["sha256"] = serde_json::json!("b".repeat(64));
        assert!(super::validate_dataset_packer(&approved, &request).is_err());
    }

    use super::*;
    fn task() -> TaskManifest {
        TaskManifest {
            schema: jain_smartcluster::jope::MANIFEST_SCHEMA.into(),
            run: "xbabe2".into(),
            generation: 1,
            task_id: "train".into(),
            kind: TaskKind::ResidentGpu,
            input_manifest_sha256: "a".repeat(64),
            dependencies: BTreeMap::new(),
            parent_checkpoint_sha256: Some("b".repeat(64)),
            corpus_sha256: Some("c".repeat(64)),
            idempotency_key: jain_smartcluster::IdempotencyKey::new("train").unwrap(),
            deadline: Timestamp::from(100000),
            retry_allowance: 2,
            resources: jain_smartcluster::ResourceRequest::cpu_and_ram(1000, 1024),
            expected_receipt_id: "result".into(),
            checkpoint_allowance_ms: Some(30000),
        }
    }
    #[test]
    fn duplicate_dispatch_preserves_bytes_and_conflict_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("intent");
        let command = SubtaskCommand {
            manifest: "/task.json".into(),
            manifest_sha256: "a".repeat(64),
        };
        freeze_dispatch_intent(&path, &command).unwrap();
        let before = fs::read(&path).unwrap();
        freeze_dispatch_intent(&path, &command).unwrap();
        let changed = SubtaskCommand {
            manifest_sha256: "b".repeat(64),
            ..command
        };
        assert!(freeze_dispatch_intent(&path, &changed).is_err());
        assert_eq!(fs::read(path).unwrap(), before);
    }
    #[test]
    fn subtask_cannot_escape_run_resources_or_deadline() {
        let parent = task();
        within_allocation(&parent, &parent).unwrap();
        let mut child = parent.clone();
        child.run = "different".into();
        assert!(within_allocation(&child, &parent).is_err());
        child = parent.clone();
        child.resources.scratch_bytes = 1;
        assert!(within_allocation(&child, &parent).is_err());
        child = parent.clone();
        child.checkpoint_allowance_ms = Some(30001);
        assert!(within_allocation(&child, &parent).is_err());
        child = parent.clone();
        child.deadline = Timestamp::from(100001);
        assert!(within_allocation(&child, &parent).is_err());
        child = parent.clone();
        child.resources.ram.hard_bytes += 1;
        assert!(within_allocation(&child, &parent).is_err());
    }
    #[test]
    fn original_optimizer_window_is_checked_before_numerical_work() {
        let paths = ["corpus", "run", "launch", "initialize", "policy"];
        let mut input = InputManifest {
            schema: "jope.smartcluster_input.v1".into(),
            root: "/tmp".into(),
            operation: Operation::Train {
                corpus: paths[0].into(),
                run: paths[1].into(),
                launch: paths[2].into(),
                micro: 256,
                checkpoint_steps: 100,
                initialize: paths[3].into(),
                replay_policy: paths[4].into(),
                initial_optimizer_step: 60000,
                requested_step_increment: 3000,
                target_optimizer_step: 63000,
                continuation_mode: None,
            },
            closure: None,
            input_artifacts: paths
                .iter()
                .map(|p| (PathBuf::from(p), "a".repeat(64)))
                .collect(),
            output_receipt: "result".into(),
        };
        let mut allocation = task();
        allocation.resources.scratch_bytes = 1024;
        validate_operation(&allocation, &input).unwrap();
        if let Operation::Train {
            target_optimizer_step,
            ..
        } = &mut input.operation
        {
            *target_optimizer_step = 66000;
        }
        assert!(validate_operation(&task(), &input).is_err());
    }
    #[test]
    fn training_closure_refuses_non_admitted_contexts_by_closure_key() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("corpus")).unwrap();
        let mut input = InputManifest {
            schema: "jope.smartcluster_input.v1".into(),
            root: dir.path().to_path_buf(),
            operation: Operation::Train {
                corpus: "corpus".into(),
                run: "run".into(),
                launch: "launch".into(),
                micro: 256,
                checkpoint_steps: 100,
                initialize: "initialize".into(),
                replay_policy: "policy".into(),
                initial_optimizer_step: 60000,
                requested_step_increment: 3000,
                target_optimizer_step: 63000,
                continuation_mode: None,
            },
            closure: None,
            input_artifacts: BTreeMap::new(),
            output_receipt: "result".into(),
        };
        assert!(verify_training_closure(dir.path(), &input)
            .unwrap_err()
            .to_string()
            .contains("not in dependency closure"));
        for (partition, admitted) in [("meta_validation", true), ("meta_test", false)] {
            let contexts = json!([{"partition": "meta_train"}, {"partition": partition}]);
            let index = json!({"schema": "parallel.teaching_shards.v1", "contexts": contexts});
            let index = serde_json::to_vec(&index).unwrap();
            let entry = json!({"path": "corpus/index.json", "sha256": hash(&index)});
            let mut closure = serde_json::to_vec(&entry).unwrap();
            closure.push(b'\n');
            fs::write(dir.path().join("corpus/index.json"), &index).unwrap();
            fs::write(dir.path().join("closure.jsonl"), &closure).unwrap();
            input.input_artifacts = read_closure(
                dir.path(),
                &ClosureManifest {
                    path: "closure.jsonl".into(),
                    sha256: hash(&closure),
                },
            )
            .unwrap();
            let result = verify_training_closure(dir.path(), &input);
            if admitted {
                result.unwrap();
            } else {
                assert!(result
                    .unwrap_err()
                    .to_string()
                    .contains("non-admitted context partition at position 1"));
            }
        }
    }
    #[test]
    fn unsupported_service_and_unbound_inputs_fail_closed() {
        let mut input = InputManifest {
            schema: "jope.smartcluster_input.v1".into(),
            root: "/tmp".into(),
            operation: Operation::Verify,
            closure: None,
            input_artifacts: BTreeMap::new(),
            output_receipt: "result".into(),
        };
        assert!(validate_operation(&task(), &input).is_err());
        input.operation = Operation::Proposal {
            preparation_overlay: None,
            corpus: "corpus".into(),
            checkpoint: "checkpoint".into(),
            output: "output".into(),
            cuts: 1,
            cut_start: 0,
            shuffled: false,
            champion_snapshot: "champion".into(),
        };
        assert!(validate_operation(&task(), &input).is_err());
    }
    #[test]
    fn scratch_accounting_rejects_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("bytes"), [0u8; 4096]).unwrap();
        fs::create_dir_all(dir.path().join("a/t")).unwrap();
        let usage = |root: &Path| scratch_usage(root, &root.join("a"), &root.join("a/t"));
        assert_eq!(
            usage(dir.path()).unwrap(),
            ScratchUsage {
                root: 4096,
                allocation: 0,
                task: 0
            }
        );
        std::os::unix::fs::symlink("/tmp", dir.path().join("escape")).unwrap();
        assert!(usage(dir.path()).is_err());
    }
    #[test]
    fn scratch_accounting_sums_nested_buckets_in_one_walk_and_needs_its_root() {
        let dir = tempfile::tempdir().unwrap();
        let (root, allocation) = (dir.path(), dir.path().join("job"));
        let task = allocation.join("task");
        fs::create_dir_all(task.join("deep")).unwrap();
        fs::write(root.join("other"), [0u8; 10]).unwrap();
        fs::write(allocation.join("sibling"), [0u8; 100]).unwrap();
        fs::write(task.join("deep/file"), [0u8; 1000]).unwrap();
        // A same-prefix name next to the task directory is not inside it.
        fs::write(allocation.join("task-not"), [0u8; 7]).unwrap();
        assert_eq!(
            scratch_usage(root, &allocation, &task).unwrap(),
            ScratchUsage {
                root: 1117,
                allocation: 1107,
                task: 1000
            }
        );
        assert!(scratch_usage(&root.join("absent"), &allocation, &task).is_err());
        assert!(scratch_usage(root, &allocation, &allocation.join("absent")).is_err());
        assert!(scratch_usage(root, &root.join("absent"), &task).is_err());
    }
    #[test]
    fn closure_fifo_is_rejected_without_waiting_for_writer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fifo");
        let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        // SAFETY: `name` is a CString that outlives the call, so the path is NUL-terminated.
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        let started = std::time::Instant::now();
        assert!(read_closure(
            dir.path(),
            &ClosureManifest {
                path: "fifo".into(),
                sha256: "a".repeat(64)
            }
        )
        .is_err());
        assert!(started.elapsed() < Duration::from_secs(1));
    }
    #[test]
    fn large_streamed_closure_preserves_every_identity_and_refuses_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("closure.jsonl");
        let mut bytes = Vec::new();
        for n in 0..12000 {
            serde_json::to_writer(&mut bytes,&json!({"path":format!("historical-observations/shards/shard-{n:08}.json"),"sha256":"a".repeat(64)})).unwrap();
            bytes.push(b'\n');
        }
        assert!(bytes.len() > FRAME_LIMIT);
        fs::write(&path, &bytes).unwrap();
        let closure = ClosureManifest {
            path: "closure.jsonl".into(),
            sha256: hash(&bytes),
        };
        assert_eq!(read_closure(dir.path(), &closure).unwrap().len(), 12000);
        bytes.extend_from_slice(b"{\"path\":\"historical-observations/shards/shard-00000000.json\",\"sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}\n");
        fs::write(&path, &bytes).unwrap();
        assert!(read_closure(dir.path(), &closure).is_err());
        assert!(read_closure(
            dir.path(),
            &ClosureManifest {
                sha256: hash(&bytes),
                ..closure
            }
        )
        .is_err());
    }
    #[test]
    fn control_descriptor_is_absent_after_child_exec() {
        let (stream, _) = UnixStream::pair().unwrap();
        let fd = stream.as_raw_fd();
        // SAFETY: `stream` stays alive for the whole test, so `fd` is open; F_SETFD takes an
        // integer flag and touches no memory.
        assert_eq!(unsafe { libc::fcntl(fd, libc::F_SETFD, 0) }, 0);
        let retained = retain_control_descriptor(fd).unwrap();
        assert!(
            // SAFETY: `retained` is a live UnixStream, so its descriptor is open; F_GETFD takes
            // no third argument and only reads the descriptor flags.
            unsafe { libc::fcntl(retained.as_raw_fd(), libc::F_GETFD) } & libc::FD_CLOEXEC != 0
        );
        let status = std::process::Command::new("sh")
            .args(["-c", "test ! -e /proc/self/fd/$1", "sh", &fd.to_string()])
            .status()
            .unwrap();
        assert!(status.success());
    }
    #[test]
    fn atomic_record_publication_preserves_old_and_interrupted_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("record");
        let interrupted = dir.path().join(".immutable-pending-crashed");
        fs::write(&interrupted, b"partial").unwrap();
        assert!(!target.exists());
        create_immutable(&target, b"complete").unwrap();
        create_immutable(&target, b"complete").unwrap();
        assert!(create_immutable(&target, b"conflict").is_err());
        assert_eq!(fs::read(&target).unwrap(), b"complete");
        assert_eq!(fs::read(interrupted).unwrap(), b"partial");
        assert!(fs::read_dir(dir.path()).unwrap().count() >= 3);
    }
    #[test]
    fn concurrent_immutable_publish_never_exposes_partial_result() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("record");
        let bytes = vec![42u8; FRAME_LIMIT];
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let target = &target;
                let bytes = &bytes;
                scope.spawn(move || create_immutable(target, bytes).unwrap());
            }
            while !target.exists() {
                std::thread::yield_now();
            }
            assert_eq!(fs::read(&target).unwrap(), bytes);
        });
    }
    #[test]
    fn truncated_control_frame_is_not_completion() {
        let (mut writer, mut reader) = UnixStream::pair().unwrap();
        writer.write_all(&8u32.to_be_bytes()).unwrap();
        writer.write_all(&[1, 2]).unwrap();
        drop(writer);
        assert!(read_frame(&mut reader).is_err());
    }
}
