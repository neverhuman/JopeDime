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

fn available_storage_bytes(path: &Path) -> Result<u64> {
    let stats = nix::sys::statvfs::statvfs(path).context("storage measurement unavailable")?;
    Ok(storage_bytes_from_stats(stats))
}

fn storage_bytes(blocks_available: u64, fragment_size: u64) -> u64 {
    blocks_available.saturating_mul(fragment_size)
}

fn storage_bytes_from_stats(stats: nix::sys::statvfs::Statvfs) -> u64 {
    storage_bytes(stats.blocks_available(), stats.fragment_size())
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
        ensure!(
            available_storage_bytes(path)? >= *reserve,
            "storage reserve blocks subtask"
        );
    }
    Ok(())
}

#[cfg(test)]
mod storage_measurement_tests {
    use super::{available_storage_bytes, storage_bytes, storage_bytes_from_stats};

    #[test]
    fn storage_byte_calculation_uses_available_blocks_and_saturates() {
        assert_eq!(storage_bytes(3, 4096), 12_288);
        assert_eq!(storage_bytes(u64::MAX, 2), u64::MAX);
    }

    #[test]
    fn available_storage_uses_available_blocks_and_fragment_size() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let stats = nix::sys::statvfs::statvfs(directory.path()).expect("filesystem stats");

        assert_eq!(
            storage_bytes_from_stats(stats),
            stats
                .blocks_available()
                .saturating_mul(stats.fragment_size())
        );
    }

    #[test]
    fn unavailable_storage_keeps_admission_error_context() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let missing = directory.path().join("missing");

        let error = available_storage_bytes(&missing).expect_err("missing path is unavailable");

        assert!(format!("{error:#}").contains("storage measurement unavailable"));
    }
}

mod execution;
pub use execution::{freeze_dispatch_intent, serve};

#[cfg(test)]
mod tests;
