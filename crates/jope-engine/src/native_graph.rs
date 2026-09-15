//! JOPE dependency eligibility and durable result reconciliation.
//! Resource admission, queue placement and process recovery belong to SmartCluster.
use crate::{
    manifest::hash,
    smartcluster_worker::{InputManifest, Operation, SubtaskCommand},
};
use anyhow::{ensure, Context, Result};
use jain_smartcluster::{
    jope::{self, TaskManifest},
    Client, FencedAttempt, JobId, JobSpec, JobState, ProjectId, TenantId, Timestamp,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::{Component, Path, PathBuf},
};

mod transport;
use transport::Transport;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Delivery {
    NativeJob,
    Resident { job_id: JobId },
}
/// Written only after the scientific planner freezes exact task and predecessor identities.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Admission {
    pub schema: String,
    pub manifest: Artifact,
    pub predecessors: BTreeMap<String, Artifact>,
    pub delivery: Delivery,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedResult {
    pub schema: String,
    pub admission_sha256: String,
    pub task_manifest_sha256: String,
    pub task_id: String,
    pub run: String,
    pub generation: u64,
    pub receipt_id: String,
    pub native_attempt: FencedAttempt,
    pub worker_completion: Artifact,
    pub output: Artifact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Submitted {
    admission_sha256: String,
    job_id: JobId,
}
#[derive(Default, Serialize)]
pub struct Tick {
    pub verified: usize,
    pub submitted: usize,
    pub waiting_dependencies: usize,
    pub waiting_execution: usize,
    pub failed: usize,
    pub blocked: usize,
}
pub struct Graph {
    root: PathBuf,
    directory: PathBuf,
    worker_state: PathBuf,
    tenant: TenantId,
    project: ProjectId,
    _lock: fs::File,
    read_only: bool,
}
fn relative(root: &Path, path: &Path) -> Result<PathBuf> {
    ensure!(
        !path.as_os_str().is_empty() && !path.is_absolute(),
        "relative graph artifact required"
    );
    ensure!(
        path.components().all(|c| matches!(c, Component::Normal(_))),
        "unsafe graph artifact path"
    );
    let mut current = root.to_path_buf();
    for part in path.components() {
        current.push(part);
        let m = fs::symlink_metadata(&current)?;
        ensure!(
            !m.file_type().is_symlink(),
            "graph artifact symlink refused"
        );
    }
    Ok(current)
}
fn read(path: &Path) -> Result<Vec<u8>> {
    read_limit(path, 1024 * 1024)
}
fn read_limit(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() <= limit,
        "graph record type/size invalid"
    );
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= limit, "graph record grew past bound");
    Ok(bytes)
}
fn checked(root: &Path, artifact: &Artifact) -> Result<Vec<u8>> {
    checked_limit(root, artifact, 1024 * 1024)
}
fn checked_limit(root: &Path, artifact: &Artifact, limit: u64) -> Result<Vec<u8>> {
    let bytes = read_limit(&relative(root, &artifact.path)?, limit)?;
    ensure!(
        hash(&bytes) == artifact.sha256,
        "graph artifact changed: {}",
        artifact.path.display()
    );
    Ok(bytes)
}
fn directory(path: &Path) -> Result<()> {
    if !path.exists() {
        let parent = path.parent().context("graph directory parent")?;
        directory(parent)?;
        match fs::create_dir(path) {
            Ok(()) => (),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(e) => return Err(e.into()),
        }
        fs::File::open(parent)?.sync_all()?;
    }
    ensure!(
        fs::symlink_metadata(path)?.is_dir(),
        "graph directory must be physical"
    );
    fs::File::open(path)?.sync_all()?;
    Ok(())
}
fn immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure!(
        bytes.len() <= 1024 * 1024,
        "graph record exceeds durable replay read bound"
    );
    let parent = path.parent().context("graph record parent")?;
    directory(parent)?;
    let (mut file, temporary) = tempfile::Builder::new()
        .prefix(".graph-pending-")
        .tempfile_in(parent)?
        .keep()?;
    file.write_all(bytes)?;
    file.sync_all()?;
    match crate::native_persistence::publish_staged(&temporary, path) {
        Ok(()) => (),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            ensure!(read(path)? == bytes, "conflicting immutable graph record")
        }
        Err(e) => return Err(e.into()),
    }
    fs::File::open(parent)?.sync_all()?;
    crate::native_persistence::discard_staging(&temporary)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}
impl Graph {
    pub fn open(
        root: &Path,
        directory_path: &Path,
        worker_state: &Path,
        tenant: TenantId,
        project: ProjectId,
    ) -> Result<Self> {
        ensure!(
            root.is_absolute() && fs::canonicalize(root)? == root,
            "canonical physical graph root required"
        );
        for state_path in [directory_path, worker_state] {
            let suffix = state_path
                .strip_prefix(root)
                .context("graph state outside root")?;
            ensure!(
                !suffix.as_os_str().is_empty()
                    && suffix
                        .components()
                        .all(|c| matches!(c, Component::Normal(_))),
                "physical graph state suffix required"
            );
            let mut part = root.to_path_buf();
            for component in suffix.components() {
                part.push(component);
                match fs::symlink_metadata(&part) {
                    Ok(meta) => ensure!(
                        meta.is_dir() && !meta.file_type().is_symlink(),
                        "graph state ancestor is not physical"
                    ),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
                    Err(e) => return Err(e.into()),
                }
            }
        }
        directory(directory_path)?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(directory_path.join("coordinator.lock"))?;
        ensure!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "another graph coordinator owns this run"
        );
        Ok(Self {
            root: root.to_owned(),
            directory: directory_path.to_owned(),
            worker_state: worker_state.to_owned(),
            tenant,
            project,
            _lock: lock,
            read_only: false,
        })
    }
    /// A verifier can read immutable dispatch evidence without acquiring the
    /// coordinator writer lock. This object refuses every mutation entrypoint.
    pub(crate) fn read_only(
        root: &Path,
        directory: &Path,
        worker_state: &Path,
        tenant: TenantId,
        project: ProjectId,
    ) -> Result<Self> {
        ensure!(
            root.is_absolute() && fs::canonicalize(root)? == root,
            "physical verifier root required"
        );
        for path in [directory, worker_state] {
            ensure!(
                relative(root, path.strip_prefix(root)?)? == *path && path.is_dir(),
                "existing physical verifier state required"
            );
        }
        Ok(Self {
            root: root.into(),
            directory: directory.into(),
            worker_state: worker_state.into(),
            tenant,
            project,
            _lock: fs::File::open(directory)?,
            read_only: true,
        })
    }
    /// Read a committed result only after revalidating its original native dispatch and dependency closure.
    pub async fn verified(&self, client: &Client, artifact: &Artifact) -> Result<VerifiedResult> {
        let pinned = Transport::load(self, client)?;
        let client = &pinned;
        let result: VerifiedResult = serde_json::from_slice(&checked(&self.root, artifact)?)?;
        self.validate_predecessor(client, artifact, &result, 0, &mut BTreeSet::new())
            .await?;
        Ok(result)
    }
    /// A terminal allocation without a numerical receipt is explicit failure,
    /// never numerical completion and never an implicit resubmission.
    pub(crate) async fn failed(
        &self,
        client: &Client,
        admission: &Admission,
    ) -> Result<Option<Artifact>> {
        let pinned = Transport::load(self, client)?;
        let client = &pinned;
        if admission.delivery != Delivery::NativeJob {
            return Ok(None);
        }
        let task: TaskManifest =
            serde_json::from_slice(&checked(&self.root, &admission.manifest)?)?;
        let directory = self.directory.join("tasks").join(Self::task_key(&task)?);
        if directory.join("verified.json").exists() {
            return Ok(None);
        }
        let path = directory.join("failed.json");
        if !path.exists() {
            return Ok(None);
        }
        let data = read(&path)?;
        let failure: Value = serde_json::from_slice(&data)?;
        ensure!(
            failure["schema"] == "jope.native_graph_failure.v1",
            "unsupported terminal failure receipt"
        );
        let original: jain_smartcluster::JobRecord =
            serde_json::from_value(failure["job"].clone())?;
        let submitted: Submitted =
            serde_json::from_slice(&read(&directory.join("submitted.json"))?)?;
        ensure!(
            submitted.admission_sha256 == hash(&serde_json::to_vec(admission)?)
                && submitted.job_id == original.id
                && original.spec == self.expected_spec(admission)?
                && original.state.is_terminal()
                && original.active_attempt.is_none(),
            "terminal failure does not bind original native dispatch"
        );
        let current = client
            .get(original.id)
            .await?
            .context("original failed native job absent")?;
        ensure!(
            current == original,
            "original failed allocation changed; explicit reconciliation required"
        );
        Ok(Some(Artifact {
            path: path.strip_prefix(&self.root)?.into(),
            sha256: hash(&data),
        }))
    }
    /// Freeze eligibility input before any submit or control side effect.
    /// The planner supplies only already-known predecessor result hashes; no mutable current-best aliases.
    pub async fn admit(&self, client: &Client, admission: &Admission) -> Result<String> {
        ensure!(!self.read_only, "read-only graph cannot admit work");
        let pinned = Transport::load(self, client)?;
        let client = &pinned;
        ensure!(
            admission.schema == "jope.native_graph_admission.v1"
                && admission.predecessors.len() <= 4096,
            "invalid graph admission"
        );
        let (task, verified_input) = crate::smartcluster_worker::verify_task_inputs(
            &self.root,
            &relative(&self.root, &admission.manifest.path)?,
            &admission.manifest.sha256,
        )?;
        task.validate_at(Timestamp::from(crate::resident_protocol::now_ms()?))?;
        for (id, predecessor) in &admission.predecessors {
            ensure!(
                relative(&self.root, &predecessor.path)?.starts_with(self.directory.join("tasks"))
                    && predecessor
                        .path
                        .file_name()
                        .is_some_and(|n| n == "verified.json"),
                "predecessor must be a committed graph result"
            );
            let prior: VerifiedResult = serde_json::from_slice(&checked(&self.root, predecessor)?)?;
            ensure!(
                prior.run == task.run && prior.generation <= task.generation,
                "predecessor run/generation changed"
            );
            ensure!(
                prior.schema == "jope.native_graph_result.v1"
                    && &prior.task_id == id
                    && id != &task.task_id,
                "invalid predecessor identity"
            );
            self.validate_predecessor(client, predecessor, &prior, 0, &mut BTreeSet::new())
                .await?;
            ensure!(
                verified_input.input_artifacts.get(&predecessor.path) == Some(&predecessor.sha256),
                "predecessor result missing from task dependency bindings"
            );
        }
        let bytes = serde_json::to_vec(admission)?;
        let key = hash(&serde_json::to_vec(&(
            &task.run,
            task.generation,
            &task.task_id,
        ))?);
        let archived = self
            .directory
            .join("archived-admissions")
            .join(format!("{key}.json"));
        if archived.exists() {
            ensure!(read(&archived)? == bytes, "archived admission conflict");
        } else {
            immutable(
                &self
                    .directory
                    .join("admissions")
                    .join(format!("{key}.json")),
                &bytes,
            )?;
        }
        Ok(key)
    }
    /// Acknowledgement is deliberately absent from this verification path.
    fn completed(
        &self,
        admission: &Admission,
        task: &TaskManifest,
        input: &InputManifest,
    ) -> Result<Option<VerifiedResult>> {
        let key = hash(&serde_json::to_vec(&(
            &task.run,
            task.generation,
            &task.task_id,
        ))?);
        let path = self
            .worker_state
            .join("tasks")
            .join(key)
            .join("complete.json");
        if !path.exists() {
            return Ok(None);
        }
        let bytes = read(&path)?;
        let complete: Value = serde_json::from_slice(&bytes)?;
        let output_path = relative(&self.root, &input.output_receipt)?;
        let output = read_limit(
            &output_path,
            if matches!(
                input.operation,
                Operation::DatasetPublication { .. } | Operation::PackAndPublish { .. }
            ) {
                256 * 1024 * 1024
            } else {
                128 * 1024 * 1024
            },
        )?;
        ensure!(
            complete["schema"] == "jope.smartcluster_completion.v1"
                && complete["task_id"] == task.task_id
                && complete["receipt_id"] == task.expected_receipt_id
                && complete["manifest_sha256"] == admission.manifest.sha256
                && complete["output_path"] == serde_json::to_value(&input.output_receipt)?
                && complete["output_sha256"] == hash(&output),
            "worker result binding changed"
        );
        let attempt: FencedAttempt = serde_json::from_value(complete["native_attempt"].clone())?;

        Ok(Some(VerifiedResult {
            schema: "jope.native_graph_result.v1".into(),
            admission_sha256: hash(&serde_json::to_vec(admission)?),
            task_manifest_sha256: admission.manifest.sha256.clone(),
            task_id: task.task_id.clone(),
            run: task.run.clone(),
            generation: task.generation,
            receipt_id: task.expected_receipt_id.clone(),
            native_attempt: attempt,
            worker_completion: Artifact {
                path: path.strip_prefix(&self.root)?.to_owned(),
                sha256: hash(&bytes),
            },
            output: Artifact {
                path: input.output_receipt.clone(),
                sha256: hash(&output),
            },
        }))
    }

    fn task_key(task: &TaskManifest) -> Result<String> {
        Ok(hash(&serde_json::to_vec(&(
            &task.run,
            task.generation,
            &task.task_id,
        ))?))
    }
    fn input(&self, admission: &Admission, task: &TaskManifest) -> Result<InputManifest> {
        let manifest = relative(&self.root, &admission.manifest.path)?;
        let bytes = read(&manifest.with_file_name(format!(
                "{}.input.json",
                manifest
                    .file_stem()
                    .context("manifest stem")?
                    .to_str()
                    .context("UTF8 stem")?
            )))?;
        ensure!(
            hash(&bytes) == task.input_manifest_sha256,
            "input manifest changed"
        );
        let input: InputManifest = serde_json::from_slice(&bytes)?;
        ensure!(
            input.root == self.root && input.schema == "jope.smartcluster_input.v1",
            "input root/schema mismatch"
        );
        Ok(input)
    }
    fn expected_spec(&self, admission: &Admission) -> Result<JobSpec> {
        // Structural reconstruction of original immutable request, not renewed admission.
        // Epoch zero preserves expired original deadlines while checking every typed field.
        Ok(jope::job_spec(
            &relative(&self.root, &admission.manifest.path)?,
            &admission.manifest.sha256,
            self.tenant.clone(),
            self.project.clone(),
            Timestamp::from(0),
        )?)
    }
    async fn native_proof(&self, client: &Transport, attempt: FencedAttempt) -> Result<Value> {
        let job = client
            .get(attempt.job_id)
            .await?
            .context("native allocation missing")?;
        ensure!(
            job.spec.tenant == self.tenant && job.spec.project == self.project,
            "native allocation scope mismatch"
        );
        let inspected = client
            .inspect_attempt(attempt)
            .await?
            .context("native attempt not retained")?;
        let record = inspected.record;
        ensure!(
            record.attempt == attempt
                && record.lease.attempt == attempt
                && record.started_at.is_some()
                && record.pid.is_some()
                && record.cgroup_path.is_some(),
            "native attempt never held an executing process allocation"
        );
        let proof = json!({"attempt":record.attempt,"pid":record.pid,"cgroup_path":record.cgroup_path,"started_at":record.started_at});
        Ok(proof)
    }
    fn admission_path(&self, key: &str) -> PathBuf {
        let active = self
            .directory
            .join("admissions")
            .join(format!("{key}.json"));
        if active.exists() {
            active
        } else {
            self.directory
                .join("archived-admissions")
                .join(format!("{key}.json"))
        }
    }
    fn release_resident(&self, task: &TaskManifest, admission: &Admission) -> Result<()> {
        if let Delivery::Resident { job_id } = admission.delivery {
            let path = self
                .directory
                .join("resident-owners")
                .join(format!("{job_id}.json"));
            if path.exists() {
                let owner: String = serde_json::from_slice(&read(&path)?)?;
                if owner == Self::task_key(task)? {
                    fs::remove_file(&path)?;
                    fs::File::open(path.parent().context("resident owner parent")?)?.sync_all()?;
                }
            }
        }
        Ok(())
    }
    fn archive_verified(&self, task: &TaskManifest, admission: &Admission) -> Result<()> {
        let key = Self::task_key(task)?;
        let active = self
            .directory
            .join("admissions")
            .join(format!("{key}.json"));
        let archive = self
            .directory
            .join("archived-admissions")
            .join(format!("{key}.json"));
        immutable(&archive, &serde_json::to_vec(admission)?)?;
        if active.exists() {
            ensure!(
                read(&active)? == read(&archive)?,
                "archival admission conflict"
            );
            fs::remove_file(&active)?;
            fs::File::open(active.parent().context("active parent")?)?.sync_all()?;
        }
        Ok(())
    }
    async fn validate_predecessor(
        &self,
        client: &Transport,
        artifact: &Artifact,
        result: &VerifiedResult,
        depth: usize,
        proofs: &mut BTreeSet<String>,
    ) -> Result<()> {
        ensure!(
            depth < 4096,
            "dependency recursion exceeds bounded graph depth"
        );
        let admission_key = hash(&serde_json::to_vec(&(
            &result.run,
            result.generation,
            &result.task_id,
        ))?);
        let expected_path = self
            .directory
            .join("tasks")
            .join(&admission_key)
            .join("verified.json");
        ensure!(
            relative(&self.root, &artifact.path)? == expected_path,
            "predecessor canonical identity mismatch"
        );
        ensure!(
            hash(&read(&expected_path)?) == artifact.sha256,
            "predecessor result changed"
        );
        let admission: Admission =
            serde_json::from_slice(&read(&self.admission_path(&admission_key))?)?;
        let task: TaskManifest =
            serde_json::from_slice(&checked(&self.root, &admission.manifest)?)?;
        Box::pin(self.validate_result(client, &admission, &task, result, depth + 1, proofs)).await
    }
    async fn validate_result(
        &self,
        client: &Transport,
        admission: &Admission,
        task: &TaskManifest,
        result: &VerifiedResult,
        depth: usize,
        proofs: &mut BTreeSet<String>,
    ) -> Result<()> {
        ensure!(
            depth < 4096
                && result.schema == "jope.native_graph_result.v1"
                && result.admission_sha256 == hash(&serde_json::to_vec(admission)?)
                && result.task_manifest_sha256 == admission.manifest.sha256
                && result.task_id == task.task_id
                && result.run == task.run
                && result.generation == task.generation
                && result.receipt_id == task.expected_receipt_id,
            "verified graph identity mismatch"
        );
        let proof_key = hash(&serde_json::to_vec(result)?);
        if proofs.contains(&proof_key) {
            return Ok(());
        }
        ensure!(
            proofs.len() < 16384,
            "verified dependency closure exceeds bounded task count"
        );
        let state = self.directory.join("tasks").join(Self::task_key(task)?);
        let submitted: Submitted = serde_json::from_slice(&read(&state.join("submitted.json"))?)?;
        ensure!(
            submitted.job_id == result.native_attempt.job_id
                && submitted.admission_sha256 == result.admission_sha256,
            "result allocation differs from dispatch"
        );
        let job = client
            .get(submitted.job_id)
            .await?
            .context("result job absent")?;
        match admission.delivery {
            Delivery::NativeJob => {
                let intent: JobSpec =
                    serde_json::from_slice(&read(&state.join("submit-intent.json"))?)?;
                ensure!(
                    intent == self.expected_spec(admission)? && job.spec == intent,
                    "submitted native spec differs from typed admission"
                );
            }
            Delivery::Resident { job_id } => {
                ensure!(
                    job_id == submitted.job_id
                        && job.spec.tenant == self.tenant
                        && job.spec.project == self.project,
                    "resident scope changed"
                );
                ensure!(
                    matches!(&job.spec.workload, jain_smartcluster::Workload::TrustedService {service,..} if service.as_str()=="jope-resident-gpu"),
                    "nonresident allocation cannot satisfy resident task"
                );
                ensure!(
                    job.spec
                        .resources
                        .gpu
                        .is_some_and(|gpu| gpu.count == 1
                            && gpu.mode == jain_smartcluster::GpuMode::Exclusive),
                    "resident exclusive GPU allocation absent"
                );
                let intent: SubtaskCommand =
                    serde_json::from_slice(&read(&state.join("dispatch-intent.json"))?)?;
                ensure!(
                    intent.manifest == relative(&self.root, &admission.manifest.path)?
                        && intent.manifest_sha256 == admission.manifest.sha256,
                    "resident original dispatch intent changed"
                );
            }
        }
        let proof = self.native_proof(client, result.native_attempt).await?;
        immutable(
            &state.join(format!(
                "native-proof-{}.json",
                result.native_attempt.fence.get()
            )),
            &serde_json::to_vec(&proof)?,
        )?;
        let input = self.input(admission, task)?;
        let expected_complete = self
            .worker_state
            .join("tasks")
            .join(Self::task_key(task)?)
            .join("complete.json");
        ensure!(
            relative(&self.root, &result.worker_completion.path)? == expected_complete
                && result.output.path == input.output_receipt,
            "result receipt paths changed"
        );
        let completion: Value =
            serde_json::from_slice(&checked(&self.root, &result.worker_completion)?)?;
        let output = checked_limit(
            &self.root,
            &result.output,
            if matches!(
                input.operation,
                Operation::DatasetPublication { .. } | Operation::PackAndPublish { .. }
            ) {
                256 * 1024 * 1024
            } else {
                128 * 1024 * 1024
            },
        )?;
        ensure!(
            completion["schema"] == "jope.smartcluster_completion.v1"
                && completion["native_attempt"] == serde_json::to_value(result.native_attempt)?
                && completion["dataset_logical_attempt"]
                    == serde_json::to_value(input.operation.dataset_logical_attempt())?
                && completion["manifest_sha256"] == admission.manifest.sha256
                && completion["task_id"] == task.task_id
                && completion["receipt_id"] == task.expected_receipt_id
                && completion["output_sha256"] == result.output.sha256
                && completion["output_path"] == serde_json::to_value(&input.output_receipt)?,
            "worker completion changed"
        );
        let root = self.root.clone();
        let operation = input.operation.clone();
        let numerical: Value = serde_json::from_slice(&output)?;
        if matches!(
            operation,
            Operation::DatasetPublication { .. } | Operation::PackAndPublish { .. }
        ) {
            ensure!(
                numerical["task_id"] == task.task_id
                    && numerical["receipt_id"] == task.expected_receipt_id
                    && numerical["attempt"]
                        == operation
                            .dataset_attempt(u64::from(result.native_attempt.number.get()))?,
                "dataset result differs from original native attempt/task"
            );
        }
        if let Operation::DatasetPublication { manifest, .. }
        | Operation::PackAndPublish { manifest, .. } = &operation
        {
            let request: Value = serde_json::from_slice(&read_limit(
                &relative(&root, manifest)?,
                256 * 1024 * 1024,
            )?)?;
            ensure!(
                request["output_receipt"] == serde_json::to_value(&input.output_receipt)?,
                "dataset original output receipt path differs"
            );
        }
        let original_task = task.clone();
        let bindings = tokio::task::spawn_blocking(move || {
            let bindings = verify_input_closure(&root, &original_task, &input)?;
            verify_numerical(&root, &operation, &numerical)?;
            Ok::<_, anyhow::Error>(bindings)
        })
        .await??;
        for (id, predecessor) in &admission.predecessors {
            ensure!(
                bindings.get(&predecessor.path) == Some(&predecessor.sha256),
                "predecessor not bound in original input closure"
            );
            let prior: VerifiedResult = serde_json::from_slice(&checked(&self.root, predecessor)?)?;
            ensure!(
                &prior.task_id == id
                    && prior.task_id != task.task_id
                    && prior.run == task.run
                    && prior.generation <= task.generation,
                "predecessor ordering mismatch"
            );
            Box::pin(self.validate_predecessor(client, predecessor, &prior, depth + 1, proofs))
                .await?;
        }
        proofs.insert(proof_key);
        Ok(())
    }
    fn blocked(&self, key: &str, error: &anyhow::Error) -> Result<()> {
        let message = format!("{error:#}");
        // Immutable deduplicated diagnostics retain earlier distinct failures.
        let value = json!({"schema":"jope.native_graph_blocked.v1","key":key,"error":message,"numeric_dispatch_granted":false});
        immutable(
            &self
                .directory
                .join("blocked")
                .join(key)
                .join(format!("{}.json", hash(message.as_bytes()))),
            &serde_json::to_vec(&value)?,
        )
    }
    /// Directory scans cannot establish scientific eligibility. Standalone execution
    /// requires a separate reviewed qualification scope; there is no ambient bypass.
    pub async fn tick(&self, _client: &Client) -> Result<Tick> {
        ensure!(
            !self.read_only,
            "read-only graph cannot reconcile or dispatch work"
        );
        anyhow::bail!("managed graph dispatch requires the independently approved native Loop")
    }
    pub(crate) fn verify_transport(
        &self,
        client: &Client,
        approval: &crate::native_loop::RunApprovalBody,
    ) -> Result<()> {
        let transport = Transport::load(self, client)?;
        ensure!(
            serde_json::to_vec(transport.approval())? == serde_json::to_vec(approval)?,
            "actual Graph/Client differs from Loop approval"
        );
        Ok(())
    }
    /// Only the compiled Loop implementation calls this after reconstructing an
    /// eligible Request and verifying its exact original Binding.
    pub(crate) async fn reconcile_eligible(
        &self,
        client: &Client,
        admission: &Admission,
    ) -> Result<Tick> {
        ensure!(!self.read_only, "read-only graph cannot reconcile work");
        let pinned = Transport::load(self, client)?;
        pinned.verify_dispatch_image()?;
        let task: TaskManifest =
            serde_json::from_slice(&checked(&self.root, &admission.manifest)?)?;
        let original = self.admission_path(&Self::task_key(&task)?);
        if !original.exists() {
            return Ok(Tick::default());
        }
        ensure!(
            read(&original)? == serde_json::to_vec(admission)?,
            "eligible original admission changed"
        );
        let mut outcome = Tick::default();
        let mut resident_busy = BTreeSet::new();
        if let Err(error) = self
            .tick_one(&pinned, admission, &task, &mut outcome, &mut resident_busy)
            .await
        {
            self.blocked(&Self::task_key(&task)?, &error)?;
            outcome.blocked += 1;
        }
        Ok(outcome)
    }
    async fn tick_one(
        &self,
        client: &Transport,
        admission: &Admission,
        task: &TaskManifest,
        outcome: &mut Tick,
        resident_busy: &mut std::collections::BTreeSet<JobId>,
    ) -> Result<()> {
        let key = Self::task_key(task)?;
        let state = self.directory.join("tasks").join(key);
        directory(&state)?;
        let verified_path = state.join("verified.json");
        if verified_path.exists() {
            let result: VerifiedResult = serde_json::from_slice(&read(&verified_path)?)?;
            self.validate_result(client, admission, task, &result, 0, &mut BTreeSet::new())
                .await?;
            self.release_resident(task, admission)?;
            self.archive_verified(task, admission)?;
            outcome.verified += 1;
            return Ok(());
        }
        let input = self.input(admission, task)?;
        if let Some(result) = self.completed(admission, task, &input)? {
            let submitted_path = state.join("submitted.json");
            if !submitted_path.exists() {
                ensure!(
                    admission.delivery == Delivery::NativeJob,
                    "resident dispatch intent absent"
                );
                let intent: JobSpec =
                    serde_json::from_slice(&read(&state.join("submit-intent.json"))?)?;
                ensure!(
                    intent == self.expected_spec(admission)?,
                    "original submit intent changed"
                );
                let job = client
                    .get(result.native_attempt.job_id)
                    .await?
                    .context("completed job absent")?;
                ensure!(
                    job.spec == intent,
                    "completed allocation differs from intent"
                );
                immutable(
                    &submitted_path,
                    &serde_json::to_vec(&Submitted {
                        admission_sha256: result.admission_sha256.clone(),
                        job_id: job.id,
                    })?,
                )?;
            }
            self.validate_result(client, admission, task, &result, 0, &mut BTreeSet::new())
                .await?;
            immutable(&verified_path, &serde_json::to_vec(&result)?)?;
            self.release_resident(task, admission)?;
            self.archive_verified(task, admission)?;
            outcome.verified += 1;
            return Ok(());
        }
        if state.join("failed.json").exists() {
            self.archive_verified(task, admission)?;
            outcome.failed += 1;
            return Ok(());
        }
        let root = self.root.clone();
        let original_task = task.clone();
        let original_input = input.clone();
        let bindings = tokio::task::spawn_blocking(move || {
            verify_input_closure(&root, &original_task, &original_input)
        })
        .await??;
        for predecessor in admission.predecessors.values() {
            ensure!(
                bindings.get(&predecessor.path) == Some(&predecessor.sha256),
                "unbound predecessor"
            );
            let prior: VerifiedResult = serde_json::from_slice(&checked(&self.root, predecessor)?)?;
            self.validate_predecessor(client, predecessor, &prior, 0, &mut BTreeSet::new())
                .await?;
        }
        let submitted_path = state.join("submitted.json");
        match admission.delivery {
            Delivery::NativeJob => {
                let expected = self.expected_spec(admission)?;
                let intent_path = state.join("submit-intent.json");
                let intent: JobSpec = if intent_path.exists() {
                    serde_json::from_slice(&read(&intent_path)?)?
                } else {
                    ensure!(
                        task.deadline > Timestamp::from(crate::resident_protocol::now_ms()?),
                        "new task deadline elapsed"
                    );
                    immutable(&intent_path, &serde_json::to_vec(&expected)?)?;
                    expected.clone()
                };
                ensure!(
                    intent == expected,
                    "replayed intent differs from typed admission"
                );
                if submitted_path.exists() {
                    let submitted: Submitted = serde_json::from_slice(&read(&submitted_path)?)?;
                    ensure!(
                        submitted.admission_sha256 == hash(&serde_json::to_vec(admission)?),
                        "submitted identity changed"
                    );
                    let job = client
                        .get(submitted.job_id)
                        .await?
                        .context("submitted native job disappeared")?;
                    ensure!(job.spec == expected, "submitted job changed");
                    if job.state.is_terminal() {
                        immutable(
                            &state.join("failed.json"),
                            &serde_json::to_vec(
                                &json!({"schema":"jope.native_graph_failure.v1","job":job,"reason":"terminal allocation without verified result"}),
                            )?,
                        )?;
                        outcome.failed += 1;
                    } else {
                        outcome.waiting_execution += 1;
                    }
                } else {
                    let job = client.submit(intent).await?;
                    ensure!(
                        job.spec == expected,
                        "submission response differs from original spec"
                    );
                    immutable(
                        &submitted_path,
                        &serde_json::to_vec(&Submitted {
                            admission_sha256: hash(&serde_json::to_vec(admission)?),
                            job_id: job.id,
                        })?,
                    )?;
                    outcome.submitted += 1;
                }
            }
            Delivery::Resident { job_id } => {
                ensure!(
                    matches!(
                        input.operation,
                        Operation::Train { .. } | Operation::Proposal { .. }
                    ),
                    "resident delivery is limited to GPU learning subtasks"
                );
                ensure!(
                    task.deadline > Timestamp::from(crate::resident_protocol::now_ms()?),
                    "resident subtask original deadline elapsed"
                );
                let owner_path = self
                    .directory
                    .join("resident-owners")
                    .join(format!("{job_id}.json"));
                let task_key = Self::task_key(task)?;
                if owner_path.exists() {
                    let owner: String = serde_json::from_slice(&read(&owner_path)?)?;
                    if owner != task_key {
                        outcome.waiting_execution += 1;
                        return Ok(());
                    }
                } else {
                    immutable(&owner_path, &serde_json::to_vec(&task_key)?)?;
                }
                if !resident_busy.insert(job_id) {
                    outcome.waiting_execution += 1;
                    return Ok(());
                }
                let job = client
                    .get(job_id)
                    .await?
                    .context("resident allocation absent")?;
                ensure!(
                    job.spec.tenant == self.tenant
                        && job.spec.project == self.project
                        && matches!(&job.spec.workload,jain_smartcluster::Workload::TrustedService{service,..} if service.as_str()=="jope-resident-gpu"),
                    "resident allocation scope/service differs"
                );
                ensure!(
                    job.spec.resources.gpu.is_some_and(
                        |g| g.count == 1 && g.mode == jain_smartcluster::GpuMode::Exclusive
                    ),
                    "resident exclusive GPU missing"
                );
                ensure!(
                    !job.state.is_terminal(),
                    "resident terminated; explicit recovery needed"
                );
                if job.state != JobState::Running {
                    outcome.waiting_execution += 1;
                    return Ok(());
                }
                ensure!(
                    task.resources.cpu.maximum_millicores
                        <= job.spec.resources.cpu.maximum_millicores
                        && task.resources.ram.hard_bytes <= job.spec.resources.ram.hard_bytes
                        && task.resources.scratch_bytes <= job.spec.resources.scratch_bytes
                        && task.resources.gpu.zip(job.spec.resources.gpu).is_some_and(
                            |(wanted, held)| wanted.count <= held.count
                                && wanted.vram_bytes <= held.vram_bytes
                                && wanted.mode == held.mode
                        ),
                    "subtask exceeds resident allocation envelope"
                );
                let attempt = job.active_attempt.context("resident fence absent")?;
                let marker = state.join(format!("delivered-{}.json", attempt.fence.get()));
                if marker.exists() {
                    let receipt: Value = serde_json::from_slice(&read(&marker)?)?;
                    ensure!(
                        receipt["attempt"] == serde_json::to_value(attempt)?
                            && receipt["admission_sha256"] == hash(&serde_json::to_vec(admission)?),
                        "resident delivery marker changed"
                    );
                    outcome.waiting_execution += 1;
                    return Ok(());
                }
                immutable(
                    &submitted_path,
                    &serde_json::to_vec(&Submitted {
                        admission_sha256: hash(&serde_json::to_vec(admission)?),
                        job_id,
                    })?,
                )?;
                let command = SubtaskCommand {
                    manifest: relative(&self.root, &admission.manifest.path)?,
                    manifest_sha256: admission.manifest.sha256.clone(),
                };
                let ack = client
                    .dispatch_subtask(job_id, &state.join("dispatch-intent.json"), &command)
                    .await?;
                immutable(
                    &marker,
                    &serde_json::to_vec(
                        &json!({"attempt":attempt,"admission_sha256":hash(&serde_json::to_vec(admission)?),"delivery":ack,"numerical_completion":false}),
                    )?,
                )?;
                outcome.submitted += 1;
            }
        }
        Ok(())
    }
}

fn verify_input_closure(
    root: &Path,
    task: &TaskManifest,
    input: &InputManifest,
) -> Result<BTreeMap<PathBuf, String>> {
    let bindings = if let Some(closure) = &input.closure {
        ensure!(
            input.input_artifacts.is_empty(),
            "inline and indexed input closure mixed"
        );
        ensure!(
            task.dependencies
                == BTreeMap::from([(
                    closure.path.to_string_lossy().into_owned(),
                    closure.sha256.clone()
                )]),
            "task closure identity differs"
        );
        use std::io::BufRead;
        let path = relative(root, &closure.path)?;
        let file = fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&path)?;
        ensure!(
            file.metadata()?.is_file() && file.metadata()?.len() <= 256 * 1024 * 1024,
            "input closure type/size bound"
        );
        let mut reader = std::io::BufReader::new(file);
        let mut result = BTreeMap::new();
        let mut total = 0usize;
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        loop {
            let mut line = Vec::new();
            let n = reader.by_ref().take(8193).read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            total += n;
            ensure!(
                n <= 8192 && total <= 256 * 1024 * 1024 && result.len() < 1000000,
                "closure line/count bound"
            );
            digest.update(&line);
            let artifact: Artifact = serde_json::from_slice(&line)?;
            ensure!(
                result.insert(artifact.path, artifact.sha256).is_none(),
                "duplicate input closure path"
            );
        }
        ensure!(
            format!("{:x}", digest.finalize()) == closure.sha256,
            "input closure changed"
        );
        result
    } else {
        ensure!(
            input.input_artifacts.len() <= 4096
                && task.dependencies
                    == input
                        .input_artifacts
                        .iter()
                        .map(|(p, h)| (p.to_string_lossy().into_owned(), h.clone()))
                        .collect(),
            "inline input bindings differ from task"
        );
        input.input_artifacts.clone()
    };
    for (path, sha) in &bindings {
        ensure!(
            digest_file(&relative(root, path)?)? == *sha,
            "task input artifact changed: {}",
            path.display()
        );
    }
    Ok(bindings)
}

fn digest_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    ensure!(
        file.metadata()?.is_file(),
        "artifact must be a regular file"
    );
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn declared_artifacts(result: &Value) -> Result<BTreeMap<PathBuf, String>> {
    let artifacts: BTreeMap<PathBuf, String> = serde_json::from_value(result["artifacts"].clone())?;
    ensure!(
        !artifacts.is_empty() && artifacts.len() <= 100000,
        "numerical artifact closure absent or oversized"
    );
    for digest in artifacts.values() {
        ensure!(
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
            "artifact digest invalid"
        );
    }
    Ok(artifacts)
}
fn verify_tree(
    directory: &Path,
    artifacts: &BTreeMap<PathBuf, String>,
    future: &[PathBuf],
) -> Result<()> {
    for (path, sha) in artifacts {
        ensure!(
            digest_file(&relative(directory, path)?)? == *sha,
            "numerical artifact changed: {}",
            path.display()
        );
    }
    let mut count = 0usize;
    for entry in walkdir::WalkDir::new(directory).follow_links(false) {
        let entry = entry?;
        count += 1;
        ensure!(count <= 200000, "output tree exceeds bounded census");
        ensure!(
            !entry.file_type().is_symlink()
                && (entry.file_type().is_dir() || entry.file_type().is_file()),
            "output tree has special file"
        );
        if entry.file_type().is_file() {
            let suffix = entry.path().strip_prefix(directory)?;
            ensure!(
                artifacts.contains_key(suffix)
                    || future
                        .iter()
                        .any(|allowed| entry.path().starts_with(allowed)),
                "unaccounted numerical output {}",
                entry.path().display()
            );
        }
    }
    Ok(())
}
fn closure_json(
    directory: &Path,
    artifacts: &BTreeMap<PathBuf, String>,
    path: &Path,
    limit: u64,
) -> Result<Value> {
    let sha = artifacts
        .get(path)
        .context("required numerical artifact not declared")?;
    let bytes = read_limit(&relative(directory, path)?, limit)?;
    ensure!(hash(&bytes) == *sha, "parsed numerical artifact changed");
    Ok(serde_json::from_slice(&bytes)?)
}
fn verify_numerical(root: &Path, operation: &Operation, result: &Value) -> Result<()> {
    match operation {
        Operation::DirectEvaluation { manifest, sha256 }
        | Operation::SeededContinuation { manifest, sha256 } => {
            let reference = Artifact {
                path: manifest.clone(),
                sha256: sha256.clone(),
            };
            let task: crate::native_genome::GenomeTask =
                crate::native_genome::load(root, &reference)?;
            let result: crate::native_genome::GenomeResult =
                serde_json::from_value(result.clone())?;
            ensure!(
                result.task == reference && result.stage == task.stage,
                "native genome result/task changed"
            );
            // The worker output itself is already hash-bound by Graph::validate_result.
            // Reuse strict original-row checks against its actual recorded file.
            let directory = relative(root, &task.output)?;
            verify_tree(&directory, &result.artifacts, &[])?;
            crate::native_genome::verify_result(root, &task, &result)?;
        }
        Operation::Verify => ensure!(
            result["schema"] == "jope.native_verification.v1" && result["verified"] == true,
            "verification receipt absent"
        ),
        Operation::Search {
            output,
            prepared,
            cut,
            evaluations,
            rng_seed,
            ..
        } => {
            ensure!(
                result["schema"] == "jope.native_gp_completion.v1"
                    && result["evaluations_budget"] == *evaluations
                    && result["seed"] == *rng_seed,
                "GP completion task changed"
            );
            let directory = relative(root, output)?;
            let artifacts = declared_artifacts(result)?;
            verify_tree(&directory, &artifacts, &[])?;
            let search = closure_json(
                &directory,
                &artifacts,
                Path::new("search.json"),
                1024 * 1024,
            )?;
            let rows_value = closure_json(
                &directory,
                &artifacts,
                Path::new("observations.json"),
                128 * 1024 * 1024,
            )?;
            let rows = rows_value.as_array().context("observation array")?;
            ensure!(
                *evaluations <= 1000
                    && rows.len() <= *evaluations
                    && search["final_tests_opened"] == false
                    && search["observations_sha256"] == artifacts[Path::new("observations.json")],
                "search observations/task bound changed"
            );
            let prepared_sha = digest_file(&relative(root, prepared)?.join("prepared.json"))?;
            let mut successes = 0usize;
            for row in rows
                .iter()
                .filter(|r| r["failure"].is_null() && r["metrics"].is_object())
            {
                ensure!(
                    row["prepared_sha256"] == prepared_sha
                        && row["cut"] == *cut
                        && matches!(
                            row["partition"].as_str(),
                            Some("meta_train" | "meta_validation")
                        ),
                    "observation prepared/cut/sealed partition changed"
                );
                ensure!(
                    row["program_sha256"]
                        == hash(
                            row["genome"]
                                .as_str()
                                .context("observed genome absent")?
                                .as_bytes()
                        ),
                    "observed genome differs from fitted program"
                );
                crate::label_admission::validate_completion(root, row)?;
                let reference = |value: &Value| -> Result<Artifact> {
                    Ok(Artifact {
                        path: PathBuf::from(
                            value["path"]
                                .as_str()
                                .context("completion reference path")?,
                        ),
                        sha256: value["sha256"]
                            .as_str()
                            .context("completion reference hash")?
                            .into(),
                    })
                };
                let completion: Value =
                    serde_json::from_slice(&checked(root, &reference(&row["clean_completion"])?)?)?;
                let fit_bytes = checked(root, &reference(&completion["fit_completion"])?)?;
                let fit = crate::eval_registry::verify_completion_receipt(&fit_bytes)?;
                ensure!(
                    fit.outcome == crate::eval_registry::ExecutionOutcome::Success
                        && row["program_sha256"] == fit.program_sha256
                        && row["training_sha256"] == fit.training_sha256
                        && row["evaluator_sha256"] == fit.executable_sha256,
                    "failed or mismatched original fit cannot count as success"
                );
                let model = reference(&completion["model"])?;
                ensure!(
                    fit.model_sha256.as_deref() == Some(model.sha256.as_str())
                        && digest_file(&relative(root, &model.path)?)? == model.sha256,
                    "original fitted model changed"
                );
                successes += 1;
            }
            ensure!(
                successes > 0 && search["successful_measurements"] == successes,
                "search success count differs from verified fits"
            );
        }
        Operation::Pack {
            output,
            contexts,
            vocabulary,
            qualification,
            ..
        } => {
            ensure!(
                result["schema"] == "jope.native_pack_completion.v1",
                "pack completion schema"
            );
            let directory = relative(root, output)?;
            let artifacts = declared_artifacts(result)?;
            verify_tree(&directory, &artifacts, &[])?;
            let index_value = closure_json(
                &directory,
                &artifacts,
                Path::new("index.json"),
                128 * 1024 * 1024,
            )?;
            let index: crate::corpus::Index = serde_json::from_value(index_value)?;
            ensure!(
                index.schema == "parallel.teaching_corpus.v1"
                    && index.qualification == *qualification
                    && index.records > 0
                    && index.record_bytes == crate::corpus::RECORD_BYTES
                    && index.records_sha256
                        == *artifacts
                            .get(Path::new("records.bin"))
                            .context("packed records absent")?
                    && index.vocabulary_sha256 == digest_file(&relative(root, vocabulary)?)?
                    && index.vocabulary_sha256
                        == *artifacts
                            .get(Path::new("vocabulary.json"))
                            .context("packed vocabulary absent")?
                    && index.context_index_sha256 == digest_file(&relative(root, contexts)?)?,
                "packed corpus parent/record closure changed"
            );
            let bytes = index
                .record_bytes
                .checked_mul(index.records)
                .context("record bytes overflow")?;
            ensure!(
                fs::metadata(relative(&directory, Path::new("records.bin"))?)?.len()
                    == u64::try_from(bytes)?,
                "packed record length changed"
            );
            for context in &index.contexts {
                if context.partition == "meta_test" {
                    continue;
                } // opaque metadata only; never materialize sealed context
                ensure!(
                    matches!(context.partition.as_str(), "meta_train" | "meta_validation")
                        && digest_file(&relative(root, &context.path)?)? == context.sha256
                        && digest_file(&relative(root, &context.prepared)?.join("prepared.json"))?
                            == context.prepared_sha256,
                    "packed context/preparation dependency changed"
                );
            }
            for group in &index.groups {
                ensure!(
                    group.context < index.contexts.len()
                        && group.records.iter().all(|i| *i < index.records)
                        && matches!(group.partition.as_str(), "meta_train" | "meta_validation")
                        && index.contexts[group.context].partition == group.partition
                        && index.contexts[group.context].family == group.family,
                    "packed group references invalid or sealed context"
                );
            }
        }
        Operation::Train {
            corpus,
            initialize,
            replay_policy,
            initial_optimizer_step,
            requested_step_increment,
            target_optimizer_step,
            ..
        } => {
            ensure!(
                result["schema"] == "jope.smartcluster_training_completion.v1"
                    && initial_optimizer_step.checked_add(*requested_step_increment)
                        == Some(*target_optimizer_step)
                    && result["window"]["initial_optimizer_step"] == *initial_optimizer_step
                    && result["window"]["requested_step_increment"] == *requested_step_increment
                    && result["window"]["target_optimizer_step"] == *target_optimizer_step
                    && result["numerical_receipt"]["steps"] == *target_optimizer_step,
                "original optimizer window incomplete"
            );
            let latest = &result["latest"];
            let checkpoint =
                PathBuf::from(latest["checkpoint"].as_str().context("checkpoint path")?);
            let state: Value = serde_json::from_slice(&checked(
                root,
                &Artifact {
                    path: checkpoint.join("state.json"),
                    sha256: latest["state_sha256"]
                        .as_str()
                        .context("state hash")?
                        .into(),
                },
            )?)?;
            let parent_hash = digest_file(&relative(root, initialize)?.join("state.json"))?;
            let corpus_hash = digest_file(&relative(root, corpus)?.join("index.json"))?;
            ensure!(
                state["step"] == *target_optimizer_step
                    && state["corpus_sha256"] == corpus_hash
                    && result["numerical_receipt"]["corpus_sha256"] == corpus_hash
                    && state["initialization"]["state_sha256"] == parent_hash
                    && state["initialization"]["optimizer_parent_step"] == *initial_optimizer_step
                    && state["initialization"]["replay_policy_sha256"]
                        == digest_file(&relative(root, replay_policy)?)?,
                "checkpoint original optimizer/corpus lineage changed"
            );
            let directory = relative(root, &checkpoint)?;
            ensure!(
                digest_file(&directory.join("model.ot"))? == state["model_sha256"]
                    && digest_file(&directory.join("optimizer.ot"))? == state["optimizer_sha256"]
                    && digest_file(&directory.join("optimizer.json"))?
                        == state["optimizer_metadata_sha256"],
                "checkpoint model/optimizer closure changed"
            );
        }
        Operation::Proposal {
            output,
            corpus,
            checkpoint,
            champion_snapshot,
            ..
        } => {
            ensure!(
                result["schema"] == "jope.smartcluster_proposal_completion.v1",
                "proposal schema changed"
            );
            let directory = relative(root, output)?;
            let artifacts = declared_artifacts(result)?;
            let numerical = closure_json(
                &directory,
                &artifacts,
                Path::new("complete-0.json"),
                1024 * 1024,
            )?;
            ensure!(
                numerical == result["numerical_receipt"]
                    && numerical["rank"] == 0
                    && numerical["world"] == 1,
                "proposal numerical identity changed"
            );
            let jobs: crate::service::GpJobs = serde_json::from_value(closure_json(
                &directory,
                &artifacts,
                Path::new("jobs-0.json"),
                16 * 1024 * 1024,
            )?)?;
            ensure!(
                jobs.schema == "parallel.gp_jobs.v1"
                    && !jobs.jobs.is_empty()
                    && jobs.jobs.len() <= 10000,
                "proposal job manifest empty/invalid"
            );
            let parent_sha = digest_file(&relative(root, checkpoint)?.join("state.json"))?;
            let champion_sha = digest_file(&relative(root, champion_snapshot)?)?;
            ensure!(
                jobs.vocabulary.as_ref().context("proposal vocabulary")?
                    == &corpus.join("vocabulary.json"),
                "proposal corpus vocabulary changed"
            );
            let mut future = Vec::new();
            for job in jobs.jobs {
                let seed = job.seeds.context("proposal seed path")?;
                let sha = job.seeds_sha256.context("proposal seed hash")?;
                let seed_path = relative(root, &seed)?;
                let suffix = seed_path
                    .strip_prefix(&directory)
                    .context("proposal seeds outside output closure")?;
                ensure!(
                    artifacts.get(suffix) == Some(&sha),
                    "proposal seed not in original producer closure"
                );
                let population: crate::gp::SeedPopulation = serde_json::from_value(closure_json(
                    &directory,
                    &artifacts,
                    suffix,
                    16 * 1024 * 1024,
                )?)?;
                ensure!(
                    population.schema == "parallel.seed_population.v1"
                        && !population.candidates.is_empty()
                        && population.checkpoint_sha256 == parent_sha
                        && population.prepared_sha256 == job.prepared_sha256
                        && population.cut == job.cut,
                    "proposal parent/prepared identity changed"
                );
                let generation = suffix
                    .parent()
                    .context("seed parent")?
                    .join("generation.json");
                let generation = closure_json(&directory, &artifacts, &generation, 1024 * 1024)?;
                ensure!(
                    generation["seeds_sha256"] == sha
                        && generation["checkpoint_sha256"] == parent_sha
                        && generation["champion_snapshot_sha256"] == champion_sha
                        && generation["final_tests_opened"] == false,
                    "proposal generation closure changed"
                );
                ensure!(
                    !job.output.is_absolute()
                        && job
                            .output
                            .components()
                            .all(|c| matches!(c, Component::Normal(_))),
                    "unsafe future search output"
                );
                let output_path = root.join(&job.output);
                ensure!(
                    output_path.starts_with(&directory) && output_path != directory,
                    "future search output outside exact proposal subtree"
                );
                ensure!(
                    output_path == seed_path.parent().context("seed directory")?.join("search"),
                    "future GP output is not the exact generated search directory"
                );
                future.push(output_path);
            }
            // Later GP tasks legitimately create files only below their declared output roots.
            // Every producer-hashed file remains mandatory even inside those roots.
            verify_tree(&directory, &artifacts, &future)?;
        }
        Operation::DatasetPublication {
            manifest, sha256, ..
        }
        | Operation::PackAndPublish {
            manifest, sha256, ..
        } => verify_dataset_result(
            root,
            manifest,
            sha256,
            matches!(operation, Operation::PackAndPublish { .. }),
            result,
        )?,
        Operation::Resident => anyhow::bail!("resident bootstrap is not a scientific dependency"),
    }
    Ok(())
}

fn dataset_binding(root: &Path, value: &Value) -> Result<(PathBuf, String)> {
    let path = PathBuf::from(
        value["path"]
            .as_str()
            .context("dataset binding path absent")?,
    );
    let sha = value["sha256"]
        .as_str()
        .context("dataset binding digest absent")?
        .to_owned();
    ensure!(
        sha.len() == 64
            && sha
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "dataset digest malformed"
    );
    ensure!(
        digest_file(&relative(root, &path)?)? == sha,
        "dataset dependency changed"
    );
    Ok((path, sha))
}
fn dataset_tool(root: &Path) -> Result<(fs::File, String, Duration)> {
    use std::os::unix::fs::MetadataExt;
    let path = Path::new(crate::smartcluster_worker::POLICY_PATH);
    for ancestor in path.ancestors() {
        let m = fs::symlink_metadata(ancestor)?;
        ensure!(
            !m.file_type().is_symlink() && m.uid() == 0 && m.mode() & 0o022 == 0,
            "dataset verifier policy is not root controlled"
        );
    }
    let policy: crate::smartcluster_worker::WorkerPolicy = serde_json::from_slice(&read(path)?)?;
    ensure!(
        policy.root == root,
        "dataset verifier root differs from policy"
    );
    let executable = policy
        .datasets_executable
        .context("dataset verifier executable absent")?;
    let sha = policy
        .datasets_sha256
        .context("dataset verifier digest absent")?;
    ensure!(
        executable.is_absolute() && fs::canonicalize(&executable)? == executable,
        "dataset executable physical path required"
    );
    let mut file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&executable)?;
    let m = file.metadata()?;
    ensure!(
        m.is_file() && m.uid() == 0 && m.mode() & 0o777 == 0o555,
        "dataset executable must be root-owned0555"
    );
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    ensure!(
        hex::encode(digest.finalize()) == sha,
        "dataset verifier executable changed"
    );
    let timeout = policy.dataset_verification_timeout_ms.unwrap_or(300_000);
    ensure!(
        (1..=14_400_000).contains(&timeout),
        "dataset verifier allowance must be positive and at most four hours"
    );
    Ok((file, sha, Duration::from_millis(timeout)))
}
fn dataset_chain_command(
    executable: &fs::File,
    path: &Path,
    sha: &str,
    deadline: std::time::Instant,
) -> Result<Value> {
    use std::{
        os::unix::process::CommandExt,
        process::{Command, Stdio},
        time::Duration,
    };
    ensure!(
        std::time::Instant::now() < deadline,
        "dataset verification deadline expired"
    );
    let mut child = Command::new(format!("/proc/self/fd/{}", executable.as_raw_fd()))
        .arg("verify-local-chain")
        .arg("--chain-manifest")
        .arg(path)
        .arg("--chain-manifest-sha256")
        .arg(sha)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("LANG", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .context("dataset verifier stdout absent")?;
    let (send, recv) = std::sync::mpsc::sync_channel(1);
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout
            .take(256 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes);
        let _ = send.send(result);
    });
    let result = recv.recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()));
    let bytes = match result {
        Ok(Ok(bytes)) if bytes.len() <= 256 * 1024 * 1024 => bytes,
        other => {
            // Original leader remains unreaped: its PID/process-group identity cannot be reused.
            let _ = nix::sys::signal::killpg(
                nix::unistd::Pid::from_raw(child.id().try_into()?),
                nix::sys::signal::Signal::SIGKILL,
            );
            let _ = child.wait();
            let _ = reader.join();
            anyhow::bail!(
                "dataset verifier timed out, failed or exceeded stdout bound: {}",
                match other {
                    Err(_) => "deadline/channel",
                    Ok(Err(_)) => "read",
                    _ => "size",
                }
            );
        }
    };
    let _ = reader.join();
    loop {
        if let Some(status) = child.try_wait()? {
            ensure!(status.success(), "dataset authority refused original chain");
            break;
        }
        if std::time::Instant::now() >= deadline {
            let _ = nix::sys::signal::killpg(
                nix::unistd::Pid::from_raw(child.id().try_into()?),
                nix::sys::signal::Signal::SIGKILL,
            );
            let _ = child.wait();
            anyhow::bail!("dataset verifier exit deadline exceeded");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok(serde_json::from_slice(&bytes)?)
}
fn verify_dataset_result(
    root: &Path,
    manifest: &Path,
    sha: &str,
    packing: bool,
    result: &Value,
) -> Result<()> {
    let bytes = read_limit(&relative(root, manifest)?, 256 * 1024 * 1024)?;
    ensure!(hash(&bytes) == sha, "original dataset manifest changed");
    let request: Value = serde_json::from_slice(&bytes)?;
    let kind = request["operation"]["kind"]
        .as_str()
        .context("dataset operation absent")?;
    ensure!(
        request["schema"] == "jain.datasets_service_manifest.v1"
            && request["root"] == serde_json::to_value(root)?
            && if packing {
                matches!(kind, "pack_and_publish" | "pack_continuation_and_publish")
            } else {
                matches!(
                    kind,
                    "winner_publish" | "corpus_publish" | "publish_pending"
                )
            },
        "dataset service manifest incompatible"
    );
    let (executable, tool_sha, allowance) = dataset_tool(root)?;
    let deadline = std::time::Instant::now() + allowance;
    ensure!(
        result["schema"] == "jain.datasets_service_receipt.v1"
            && result["passed"] == true
            && result["final_tests_opened"] == false
            && result["manifest_sha256"] == sha
            && result["task_id"] == request["task_id"]
            && result["attempt"] == request["attempt"]
            && result["receipt_id"] == request["expected_receipt_id"]
            && result["operation"] == kind
            && result["tool_sha256"] == tool_sha,
        "dataset completion identity changed"
    );
    if kind == "publish_pending" {
        return verify_pending_publications(&request, result, &executable, deadline);
    }
    let rows = result["output_files"]
        .as_array()
        .context("dataset output closure absent")?;
    ensure!(rows.len() <= 200000, "dataset output closure count bound");
    let mut outputs = BTreeMap::new();
    for row in rows {
        ensure!(
            std::time::Instant::now() < deadline,
            "dataset verification deadline expired"
        );
        let (path, sha) = dataset_binding(root, row)?;
        ensure!(
            outputs
                .insert(path, sha.clone())
                .is_none_or(|old| old == sha),
            "conflicting dataset output identity"
        );
    }
    let captures = result["captures"]
        .as_array()
        .context("dataset captures absent")?;
    let publications = result["publications"]
        .as_array()
        .context("dataset publication chains absent")?;
    ensure!(
        captures.len() == publications.len() && publications.len() <= 4096,
        "capture/publication count differs or exceeds bound"
    );
    ensure!(
        request["publication"].is_object(),
        "original publication target absent"
    );
    let store = PathBuf::from(request["store"].as_str().context("dataset store absent")?);
    ensure!(
        store.is_absolute() && fs::canonicalize(&store)? == store,
        "physical absolute dataset authority store required"
    );
    ensure!(
        result["store"] == serde_json::to_value(&store)?
            && result["namespace"] == request["namespace"],
        "dataset store/namespace changed"
    );
    let initial_parent = result["cursor_before"]["last_local_commit"]
        .as_str()
        .or_else(|| request["publication"]["parent"].as_str())
        .context("original chain parent absent")?;
    let mut captured_files = BTreeMap::new();
    let mut previous = Some(Value::String(initial_parent.to_owned()));
    for (capture, publication) in captures.iter().zip(publications) {
        ensure!(
            capture["acknowledged"] == true
                && capture["producer"] == publication["producer"]
                && capture["sequence"] == publication["sequence"]
                && capture["envelope_sha256"] == publication["envelope_sha256"],
            "publication differs from original acknowledged capture"
        );
        let path = Path::new(
            publication["chain_manifest_path"]
                .as_str()
                .context("chain path absent")?,
        );
        let sha = publication["chain_manifest_sha256"]
            .as_str()
            .context("chain digest absent")?;
        ensure!(
            path.is_absolute() && fs::canonicalize(path)? == path,
            "chain manifest must be physical"
        );
        let bytes = read_limit(path, 256 * 1024 * 1024)?;
        ensure!(hash(&bytes) == sha, "original publication chain changed");
        let chain: Value = serde_json::from_slice(&bytes)?;
        for field in [
            "producer",
            "sequence",
            "envelope_sha256",
            "final_event_commit",
            "final_event_ref",
            "batches",
            "target",
        ] {
            ensure!(
                chain[field] == publication[field],
                "publication field differs from original chain: {field}"
            );
        }
        ensure!(
            chain["store_root"] == serde_json::to_value(&store)?,
            "chain refers to unrelated receiver store"
        );
        for field in [
            "repository",
            "state_directory",
            "lfs_binary",
            "lfs_binary_sha256",
            "timestamp",
            "scratch_directory",
        ] {
            ensure!(
                chain["target"][field] == request["publication"][field],
                "chain target differs from original dataset task: {field}"
            );
        }
        let verified = dataset_chain_command(&executable, path, sha, deadline)?;
        if let Some(ref commit) = previous {
            ensure!(
                verified["requested_parent"] == *commit
                    && verified["actual_parent"] == chain["target"]["parent"]
                    && verified["requested_parent_is_ancestor"] == true,
                "publication original requested ancestry not independently proven"
            );
        }
        ensure!(
            verified["dispatch_intent_sha256"] == chain["dispatch_intent_sha256"]
                && verified["dispatch_intent_sha256"]
                    .as_str()
                    .is_some_and(|s| s.len() == 64),
            "original publication dispatch intent proof missing"
        );
        ensure!(
            verified["original_manifest_sha256"] == result["manifest_sha256"]
                || capture["already_bound"] == true,
            "new capture publication belongs to another original manifest"
        );
        ensure!(
            verified["schema"] == "jain.publication.chain_verification.v1"
                && verified["passed"] == true
                && verified["chain_manifest_path"] == publication["chain_manifest_path"]
                && verified["chain_manifest_sha256"] == sha
                && verified["producer"] == capture["producer"]
                && verified["sequence"] == capture["sequence"]
                && verified["envelope_sha256"] == capture["envelope_sha256"]
                && verified["final_event_commit"] == publication["final_event_commit"]
                && verified["final_event_ref"] == publication["final_event_ref"],
            "independent chain verification differs from original event"
        );
        if kind != "winner_publish" {
            for file in verified["captured_files"]
                .as_array()
                .context("authority has not proven original captured file coverage")?
            {
                let path =
                    PathBuf::from(file["path"].as_str().context("captured file path absent")?);
                let path = if path.is_absolute() {
                    path.strip_prefix(root)?.to_path_buf()
                } else {
                    path
                };
                let sha = file["sha256"]
                    .as_str()
                    .context("captured file digest absent")?
                    .to_owned();
                ensure!(
                    file["bytes"].as_u64() == Some(fs::metadata(relative(root, &path)?)?.len()),
                    "captured file byte count differs"
                );
                ensure!(
                    captured_files
                        .insert(path, sha.clone())
                        .is_none_or(|v| v == sha),
                    "captured file has conflicting historical identity"
                );
            }
        }
        previous = Some(publication["final_event_commit"].clone());
    }
    if kind == "winner_publish" {
        if !publications.is_empty() {
            ensure!(
                result["cursor_after"]["last_local_commit"]
                    == previous.clone().context("final winner commit absent")?,
                "winner cursor final commit changed"
            );
        }
        ensure!(
            result["cursor_before"].is_object() && result["cursor_after"].is_object(),
            "winner cursor proof absent"
        );
        let before = result["cursor_before"]["next_sequence"]
            .as_u64()
            .context("winner initial cursor")?;
        let after = result["cursor_after"]["next_sequence"]
            .as_u64()
            .context("winner final cursor")?;
        ensure!(
            after.checked_sub(before) == Some(captures.len() as u64),
            "winner cursor skipped strict identities"
        );
    } else {
        ensure!(
            result["local_corpus_committed"] == true && !publications.is_empty(),
            "corpus has no independently verified local commit"
        );
        for (path, sha) in &outputs {
            ensure!(
                captured_files.get(path) == Some(sha),
                "corpus/input dependency lacks original captured Git closure: {}",
                path.display()
            );
        }
        verify_dataset_corpus(root, &request["operation"], result, &outputs)?;
    }
    ensure!(
        std::time::Instant::now() <= deadline,
        "dataset closure verification exceeded300seconds"
    );
    Ok(())
}
fn verify_pending_publications(
    request: &Value,
    result: &Value,
    executable: &fs::File,
    deadline: std::time::Instant,
) -> Result<()> {
    ensure!(
        result["scope"] == "remote" && result["captures"].as_array().is_some_and(|v| v.is_empty()),
        "pending publication cannot impersonate a new capture"
    );
    let limit = request["operation"]["limit"]
        .as_u64()
        .context("pending publication limit")?;
    let rows = result["publications"]
        .as_array()
        .context("pending publication evidence absent")?;
    let settled = result["status"]["settled_publications_this_attempt"]
        .as_u64()
        .context("settled publication count absent")?;
    ensure!(
        settled <= limit && rows.len() as u64 <= settled && rows.len() <= 4096,
        "pending publication count changed"
    );
    let store = PathBuf::from(request["store"].as_str().context("publication store")?);
    ensure!(
        store.is_absolute()
            && fs::canonicalize(&store)? == store
            && result["store"] == request["store"],
        "publication store changed"
    );
    let mut seen = BTreeSet::new();
    for row in rows {
        let path = PathBuf::from(row["chain_manifest_path"].as_str().context("chain path")?);
        let saved = PathBuf::from(
            row["published_receipt_path"]
                .as_str()
                .context("publication receipt path")?,
        );
        ensure!(
            path.starts_with(&store)
                && saved.starts_with(&store)
                && fs::canonicalize(&path)? == path
                && fs::canonicalize(&saved)? == saved,
            "publication artifact escaped fixed store"
        );
        let sha = row["chain_manifest_sha256"]
            .as_str()
            .context("chain digest")?;
        let verified = dataset_chain_command(executable, &path, sha, deadline)?;
        let bytes = read_limit(&saved, 256 * 1024 * 1024)?;
        ensure!(
            hash(&bytes) == row["published_receipt_sha256"],
            "original published receipt changed"
        );
        let original: Value = serde_json::from_slice(&bytes)?;
        let chain: Value = serde_json::from_slice(&read_limit(&path, 256 * 1024 * 1024)?)?;
        ensure!(
            hash(&read_limit(&path, 256 * 1024 * 1024)?) == sha
                && chain["store_root"] == request["store"],
            "original publication chain/store changed"
        );
        for field in [
            "repository",
            "state_directory",
            "lfs_binary",
            "lfs_binary_sha256",
            "scratch_directory",
        ] {
            ensure!(
                chain["target"][field] == request["publication"][field],
                "pending publication target changed"
            );
        }
        ensure!(
            original == row["result"]
                && original["schema"] == "jain.publication.chain_receipt.v1"
                && original["verified_download"] == true
                && original["chain_manifest_sha256"] == sha
                && original["final_event_commit"] == verified["final_event_commit"]
                && original["final_event_ref"] == verified["final_event_ref"]
                && original["producer"] == verified["producer"]
                && original["sequence"] == verified["sequence"]
                && seen.insert((
                    original["producer"].to_string(),
                    original["sequence"].to_string()
                )),
            "recorded publication evidence differs from independently verified local identity"
        );
        // This validates original successful remote evidence, not current forge state.
    }
    Ok(())
}

fn verify_continuation_corpus(
    root: &Path,
    operation: &Value,
    result: &Value,
    outputs: &BTreeMap<PathBuf, String>,
) -> Result<()> {
    let (path, sha256) = dataset_binding(root, &operation["target_set"])?;
    let target = Artifact { path, sha256 };
    let set: crate::continuation_corpus::TargetSet = crate::native_genome::load(root, &target)?;
    ensure!(
        !set.qualification,
        "qualification target corpus cannot enter production publication"
    );
    let (closure_path, closure_sha) = dataset_binding(root, &operation["closure"])?;
    let bytes = read_limit(&relative(root, &closure_path)?, 256 * 1024 * 1024)?;
    ensure!(
        hash(&bytes) == closure_sha,
        "continuation source closure changed"
    );
    let mut closure = BTreeMap::new();
    for line in bytes.split(|b| *b == b'\n').filter(|b| !b.is_empty()) {
        ensure!(
            line.len() <= 8192 && closure.len() < 1_000_000,
            "continuation closure bounds"
        );
        let (path, sha) = dataset_binding(root, &serde_json::from_slice::<Value>(line)?)?;
        ensure!(
            closure.insert(path, sha).is_none(),
            "duplicate continuation source identity"
        );
    }
    let required = crate::continuation_corpus::input_closure(
        root,
        &target,
        &relative(root, &set.source_graph_state)?,
    )?;
    for (path, sha) in required {
        ensure!(
            closure.get(&path) == Some(&sha) && outputs.get(&path) == Some(&sha),
            "continuation original dependency missing from captured closure: {}",
            path.display()
        );
    }
    let output = Path::new(
        operation["output"]
            .as_str()
            .context("continuation output path")?,
    );
    relative(root, output)?;
    let expected = crate::continuation_corpus::render(root, &target)?;
    for (name, bytes) in expected {
        let path = output.join(name);
        let sha = hash(&bytes);
        ensure!(
            outputs.get(&path) == Some(&sha) && digest_file(&relative(root, &path)?)? == sha,
            "continuation corpus differs from independently reconstructed strict targets"
        );
    }
    let index = output.join("index.json");
    ensure!(
        result["corpus_index"]
            == json!({"path":index,"sha256":outputs.get(&index).context("continuation index absent")?})
            && result["corpus_index_sha256"] == outputs[&index],
        "continuation index receipt identity differs"
    );
    Ok(())
}

fn verify_dataset_corpus(
    root: &Path,
    operation: &Value,
    result: &Value,
    outputs: &BTreeMap<PathBuf, String>,
) -> Result<()> {
    if operation["kind"] == "pack_continuation_and_publish" {
        return verify_continuation_corpus(root, operation, result, outputs);
    }
    let packing = operation["kind"] == "pack_and_publish";
    let corpus = Path::new(
        operation[if packing { "output" } else { "corpus" }]
            .as_str()
            .context("corpus path absent")?,
    );
    let directory = relative(root, corpus)?;
    let index_path = corpus.join("index.json");
    let index_sha = outputs
        .get(&index_path)
        .context("corpus index not in captured output closure")?;
    let bytes = read_limit(&directory.join("index.json"), 256 * 1024 * 1024)?;
    ensure!(
        hash(&bytes) == *index_sha && result["corpus_index_sha256"] == *index_sha,
        "corpus index identity differs"
    );
    let index: crate::teaching_store::Manifest = serde_json::from_slice(&bytes)?;
    ensure!(
        index.schema == crate::teaching_store::SCHEMA
            && !index.qualification
            && index.label_policy == "strict_midrank_v4"
            && index.records > 0
            && index.record_bytes == crate::corpus::RECORD_BYTES
            && index.records_sha256 == hash(&serde_json::to_vec(&index.shards)?),
        "unsupported or changed production teaching corpus"
    );
    let closure_binding = dataset_binding(root, &operation["closure"])?;
    let closure_bytes = read_limit(&relative(root, &closure_binding.0)?, 256 * 1024 * 1024)?;
    let mut closure = BTreeMap::new();
    for line in closure_bytes
        .split(|b| *b == b'\n')
        .filter(|b| !b.is_empty())
    {
        ensure!(
            line.len() <= 1024 * 1024 && closure.len() < 200000,
            "input closure bound"
        );
        let row: Value = serde_json::from_slice(line)?;
        let (path, sha) = dataset_binding(root, &row)?;
        ensure!(
            closure.insert(path, sha).is_none(),
            "duplicate original input closure"
        );
    }
    ensure!(
        !closure.is_empty(),
        "original corpus dependency closure empty"
    );
    if packing {
        for key in ["contexts", "vocabulary", "observations", "admission"] {
            let (path, sha) = dataset_binding(root, &operation[key])?;
            ensure!(
                closure.get(&path) == Some(&sha),
                "packer input absent from original dependency closure"
            );
        }
        ensure!(
            index.context_index_sha256 == operation["contexts"]["sha256"]
                && index.vocabulary_sha256 == operation["vocabulary"]["sha256"],
            "packed corpus input identities changed"
        );
        let observation_path = Path::new(
            operation["observations"]["path"]
                .as_str()
                .context("observation manifest absent")?,
        );
        let observations: Vec<PathBuf> = serde_json::from_slice(&read_limit(
            &relative(root, observation_path)?,
            256 * 1024 * 1024,
        )?)?;
        ensure!(!observations.is_empty(), "no observations packed");
        for path in observations {
            let suffix = if path.is_absolute() {
                path.strip_prefix(root)?.to_path_buf()
            } else {
                path
            };
            ensure!(
                closure.contains_key(&suffix),
                "observation outside original admitted closure"
            );
        }
    } else {
        ensure!(
            closure
                .values()
                .any(|sha| sha == &index.context_index_sha256)
                && closure.values().any(|sha| sha == &index.vocabulary_sha256),
            "published context index/vocabulary absent from original closure"
        );
        ensure!(
            index.context_index_sha256 == operation["contexts_sha256"]
                && index.vocabulary_sha256 == operation["vocabulary_sha256"]
                && *index_sha == operation["index_sha256"],
            "published corpus identities changed"
        );
    }
    let mut required = BTreeMap::new();
    required.insert(
        corpus.join("vocabulary.json"),
        index.vocabulary_sha256.clone(),
    );
    required.insert(corpus.join("groups.bin"), index.groups_sha256.clone());
    let provenance: Value = serde_json::from_slice(&read_limit(
        &directory.join("provenance.json"),
        256 * 1024 * 1024,
    )?)?;
    ensure!(
        provenance["index_sha256"] == *index_sha
            && outputs.contains_key(&corpus.join("provenance.json")),
        "corpus provenance not captured/bound"
    );
    let mut count = 0usize;
    let mut offsets = Vec::new();
    let mut previous = None;
    for shard in &index.shards {
        ensure!(
            previous.is_none_or(|v| v < shard.context) && shard.records >= 8,
            "shard order/count invalid"
        );
        previous = Some(shard.context);
        let context = index
            .contexts
            .get(shard.context)
            .context("shard context absent")?;
        ensure!(
            matches!(context.partition.as_str(), "meta_train" | "meta_validation"),
            "sealed context admitted into teaching shard"
        );
        let path = relative(root, &shard.path)?;
        for (name, sha) in [
            ("shard.json", &shard.sha256),
            ("records.bin", &shard.records_sha256),
            ("labels.jsonl", &shard.labels_sha256),
        ] {
            required.insert(shard.path.join(name), sha.clone());
        }
        let saved: crate::teaching_store::Shard =
            serde_json::from_slice(&read_limit(&path.join("shard.json"), 256 * 1024 * 1024)?)?;
        let mut expected = shard.clone();
        expected.sha256.clear();
        ensure!(
            serde_json::to_value(saved)? == serde_json::to_value(expected)?,
            "shard metadata differs from parent"
        );
        ensure!(
            fs::metadata(path.join("records.bin"))?.len()
                == u64::try_from(
                    shard
                        .records
                        .checked_mul(index.record_bytes)
                        .context("record bytes overflow")?
                )?,
            "teaching shard byte dimensions differ"
        );
        offsets.extend((count as u64).to_le_bytes());
        offsets.extend((shard.records as u64).to_le_bytes());
        count = count
            .checked_add(shard.records)
            .context("corpus count overflow")?;
        ensure!(
            closure.get(&context.path) == Some(&context.sha256)
                && closure.get(&context.prepared.join("prepared.json"))
                    == Some(&context.prepared_sha256),
            "context/prepared dependency absent from original closure"
        );
    }
    ensure!(
        count == index.records && hash(&offsets) == index.groups_sha256,
        "teaching record/group totals changed"
    );
    for (path, sha) in required {
        ensure!(
            outputs.get(&path) == Some(&sha) && digest_file(&relative(root, &path)?)? == sha,
            "required corpus dependency not captured or changed"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn ambient_graph_directory_cannot_dispatch_outside_reviewed_loop() {
        let root = tempfile::tempdir().unwrap();
        let graph = Graph::open(
            root.path(),
            &root.path().join("graph"),
            &root.path().join("worker"),
            TenantId::new("system:jope").unwrap(),
            ProjectId::new("test").unwrap(),
        )
        .unwrap();
        fs::create_dir(root.path().join("graph/admissions")).unwrap();
        let path = root.path().join("graph/admissions/operator-written.json");
        fs::write(&path, b"preserved unreviewed admission").unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let error = runtime
            .block_on(graph.tick(&Client::new(root.path().join("absent.sock"))))
            .err()
            .unwrap();
        assert!(error
            .to_string()
            .contains("independently approved native Loop"));
        assert_eq!(fs::read(path).unwrap(), b"preserved unreviewed admission");
        assert!(!root.path().join("graph/tasks").exists());
    }

    use super::*;
    #[test]
    fn immutable_replay_preserves_original_and_retains_conflict() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("new/nested/record.json");
        immutable(&path, b"original").unwrap();
        immutable(&path, b"original").unwrap();
        assert!(immutable(&path, b"conflict").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"original");
        assert!(fs::read_dir(path.parent().unwrap()).unwrap().count() >= 2);
    }
    #[test]
    fn dependency_path_rejects_symlink_and_parent_escape() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("receipt"), b"{}").unwrap();
        std::os::unix::fs::symlink("receipt", root.path().join("alias")).unwrap();
        assert!(relative(root.path(), Path::new("../receipt")).is_err());
        assert!(relative(root.path(), Path::new("alias")).is_err());
        assert!(relative(root.path(), Path::new("receipt")).is_ok());
    }
    #[test]
    fn delivery_ack_and_idle_worker_are_never_scientific_completion() {
        let root = tempfile::tempdir().unwrap();
        assert!(
            verify_numerical(root.path(), &Operation::Verify, &json!({"accepted":true})).is_err()
        );
        assert!(
            verify_numerical(root.path(), &Operation::Resident, &json!({"verified":true})).is_err()
        );
        assert!(verify_numerical(
            root.path(),
            &Operation::Verify,
            &json!({"schema":"jope.native_verification.v1","verified":true})
        )
        .is_ok());
    }
    #[test]
    fn graph_requires_exclusive_coordinator_and_physical_root() {
        let root = tempfile::tempdir().unwrap();
        let open = || {
            Graph::open(
                root.path(),
                &root.path().join("graph"),
                &root.path().join("worker"),
                TenantId::new("system:jope").unwrap(),
                ProjectId::new("test").unwrap(),
            )
        };
        let first = open().unwrap();
        assert!(open().is_err());
        drop(first);
        assert!(open().is_ok());
    }
    #[test]
    fn physical_state_paths_reject_parent_and_symlink_ancestors() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let tenant = TenantId::new("system:jope").unwrap();
        let project = ProjectId::new("test").unwrap();
        assert!(Graph::open(
            root.path(),
            &root.path().join("../escape"),
            &root.path().join("worker"),
            tenant.clone(),
            project.clone()
        )
        .is_err());
        std::os::unix::fs::symlink(outside.path(), root.path().join("alias")).unwrap();
        assert!(Graph::open(
            root.path(),
            &root.path().join("alias/graph"),
            &root.path().join("worker"),
            tenant,
            project
        )
        .is_err());
        assert!(!outside.path().join("graph").exists());
    }
    #[test]
    fn timing_only_proposal_completion_cannot_prove_generated_artifacts() {
        let root = tempfile::tempdir().unwrap();
        let operation = Operation::Proposal {
            preparation_overlay: None,
            output: "proposal".into(),
            corpus: "corpus".into(),
            checkpoint: "checkpoint".into(),
            champion_snapshot: "champion.json".into(),
            cuts: 1,
            cut_start: 0,
            shuffled: false,
        };
        fs::create_dir(root.path().join("proposal")).unwrap();
        let result = json!({"schema":"jope.smartcluster_proposal_completion.v1","numerical_receipt":{"rank":0,"world":1,"seconds":0.1}});
        assert!(verify_numerical(root.path(), &operation, &result).is_err());
    }
    #[test]
    fn later_declared_gp_outputs_do_not_exempt_original_proposal_bytes() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("search")).unwrap();
        fs::write(root.path().join("seeds.json"), b"original").unwrap();
        let artifacts = BTreeMap::from([(PathBuf::from("seeds.json"), hash(b"original"))]);
        fs::write(root.path().join("search/later-result"), b"child result").unwrap();
        let future = [root.path().join("search")];
        verify_tree(root.path(), &artifacts, &future).unwrap();
        assert!(verify_tree(root.path(), &artifacts, &[]).is_err());
        fs::write(root.path().join("seeds.json"), b"corrupt").unwrap();
        assert!(verify_tree(root.path(), &artifacts, &future).is_err());
    }
    #[test]
    fn incomplete_pack_cannot_pass_with_only_an_arbitrary_artifact() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("pack")).unwrap();
        fs::write(root.path().join("pack/arbitrary"), b"x").unwrap();
        let operation = Operation::Pack {
            output: "pack".into(),
            contexts: "contexts.json".into(),
            vocabulary: "vocabulary.json".into(),
            observations: vec![],
            qualification: true,
        };
        assert!(verify_numerical(
            root.path(),
            &operation,
            &json!({"schema":"jope.native_pack_completion.v1","artifacts":{"arbitrary":hash(b"x")}})
        )
        .is_err());
    }
    fn dataset_corpus_fixture() -> (tempfile::TempDir, Value, Value, BTreeMap<PathBuf, String>) {
        let root = tempfile::tempdir().unwrap();
        let r = root.path();
        for d in ["corpus", "shard", "prepared"] {
            fs::create_dir(r.join(d)).unwrap();
        }
        let mut outputs = BTreeMap::new();
        let mut put = |path: &str, bytes: &[u8]| {
            fs::write(r.join(path), bytes).unwrap();
            let sha = hash(bytes);
            outputs.insert(PathBuf::from(path), sha.clone());
            sha
        };
        let context_sha = put("context.bin", b"context");
        let prepared_sha = put("prepared/prepared.json", b"{}");
        let contexts_sha = put("contexts.json", b"[]");
        let vocabulary_sha = put("corpus/vocabulary.json", b"{}");
        let records_sha = put(
            "shard/records.bin",
            &vec![0; 8 * crate::corpus::RECORD_BYTES],
        );
        let labels_sha = put("shard/labels.jsonl", b"{}\n");
        let mut shard = crate::teaching_store::Shard {
            context: 0,
            path: "shard".into(),
            sha256: String::new(),
            records: 8,
            max_length: 1,
            records_sha256: records_sha,
            labels_sha256: labels_sha,
            legal_sets: vec![vec![0]],
        };
        shard.sha256 = put("shard/shard.json", &serde_json::to_vec(&shard).unwrap());
        let mut groups = Vec::new();
        groups.extend(0u64.to_le_bytes());
        groups.extend(8u64.to_le_bytes());
        let groups_sha = put("corpus/groups.bin", &groups);
        let shards = vec![shard];
        let index = json!({"schema":"parallel.teaching_shards.v1","qualification":false,"label_policy":"strict_midrank_v4","vocabulary_sha256":vocabulary_sha,"context_index_sha256":contexts_sha,"records":8,"record_bytes":crate::corpus::RECORD_BYTES,"records_sha256":hash(&serde_json::to_vec(&shards).unwrap()),"groups_sha256":groups_sha,"shards":shards,"sources":[],"counts":{},"contexts":[{"dataset":"fixture","family":"fixture","cut":1,"partition":"meta_train","path":"context.bin","sha256":context_sha,"prepared":"prepared","prepared_sha256":prepared_sha}]});
        let index_sha = put("corpus/index.json", &serde_json::to_vec(&index).unwrap());
        put(
            "corpus/provenance.json",
            &serde_json::to_vec(&json!({"index_sha256":index_sha})).unwrap(),
        );
        let closure = format!(
            "{}\n{}\n{}\n{}\n",
            json!({"path":"context.bin","sha256":context_sha}),
            json!({"path":"prepared/prepared.json","sha256":prepared_sha}),
            json!({"path":"contexts.json","sha256":contexts_sha}),
            json!({"path":"corpus/vocabulary.json","sha256":vocabulary_sha})
        );
        let closure_sha = put("closure.jsonl", closure.as_bytes());
        let operation = json!({"kind":"corpus_publish","corpus":"corpus","index_sha256":index_sha,"contexts_sha256":contexts_sha,"vocabulary_sha256":vocabulary_sha,"closure":{"path":"closure.jsonl","sha256":closure_sha},"qualification":false});
        (
            root,
            operation,
            json!({"corpus_index_sha256":index_sha}),
            outputs,
        )
    }
    #[test]
    fn dataset_corpus_requires_actual_record_and_label_closure() {
        let (root, operation, result, outputs) = dataset_corpus_fixture();
        verify_dataset_corpus(root.path(), &operation, &result, &outputs).unwrap();
        fs::write(
            root.path().join("shard/records.bin"),
            vec![1; 8 * crate::corpus::RECORD_BYTES],
        )
        .unwrap();
        assert!(verify_dataset_corpus(root.path(), &operation, &result, &outputs).is_err());
    }
    #[test]
    fn dataset_corpus_rejects_omitted_captured_shard() {
        let (root, operation, result, mut outputs) = dataset_corpus_fixture();
        outputs.remove(Path::new("shard/labels.jsonl"));
        assert!(verify_dataset_corpus(root.path(), &operation, &result, &outputs).is_err());
    }
    #[test]
    fn dataset_receipt_cannot_stand_in_for_a_corpus() {
        let root = tempfile::tempdir().unwrap();
        assert!(verify_dataset_corpus(
            root.path(),
            &json!({"kind":"corpus_publish","corpus":"missing"}),
            &json!({"passed":true,"local_corpus_committed":true}),
            &BTreeMap::new()
        )
        .is_err());
    }
    #[test]
    fn read_only_verifier_cannot_dispatch_or_create_coordinator_state() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("graph");
        let worker = root.path().join("worker");
        fs::create_dir(&directory).unwrap();
        fs::create_dir(&worker).unwrap();
        let graph = Graph::read_only(
            root.path(),
            &directory,
            &worker,
            TenantId::new("system:jope").unwrap(),
            ProjectId::new("proof").unwrap(),
        )
        .unwrap();
        assert!(!directory.join("coordinator.lock").exists());
        let client = Client::new(root.path().join("missing.sock"));
        let admission = Admission {
            schema: "jope.native_graph_admission.v1".into(),
            manifest: Artifact {
                path: "task.json".into(),
                sha256: "a".repeat(64),
            },
            predecessors: BTreeMap::new(),
            delivery: Delivery::NativeJob,
        };
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                assert!(graph
                    .admit(&client, &admission)
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("read-only"));
                assert!(graph
                    .tick(&client)
                    .await
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("read-only"));
            });
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 0);
        assert_eq!(fs::read_dir(&worker).unwrap().count(), 0);
    }
}
