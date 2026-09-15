//! Finite scientific eligibility controller. SmartCluster alone allocates and executes jobs.
//! Frozen requests are reviewable artifacts. External signed authority grants remain mandatory.
mod grant_proof;
pub(crate) mod materialize;
mod run_approval;
use crate::{
    manifest::hash,
    native_genome::{self, Arm, Candidate, GenomeTask, Stage},
    native_graph::{Admission, Artifact, Delivery, Graph, VerifiedResult},
    smartcluster_worker::{ClosureManifest, InputManifest, Operation},
};
use anyhow::{ensure, Context, Result};
pub use grant_proof::{GrantProofInputs, NativeGrantEligibilityProof};
use jain_smartcluster::{
    jope::{TaskKind, TaskManifest},
    Client, IdempotencyKey, JobId, ResourceRequest, Timestamp,
};
pub use materialize::Settings as MaterializationSettings;
pub(crate) use run_approval::verify_installed_run_signature;
pub use run_approval::{OriginalAllocation, OriginalJobCensus, RunApprovalBody, SignedRunApproval};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::{Path, PathBuf},
};
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema: String,
    pub run: String,
    pub accepted_checkpoint: Artifact,
    pub initial_corpus: Artifact,
    pub accepted_decision: Artifact,
    pub scientific_policy: Artifact,
    pub label_admission_policy: Artifact,
    pub fit_policy: PathBuf,
    /// Independently signed root-protected approval; path only avoids a config digest cycle.
    pub run_approval: PathBuf,
    pub resident_job: JobId,
    pub workers: usize,
    pub cycles: u64,
    pub initial_optimizer_step: u64,
    /// Exact immutable completion of a separately reviewed preceding two-block run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predecessor_run: Option<Artifact>,
    pub updates_per_block: u64,
    pub global_batch: usize,
    pub microbatch: usize,
    pub direct_seconds: u64,
    pub continuation_evaluations: usize,
    pub continuation_seconds: u64,
    pub cpu_resources: ResourceRequest,
    pub task_deadline: Timestamp,
    pub qualification: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation_overlay: Option<Artifact>,
    pub cuts: usize,
    pub cut_start: usize,
    #[serde(default)]
    pub materialization: Option<MaterializationSettings>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunCompletion {
    pub schema: String,
    pub run: String,
    pub config: Artifact,
    pub initial_optimizer_step: u64,
    pub target_optimizer_step: u64,
    pub checkpoint: Artifact,
    pub corpus: Artifact,
    pub accepted_parent_decision: Artifact,
    pub final_proposals: BTreeMap<Arm, Artifact>,
}
fn verify_successor(next: &Config, previous: &Config, completion: &RunCompletion) -> Result<()> {
    previous.validate()?;
    ensure!(completion.schema == "jope.native_loop_completion.v1"
        && completion.run == previous.run && next.run != previous.run
        && completion.initial_optimizer_step == previous.initial_optimizer_step
        && completion.target_optimizer_step == previous.initial_optimizer_step + 6000
        && next.initial_optimizer_step == completion.target_optimizer_step
        && next.accepted_checkpoint == completion.checkpoint && next.initial_corpus == completion.corpus
        && completion.final_proposals.keys().copied().collect::<BTreeSet<_>>() == BTreeSet::from([Arm::Model, Arm::Shuffled]),
        "successor must preserve a different completed run and its exact accepted optimizer/corpus window");
    ensure!(
        next.scientific_policy == previous.scientific_policy
            && next.label_admission_policy == previous.label_admission_policy
            && next.qualification == previous.qualification
            && next.cuts == previous.cuts
            && next.cut_start == previous.cut_start
            && next.direct_seconds == previous.direct_seconds
            && next.continuation_evaluations == previous.continuation_evaluations
            && next.continuation_seconds == previous.continuation_seconds,
        "successor cannot silently change scientific protocol, partition or numerical budgets"
    );
    Ok(())
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScientificPolicy {
    pub schema: String,
    pub training_target: String,
    pub training_arms: BTreeSet<Arm>,
    pub excluded_families: BTreeSet<String>,
    pub sealed_partition: String,
    pub family_partition_receipt: Artifact,
    pub exclusion_receipt: Artifact,
    pub matched_arms: BTreeSet<Arm>,
    pub proposal_seed_protocol: String,
    pub require_remote_publication: bool,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    ProposalModel,
    ProposalShuffled,
    Direct,
    Continuation,
    PackAndPublish,
    Aggregate,
    Train,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema: String,
    pub run: String,
    pub generation: u64,
    pub id: String,
    pub role: Role,
    pub parameters: Value,
    pub predecessors: BTreeMap<String, Artifact>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub schema: String,
    pub request: Artifact,
    pub admission: Admission,
}
#[derive(Default, Serialize)]
pub struct Progress {
    pub completed_transitions: u64,
    pub offered: usize,
    pub waiting_execution: usize,
    pub waiting_materialization: usize,
    pub waiting_authority: usize,
    pub pending_direct: usize,
    pub pending_continuation: usize,
    pub complete: bool,
    pub idle_reasons: BTreeSet<String>,
}
pub struct Loop {
    root: PathBuf,
    state: PathBuf,
    graph_state: PathBuf,
    config: Config,
    policy: ScientificPolicy,
    _lock: fs::File,
    read_only: bool,
}
fn now() -> Result<u64> {
    crate::resident_protocol::now_ms()
}
fn read(path: &Path) -> Result<Vec<u8>> {
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    ensure!(
        f.metadata()?.is_file() && f.metadata()?.len() <= 16 * 1024 * 1024,
        "loop record type/size"
    );
    let mut b = Vec::new();
    f.take(16 * 1024 * 1024 + 1).read_to_end(&mut b)?;
    ensure!(b.len() <= 16 * 1024 * 1024, "loop record growth");
    Ok(b)
}
fn physical_directory(root: &Path, path: &Path) -> Result<()> {
    let suffix = path
        .strip_prefix(root)
        .context("loop state outside canonical root")?;
    crate::manifest::relative(suffix)?;
    let mut at = root.to_path_buf();
    for part in suffix.components() {
        at.push(part);
        match fs::symlink_metadata(&at) {
            Ok(m) => ensure!(
                m.is_dir() && !m.file_type().is_symlink(),
                "loop state symlink/non-directory"
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&at)?;
                fs::File::open(at.parent().unwrap())?.sync_all()?;
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
fn immutable(root: &Path, path: &Path, bytes: &[u8]) -> Result<Artifact> {
    ensure!(
        bytes.len() <= 256 * 1024 * 1024,
        "loop artifact exceeds 256 MiB"
    );
    let parent = path.parent().context("loop record parent")?;
    physical_directory(root, parent)?;
    let (mut f, tmp) = tempfile::Builder::new()
        .prefix(".loop-pending-")
        .tempfile_in(parent)?
        .keep()?;
    f.write_all(bytes)?;
    f.sync_all()?;
    match crate::native_persistence::publish_staged(&tmp, path) {
        Ok(()) => (),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(path)?;
            ensure!(
                existing.metadata()?.is_file() && existing.metadata()?.len() == bytes.len() as u64,
                "conflicting original loop record type/length"
            );
            let mut old = Vec::new();
            existing
                .take(bytes.len() as u64 + 1)
                .read_to_end(&mut old)?;
            ensure!(old == bytes, "conflicting original loop record")
        }
        Err(e) => return Err(e.into()),
    }
    fs::File::open(parent)?.sync_all()?;
    crate::native_persistence::discard_staging(&tmp)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(Artifact {
        path: path.strip_prefix(root)?.to_owned(),
        sha256: hash(bytes),
    })
}
fn frozen<T: Serialize>(root: &Path, path: &Path, value: &T) -> Result<Artifact> {
    immutable(root, path, &serde_json::to_vec(value)?)
}
fn artifact(root: &Path, path: &Path) -> Result<Artifact> {
    let b = read(path)?;
    Ok(Artifact {
        path: path.strip_prefix(root)?.to_owned(),
        sha256: hash(&b),
    })
}
fn identity(run: &str, generation: u64, id: &str) -> Result<String> {
    Ok(hash(&serde_json::to_vec(&(run, generation, id))?))
}
pub fn generation_seed(run: &str, generation: u64, checkpoint: &str) -> Result<u64> {
    let h = hash(&serde_json::to_vec(&(
        "jope.generation_seed.v1",
        run,
        generation,
        checkpoint,
    ))?);
    Ok(u64::from_str_radix(&h[..16], 16)?)
}
impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "jope.native_loop.v1"
                && !self.run.is_empty()
                && self.cycles == 2
                && ((self.predecessor_run.is_none() && self.initial_optimizer_step == 60000)
                    || (self.predecessor_run.is_some() && self.initial_optimizer_step > 60000
                        && (self.initial_optimizer_step - 60000) % 6000 == 0))
                && self.initial_optimizer_step.checked_add(6000).is_some()
                && self.updates_per_block == 3000
                && self.global_batch == 2048,
            "native loop requires original60000 or an explicit completed predecessor and two3000-update blocks"
        );
        ensure!(
            [48, 72, 96].contains(&self.workers)
                && [64, 128, 256].contains(&self.microbatch)
                && self.global_batch % self.microbatch == 0
                && self.cuts > 0,
            "unqualified loop concurrency/batch/cut configuration"
        );
        ensure!(
            self.direct_seconds > 0
                && self.direct_seconds <= 600
                && self.continuation_evaluations > 0
                && self.continuation_evaluations <= 1000
                && self.continuation_seconds > 0
                && self.continuation_seconds <= 600,
            "loop numerical budget invalid"
        );
        if self.preparation_overlay.is_some() {
            ensure!(
                self.continuation_evaluations == 24 && self.continuation_seconds == 54,
                "signed first-window overlay requires exactly 24 continuation fits and 54 seconds"
            );
        }
        Ok(())
    }
}
impl Loop {
    pub fn open(root: &Path, state: &Path, graph_state: &Path, config: &Artifact) -> Result<Self> {
        Self::open_mode(root, state, graph_state, config, false)
    }
    fn open_mode(
        root: &Path,
        state: &Path,
        graph_state: &Path,
        config: &Artifact,
        read_only: bool,
    ) -> Result<Self> {
        ensure!(
            root.is_absolute() && fs::canonicalize(root)? == root,
            "loop requires canonical physical root"
        );
        let cfg: Config = native_genome::load(root, config)?;
        cfg.validate()?;
        let policy: ScientificPolicy = native_genome::load(root, &cfg.scientific_policy)?;
        ensure!(
            policy.schema == "jope.native_learning_policy.v1"
                && policy.training_target == "continuation_only_v1"
                && policy.sealed_partition == "meta_test"
                && policy.proposal_seed_protocol == "generation_sha256_v1"
                && policy.matched_arms == BTreeSet::from([Arm::Model, Arm::Generic, Arm::Shuffled])
                && !policy.training_arms.is_empty(),
            "scientific objective, matched arms or sealed policy not explicitly frozen"
        );
        let _: Value = native_genome::load(root, &policy.family_partition_receipt)?;
        let _: Value = native_genome::load(root, &policy.exclusion_receipt)?;
        let accepted: Value = native_genome::load(root, &cfg.accepted_decision)?;
        ensure!(
            accepted["schema"] == "jope.initial_proposal_parent_decision.v1"
                && accepted["checkpoint"] == serde_json::to_value(&cfg.accepted_checkpoint)?
                && accepted["corpus"] == serde_json::to_value(&cfg.initial_corpus)?
                && accepted["accepted"] == true
                && accepted["optimizer_step"] == cfg.initial_optimizer_step
                && (cfg.predecessor_run.is_none()
                    || accepted["predecessor_run"] == serde_json::to_value(&cfg.predecessor_run)?),
            "initial accepted decision does not bind original checkpoint/corpus"
        );
        if let Some(a) = &cfg.preparation_overlay {
            ensure!(
                !cfg.qualification
                    && cfg.initial_optimizer_step == 60000
                    && cfg.cut_start == 0
                    && cfg.cuts == 2,
                "first-window overlay cannot extend or relabel its reviewed scope"
            );
            crate::preparation_overlay::load(root, a, &cfg.run)?;
        }
        let checkpoint: Value = native_genome::load(root, &cfg.accepted_checkpoint)?;
        let _: Value = native_genome::load_corpus_index(root, &cfg.initial_corpus)?;
        ensure!(
            checkpoint["step"] == cfg.initial_optimizer_step
                && checkpoint["corpus_sha256"] == cfg.initial_corpus.sha256,
            "accepted checkpoint/corpus/window changed"
        );
        let cp = cfg
            .accepted_checkpoint
            .path
            .parent()
            .context("checkpoint parent")?;
        for (file, key) in [
            ("model.ot", "model_sha256"),
            ("optimizer.ot", "optimizer_sha256"),
        ] {
            ensure!(
                crate::manifest::file_hash(&crate::manifest::resolve(root, &cp.join(file))?)?
                    == checkpoint[key],
                "accepted model or optimizer bytes changed"
            );
        }
        if read_only {
            ensure!(
                fs::canonicalize(state)? == state
                    && state.starts_with(root)
                    && fs::canonicalize(graph_state)? == graph_state
                    && graph_state.starts_with(root),
                "existing physical read-only loop state required"
            );
            let frozen_config: Config = serde_json::from_slice(&read(&state.join("config.json"))?)?;
            ensure!(
                serde_json::to_vec(&frozen_config)? == serde_json::to_vec(&cfg)?,
                "read-only proof config differs from original frozen loop"
            );
            return Ok(Self {
                root: root.into(),
                state: state.into(),
                graph_state: graph_state.into(),
                config: cfg,
                policy,
                _lock: fs::File::open(state)?,
                read_only: true,
            });
        }
        physical_directory(root, state)?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(state.join("loop.lock"))?;
        ensure!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "another learning planner owns this run"
        );
        frozen(root, &state.join("config.json"), &cfg)?;
        Ok(Self {
            root: root.into(),
            state: state.into(),
            graph_state: graph_state.into(),
            config: cfg,
            policy,
            _lock: lock,
            read_only: false,
        })
    }
    fn request(
        &self,
        generation: u64,
        id: String,
        role: Role,
        parameters: Value,
        predecessors: BTreeMap<String, Artifact>,
    ) -> Result<(Artifact, Request)> {
        let r = Request {
            schema: "jope.native_loop_request.v1".into(),
            run: self.config.run.clone(),
            generation,
            id,
            role,
            parameters,
            predecessors,
        };
        let key = identity(&r.run, generation, &r.id)?;
        Ok((
            frozen(
                &self.root,
                &self.state.join("requests").join(format!("{key}.json")),
                &r,
            )?,
            r,
        ))
    }
    /// Materializers freeze actual executable TaskManifests; this never fabricates an authority grant.
    pub fn bind(&self, binding: &Binding) -> Result<()> {
        ensure!(
            binding.schema == "jope.native_loop_binding.v1",
            "loop binding schema"
        );
        let request: Request = native_genome::load(&self.root, &binding.request)?;
        let key = identity(&request.run, request.generation, &request.id)?;
        ensure!(
            binding.request.path
                == self
                    .state
                    .join("requests")
                    .join(format!("{key}.json"))
                    .strip_prefix(&self.root)?,
            "binding is not original loop request"
        );
        self.validate_binding(&request, binding)?;
        frozen(
            &self.root,
            &self.state.join("bindings").join(format!("{key}.json")),
            binding,
        )?;
        Ok(())
    }
    /// Record an operator-reviewed quality decision, separate from training commitment and serving.
    pub fn bind_parent_decision(&self, reference: &Artifact) -> Result<()> {
        let decision: ProposalParentDecision = native_genome::load(&self.root, reference)?;
        ensure!(
            decision.schema == "jope.proposal_parent_decision.v1"
                && decision.run == self.config.run
                && decision.generation < self.config.cycles,
            "proposal parent decision scope differs"
        );
        for a in [
            &decision.checkpoint,
            &decision.previous_parent,
            &decision.training_result,
            &decision.comparison,
            &decision.assessment,
        ] {
            let _: Value = native_genome::load(&self.root, a)?;
        }
        frozen(
            &self.root,
            &self
                .state
                .join("parent-decisions")
                .join(format!("g{}.json", decision.generation)),
            reference,
        )?;
        Ok(())
    }
    fn validate_binding(&self, r: &Request, b: &Binding) -> Result<TaskManifest> {
        let original_path = self
            .state
            .join("requests")
            .join(format!("{}.json", identity(&r.run, r.generation, &r.id)?));
        ensure!(
            b.schema == "jope.native_loop_binding.v1"
                && b.request == artifact(&self.root, &original_path)?
                && native_genome::checked(&self.root, &b.request, 1024 * 1024)?
                    == serde_json::to_vec(r)?,
            "binding does not preserve the exact frozen loop request"
        );
        ensure!(
            r.run == self.config.run
                && r.generation <= self.config.cycles
                && b.admission.predecessors == r.predecessors,
            "binding changes frozen predecessor graph"
        );
        let (task, input) = crate::smartcluster_worker::verify_task_inputs(
            &self.root,
            &crate::manifest::resolve(&self.root, &b.admission.manifest.path)?,
            &b.admission.manifest.sha256,
        )?;
        ensure!(
            input.input_artifacts.get(&b.request.path) == Some(&b.request.sha256),
            "task input closure omits original scientific loop request"
        );
        ensure!(
            task.run == r.run
                && task.generation == r.generation
                && task.task_id == r.id
                && task.deadline == self.config.task_deadline,
            "binding changes run/task/original deadline"
        );
        match (&r.role, &input.operation) {
            (
                Role::ProposalModel,
                Operation::Proposal {
                    corpus,
                    checkpoint,
                    shuffled: false,
                    cuts,
                    cut_start,
                    output,
                    preparation_overlay,
                    ..
                },
            )
            | (
                Role::ProposalShuffled,
                Operation::Proposal {
                    corpus,
                    checkpoint,
                    shuffled: true,
                    cuts,
                    cut_start,
                    output,
                    preparation_overlay,
                    ..
                },
            ) => {
                ensure!(
                    b.admission.delivery
                        == Delivery::Resident {
                            job_id: self.config.resident_job
                        }
                        && task.kind == TaskKind::ResidentGpu
                        && *preparation_overlay == self.config.preparation_overlay
                        && *cuts == self.config.cuts
                        && *cut_start == self.config.cut_start
                        && task.parent_checkpoint_sha256.as_ref()
                            == r.parameters["checkpoint"]["sha256"]
                                .as_str()
                                .map(|s| s.to_string())
                                .as_ref()
                        && task.corpus_sha256.as_ref()
                            == r.parameters["corpus"]["sha256"]
                                .as_str()
                                .map(|s| s.to_string())
                                .as_ref(),
                    "proposal frozen parent/resource binding changed"
                );
                ensure!(
                    corpus.join("index.json")
                        == PathBuf::from(
                            r.parameters["corpus"]["path"]
                                .as_str()
                                .context("corpus path")?
                        )
                        && checkpoint.join("state.json")
                            == PathBuf::from(
                                r.parameters["checkpoint"]["path"]
                                    .as_str()
                                    .context("parent path")?
                            ),
                    "proposal path changed"
                );
                let run: Value = crate::manifest::read_json(&crate::manifest::resolve(
                    &self.root,
                    &output.join("run.json"),
                )?)?;
                ensure!(
                    run["proposal_seed"] == r.parameters["proposal_seed"],
                    "proposal RNG is not generation-derived"
                );
            }
            (Role::Direct, Operation::DirectEvaluation { manifest, sha256 })
            | (Role::Continuation, Operation::SeededContinuation { manifest, sha256 }) => {
                ensure!(
                    b.admission.delivery == Delivery::NativeJob
                        && r.parameters["genome_task"] == json!({"path":manifest,"sha256":sha256}),
                    "genome task changed"
                );
            }
            (Role::Aggregate, Operation::Verify) => {
                ensure!(
                    b.admission.delivery == Delivery::NativeJob
                        && task.kind == TaskKind::Verification
                        && r.parameters["members"] == serde_json::to_value(&r.predecessors)?,
                    "aggregation changed predecessor membership"
                );
            }
            (
                Role::PackAndPublish,
                Operation::PackAndPublish {
                    manifest, sha256, ..
                },
            ) => {
                ensure!(
                    b.admission.delivery == Delivery::NativeJob && task.kind == TaskKind::Packing,
                    "corpus must be native packing job"
                );
                let packet: Value = native_genome::load(
                    &self.root,
                    &Artifact {
                        path: manifest.clone(),
                        sha256: sha256.clone(),
                    },
                )?;
                ensure!(
                    packet["operation"]["kind"] == "pack_continuation_and_publish"
                        && packet["operation"]["target_set"] == r.parameters["target_set"]
                        && packet["operation"]["closure"] == r.parameters["closure"],
                    "ordinary direct corpus pack is not continuation-target publication"
                );
            }
            (
                Role::Train,
                Operation::Train {
                    corpus,
                    initialize,
                    initial_optimizer_step,
                    requested_step_increment,
                    target_optimizer_step,
                    micro,
                    replay_policy,
                    ..
                },
            ) => {
                let initial = self.config.initial_optimizer_step + r.generation * 3000;
                ensure!(
                    b.admission.delivery
                        == Delivery::Resident {
                            job_id: self.config.resident_job
                        }
                        && *initial_optimizer_step == initial
                        && *requested_step_increment == 3000
                        && *target_optimizer_step == initial + 3000
                        && *micro == self.config.microbatch
                        && initialize.join("state.json")
                            == PathBuf::from(
                                r.parameters["checkpoint"]["path"]
                                    .as_str()
                                    .context("train parent")?
                            )
                        && corpus.join("index.json")
                            == PathBuf::from(
                                r.parameters["corpus"]["path"]
                                    .as_str()
                                    .context("train corpus")?
                            ),
                    "original optimizer window/corpus changed"
                );
                let replay: Value = crate::manifest::read_json(&crate::manifest::resolve(
                    &self.root,
                    replay_policy,
                )?)?;
                let expected = materialize::training_eligibility(
                    &self.root,
                    &serde_json::from_value(r.parameters["corpus"].clone())?,
                    &serde_json::from_value(r.parameters["checkpoint"].clone())?,
                    &self.config.scientific_policy,
                    r,
                )?;
                ensure!(
                    replay == expected,
                    "external training replay differs from independently reconstructed eligibility"
                );
            }
            _ => anyhow::bail!("materialized operation differs from frozen scientific request"),
        }
        Ok(task)
    }
    fn binding(&self, r: &Request) -> Result<Option<Binding>> {
        let key = identity(&r.run, r.generation, &r.id)?;
        let path = self.state.join("bindings").join(format!("{key}.json"));
        if !path.exists() {
            return Ok(None);
        };
        let b: Binding = serde_json::from_slice(&read(&path)?)?;
        self.validate_binding(r, &b)?;
        Ok(Some(b))
    }
    async fn result(
        &self,
        graph: &Graph,
        client: &Client,
        r: &Request,
    ) -> Result<Option<(Artifact, VerifiedResult)>> {
        if !self.read_only {
            if let Some(binding) = self.binding(r)? {
                // Unadmitted/waiting-authority bindings cannot execute. Admission
                // occurs only in offer after the independent exact task grant.
                if !matches!(r.role, Role::Direct | Role::Continuation)
                    || crate::eval_registry::native_scq::policy::InstalledPolicy::load(
                        &self.config.fit_policy,
                    )?
                    .task_policy(&binding.admission.manifest.sha256)?
                    .is_some()
                {
                    graph.reconcile_eligible(client, &binding.admission).await?;
                }
            }
        }
        let key = identity(&r.run, r.generation, &r.id)?;
        let path = self
            .graph_state
            .join("tasks")
            .join(key)
            .join("verified.json");
        if !path.exists() {
            return Ok(None);
        };
        let binding = self
            .binding(r)?
            .context("original loop binding is required before consuming graph result")?;
        let a = artifact(&self.root, &path)?;
        let result = graph.verified(client, &a).await?;
        verify_result_binding(r, &binding.admission, &result)?;
        ensure!(
            result.run == r.run && result.generation == r.generation && result.task_id == r.id,
            "loop result identity changed"
        );
        Ok(Some((a, result)))
    }
    async fn failed(
        &self,
        graph: &Graph,
        client: &Client,
        request: &Request,
    ) -> Result<Option<Artifact>> {
        let Some(binding) = self.binding(request)? else {
            return Ok(None);
        };
        graph.failed(client, &binding.admission).await
    }
    async fn offer(
        &self,
        graph: &Graph,
        client: &Client,
        r: &Request,
        progress: &mut Progress,
    ) -> Result<()> {
        if self.binding(r)?.is_none() {
            self.materialize_request(graph, client, r).await?;
        }
        let Some(binding) = self.binding(r)? else {
            progress.waiting_materialization += 1;
            progress
                .idle_reasons
                .insert("awaiting_hash_bound_task_materialization".into());
            return Ok(());
        };
        if let Some(failure) = self.failed(graph, client, r).await? {
            progress.idle_reasons.insert(format!(
                "terminal_native_failure_requires_explicit_recovery:{}",
                r.id
            ));
            frozen(
                &self.root,
                &self
                    .state
                    .join("refusals")
                    .join(format!("{}.json", identity(&r.run, r.generation, &r.id)?)),
                &json!({"schema":"jope.native_task_refusal.v1","request":binding.request,"failure":failure,"numerical_completion_claimed":false}),
            )?;
            return Ok(());
        }
        if matches!(r.role, Role::Direct | Role::Continuation) {
            let key = identity(&r.run, r.generation, &r.id)?;
            let admission = frozen(
                &self.root,
                &self
                    .state
                    .join("authority-requests")
                    .join(format!("{key}.admission.json")),
                &binding.admission,
            )?;
            frozen(
                &self.root,
                &self
                    .state
                    .join("authority-requests")
                    .join(format!("{key}.json")),
                &json!({"schema":"jope.native_grant_request_outbox.v1","run":r.run,"generation":r.generation,"role":r.role,
                    "frozen_config":artifact(&self.root,&self.state.join("config.json"))?,"request":binding.request,
                    "admission":admission,"task_manifest":binding.admission.manifest,"authorization_issued":false}),
            )?;
            let policy = crate::eval_registry::native_scq::policy::InstalledPolicy::load(
                &self.config.fit_policy,
            )?;
            if policy
                .task_policy(&binding.admission.manifest.sha256)?
                .is_none()
            {
                progress.waiting_authority += 1;
                progress
                    .idle_reasons
                    .insert("awaiting_independent_exact_task_authorization".into());
                return Ok(());
            }
        }
        graph.admit(client, &binding.admission).await?;
        graph.reconcile_eligible(client, &binding.admission).await?;
        progress.offered += 1;
        Ok(())
    }
    fn freeze_genome(
        &self,
        generation: u64,
        candidate: &Artifact,
        direct: Option<Artifact>,
        predecessors: BTreeMap<String, Artifact>,
        vocabulary: Artifact,
    ) -> Result<Request> {
        let c: Candidate = native_genome::load(&self.root, candidate)?;
        let stage = if direct.is_some() {
            Stage::Continuation
        } else {
            Stage::Direct
        };
        let id = format!(
            "g{generation}-{}-{}",
            if stage == Stage::Direct {
                "direct"
            } else {
                "continuation"
            },
            candidate.sha256
        );
        let base = self.state.join("numerical").join(&id);
        let output = base.join("output").strip_prefix(&self.root)?.to_owned();
        let task = GenomeTask {
            schema: "jope.native_genome_task.v1".into(),
            stage,
            candidate: candidate.clone(),
            output,
            vocabulary,
            admission_policy: self.config.label_admission_policy.clone(),
            qualification: self.config.qualification,
            preparation_overlay: self.config.preparation_overlay.clone(),
            evaluations: if stage == Stage::Direct {
                1
            } else {
                self.config.continuation_evaluations
            },
            seconds: if stage == Stage::Direct {
                self.config.direct_seconds
            } else {
                self.config.continuation_seconds
            },
            rng_seed: generation_seed(
                &self.config.run,
                generation,
                &format!(
                    "{}:{}:{}",
                    c.context.prepared_sha256, c.context.cut, c.matched_slot
                ),
            )?,
            direct_parent: direct,
        };
        let reference = frozen(&self.root, &base.join("genome-task.json"), &task)?;
        let (request_ref, r) = self.request(
            generation,
            id.clone(),
            if stage == Stage::Direct {
                Role::Direct
            } else {
                Role::Continuation
            },
            json!({"genome_task":reference}),
            predecessors,
        )?;
        let mut dependencies = native_genome::required_inputs(&self.root, &reference)?;
        dependencies.insert(request_ref.path.clone(), request_ref.sha256.clone());
        let mut closure = Vec::new();
        for (path, sha256) in &dependencies {
            serde_json::to_writer(
                &mut closure,
                &Artifact {
                    path: path.clone(),
                    sha256: sha256.clone(),
                },
            )?;
            closure.push(b'\n');
        }
        let closure_ref = immutable(&self.root, &base.join("closure.jsonl"), &closure)?;
        let input = InputManifest {
            schema: "jope.smartcluster_input.v1".into(),
            root: self.root.clone(),
            operation: if stage == Stage::Direct {
                Operation::DirectEvaluation {
                    manifest: reference.path,
                    sha256: reference.sha256,
                }
            } else {
                Operation::SeededContinuation {
                    manifest: reference.path,
                    sha256: reference.sha256,
                }
            },
            input_artifacts: BTreeMap::new(),
            closure: Some(ClosureManifest {
                path: closure_ref.path.clone(),
                sha256: closure_ref.sha256.clone(),
            }),
            output_receipt: base
                .join("result.json")
                .strip_prefix(&self.root)?
                .to_owned(),
        };
        let input_ref = frozen(&self.root, &base.join("task.input.json"), &input)?;
        let manifest = TaskManifest {
            schema: "jope.smartcluster_task.v1".into(),
            run: self.config.run.clone(),
            generation,
            task_id: id.clone(),
            kind: if stage == Stage::Direct {
                TaskKind::GpSearch
            } else {
                TaskKind::Continuation
            },
            input_manifest_sha256: input_ref.sha256,
            dependencies: BTreeMap::from([(
                closure_ref.path.to_string_lossy().into_owned(),
                closure_ref.sha256,
            )]),
            parent_checkpoint_sha256: Some(c.parent_checkpoint.sha256.clone()),
            corpus_sha256: Some(
                native_genome::load::<Value>(&self.root, &c.parent_checkpoint)?["corpus_sha256"]
                    .as_str()
                    .context("candidate parent corpus identity absent")?
                    .to_owned(),
            ),
            idempotency_key: IdempotencyKey::new(format!("{}/{id}", self.config.run))?,
            deadline: self.config.task_deadline,
            retry_allowance: 1,
            resources: self.config.cpu_resources.clone(),
            expected_receipt_id: format!("{}/{id}/complete", self.config.run),
            checkpoint_allowance_ms: None,
        };
        let manifest_ref = frozen(&self.root, &base.join("task.json"), &manifest)?;
        self.bind(&Binding {
            schema: "jope.native_loop_binding.v1".into(),
            request: request_ref,
            admission: Admission {
                schema: "jope.native_graph_admission.v1".into(),
                manifest: manifest_ref,
                predecessors: r.predecessors.clone(),
                delivery: Delivery::NativeJob,
            },
        })?;
        Ok(r)
    }
    async fn aggregate(
        &self,
        graph: &Graph,
        client: &Client,
        generation: u64,
        mut members: BTreeMap<String, Artifact>,
        progress: &mut Progress,
    ) -> Result<Option<BTreeMap<String, Artifact>>> {
        while members.len() > 1024 {
            let rows: Vec<_> = members.into_iter().collect();
            let mut next = BTreeMap::new();
            let mut ready = true;
            for chunk in rows.chunks(1024) {
                let predecessors: BTreeMap<_, _> = chunk.iter().cloned().collect();
                let id = format!(
                    "g{generation}-aggregate-{}",
                    hash(&serde_json::to_vec(&predecessors)?)
                );
                let (request_ref, r) = self.request(
                    generation,
                    id.clone(),
                    Role::Aggregate,
                    json!({"members":predecessors}),
                    predecessors.clone(),
                )?;
                let base = self.state.join("aggregates").join(&id);
                let input = InputManifest {
                    schema: "jope.smartcluster_input.v1".into(),
                    root: self.root.clone(),
                    operation: Operation::Verify,
                    input_artifacts: predecessors
                        .values()
                        .map(|a| (a.path.clone(), a.sha256.clone()))
                        .chain(std::iter::once((
                            request_ref.path.clone(),
                            request_ref.sha256.clone(),
                        )))
                        .collect(),
                    closure: None,
                    output_receipt: base.join("result.json").strip_prefix(&self.root)?.into(),
                };
                let input_ref = frozen(&self.root, &base.join("task.input.json"), &input)?;
                let manifest = TaskManifest {
                    schema: "jope.smartcluster_task.v1".into(),
                    run: self.config.run.clone(),
                    generation,
                    task_id: id.clone(),
                    kind: TaskKind::Verification,
                    input_manifest_sha256: input_ref.sha256,
                    dependencies: input
                        .input_artifacts
                        .iter()
                        .map(|(p, h)| (p.to_string_lossy().into_owned(), h.clone()))
                        .collect(),
                    parent_checkpoint_sha256: None,
                    corpus_sha256: None,
                    idempotency_key: IdempotencyKey::new(format!("{}/{id}", self.config.run))?,
                    deadline: self.config.task_deadline,
                    retry_allowance: 1,
                    resources: self.config.cpu_resources.clone(),
                    expected_receipt_id: format!("{}/{id}/complete", self.config.run),
                    checkpoint_allowance_ms: None,
                };
                let manifest_ref = frozen(&self.root, &base.join("task.json"), &manifest)?;
                self.bind(&Binding {
                    schema: "jope.native_loop_binding.v1".into(),
                    request: request_ref,
                    admission: Admission {
                        schema: "jope.native_graph_admission.v1".into(),
                        manifest: manifest_ref,
                        predecessors,
                        delivery: Delivery::NativeJob,
                    },
                })?;
                if let Some((reference, result)) = self.result(graph, client, &r).await? {
                    next.insert(result.task_id, reference);
                } else {
                    self.offer(graph, client, &r, progress).await?;
                    ready = false;
                }
            }
            if !ready {
                progress
                    .idle_reasons
                    .insert("awaiting_complete_dependency_aggregation".into());
                return Ok(None);
            }
            members = next;
        }
        Ok(Some(members))
    }

    fn recorded_candidate(&self, path: &Path, bytes: &[u8]) -> Result<Artifact> {
        if self.read_only {
            let a = artifact(&self.root, path)?;
            ensure!(
                native_genome::checked(&self.root, &a, 1024 * 1024)? == bytes,
                "original candidate differs from deterministic proposal expansion"
            );
            Ok(a)
        } else {
            immutable(&self.root, path, bytes)
        }
    }
    /// Expand every valid recorded neural program, including those outside the ranked64 pool.
    /// Matched generic controls use the same cut/slot budget and independent deterministic RNG.
    fn candidates(
        &self,
        generation: u64,
        arm: Arm,
        proposal: &Artifact,
        result: &VerifiedResult,
        checkpoint: &Artifact,
        corpus: &Artifact,
    ) -> Result<Vec<Artifact>> {
        let value: Value = native_genome::load(&self.root, &result.output)?;
        let outputs: BTreeMap<PathBuf, String> =
            serde_json::from_value(value["artifacts"].clone())?;
        let binding_path = self.state.join("bindings").join(format!(
            "{}.json",
            identity(&self.config.run, generation, &result.task_id)?
        ));
        let binding: Binding = serde_json::from_slice(&read(&binding_path)?)?;
        let (_, input) = crate::smartcluster_worker::verify_task_inputs(
            &self.root,
            &self.root.join(&binding.admission.manifest.path),
            &binding.admission.manifest.sha256,
        )?;
        let Operation::Proposal { output, .. } = input.operation else {
            anyhow::bail!("candidate source is not a verified proposal")
        };
        let jobs_file = Artifact {
            path: output.join("jobs-0.json"),
            sha256: outputs
                .get(Path::new("jobs-0.json"))
                .context("proposal jobs absent")?
                .clone(),
        };
        let jobs: crate::service::GpJobs = native_genome::load(&self.root, &jobs_file)?;
        let corpus_root = corpus.path.parent().context("corpus directory")?;
        let index = crate::teaching_store::read_index(&self.root, &self.root.join(corpus_root))?;
        let vocab: crate::grammar::Vocab =
            crate::manifest::read_json(&self.root.join(corpus_root).join("vocabulary.json"))?;
        let mut result_refs = Vec::new();
        for job in jobs.jobs {
            let c = index
                .contexts
                .iter()
                .find(|c| c.prepared_sha256 == job.prepared_sha256 && c.cut == job.cut)
                .context("proposal context absent from parent corpus")?
                .clone();
            ensure!(
                c.partition == "meta_train" && !self.policy.excluded_families.contains(&c.family),
                "proposal selected excluded or sealed family"
            );
            let seed = job.seeds.context("proposal seed parent absent")?;
            let capture = seed
                .parent()
                .context("seed parent")?
                .join("all-valid-neural.json");
            let suffix = capture.strip_prefix(&output)?;
            let a = Artifact {
                path: capture.clone(),
                sha256: outputs
                    .get(suffix)
                    .context("all valid model proposals were not preserved")?
                    .clone(),
            };
            let all: Value = native_genome::load(&self.root, &a)?;
            ensure!(
                all["schema"] == "jope.all_valid_neural.v1" && all["final_tests_opened"] == false,
                "invalid all-valid capture"
            );
            let rows = all["rows"]
                .as_array()
                .context("valid candidate rows absent")?;
            ensure!(
                rows.len() <= 1024 && all["valid"] == rows.len(),
                "valid candidate coverage differs"
            );
            let mut unique = BTreeSet::new();
            let mut generic_unique = BTreeSet::new();
            let mut matched_slot = 0u64;
            for (origin_row, row) in rows.iter().enumerate() {
                let genome = row["genome"].as_str().context("candidate genome absent")?;
                ensure!(
                    row["program_sha256"] == hash(genome.as_bytes()),
                    "captured genotype hash differs"
                );
                if !unique.insert(genome.to_string()) {
                    continue;
                }
                let cand = Candidate {
                    schema: "jope.native_candidate.v1".into(),
                    run: self.config.run.clone(),
                    generation,
                    arm,
                    matched_slot,
                    context: c.clone(),
                    genome: genome.into(),
                    genome_sha256: hash(genome.as_bytes()),
                    proposal: proposal.clone(),
                    parent_checkpoint: checkpoint.clone(),
                    origin_capture: a.clone(),
                    origin_row,
                    generic_draw: None,
                    sampling_seed: all["sampling_seed"]
                        .as_u64()
                        .context("proposal RNG absent")?,
                };
                let bytes = serde_json::to_vec(&cand)?;
                result_refs.push(
                    self.recorded_candidate(
                        &self
                            .state
                            .join("candidates")
                            .join(format!("{}.json", hash(&bytes))),
                        &bytes,
                    )?,
                );
                if arm == Arm::Model {
                    let rng = generation_seed(
                        &self.config.run,
                        generation,
                        &format!("generic:{}:{}:{matched_slot}", c.prepared_sha256, c.cut),
                    )?;
                    let mut generated = None;
                    for offset in 0..10000u64 {
                        let raw = feat_math::evolve::random_canonical_pipeline(
                            rng.wrapping_add(offset),
                            6,
                        );
                        if let Ok(program) = vocab.project_proposal(&raw) {
                            if generic_unique.insert(program.clone()) {
                                generated = Some((program, offset));
                                break;
                            }
                        }
                    }
                    let (genome, draw) = generated.context("matched generic genome unavailable")?;
                    let generic = Candidate {
                        arm: Arm::Generic,
                        genome_sha256: hash(genome.as_bytes()),
                        genome,
                        sampling_seed: rng,
                        generic_draw: Some(draw),
                        ..cand
                    };
                    let bytes = serde_json::to_vec(&generic)?;
                    result_refs.push(
                        self.recorded_candidate(
                            &self
                                .state
                                .join("candidates")
                                .join(format!("{}.json", hash(&bytes))),
                            &bytes,
                        )?,
                    );
                }
                matched_slot += 1;
            }
        }
        ensure!(
            !result_refs.is_empty() && result_refs.len() <= 100000,
            "empty or unbounded candidate expansion"
        );
        Ok(result_refs)
    }
    async fn verify_completed_predecessor(
        &self,
        graph: &Graph,
        client: &Client,
        completed: &RunCompletion,
    ) -> Result<()> {
        let old_state = self.root.join(
            completed
                .config
                .path
                .parent()
                .context("old config parent")?,
        );
        let previous = Self::open_mode(
            &self.root,
            &old_state,
            &self.graph_state,
            &completed.config,
            true,
        )?;
        let mut parent = previous.config.accepted_checkpoint.clone();
        let mut corpus = previous.config.initial_corpus.clone();
        let mut acceptance = previous.config.accepted_decision.clone();
        for generation in 0..2 {
            let pack = previous
                .original_request(generation, &format!("g{generation}-pack-and-publish"))?;
            ensure!(
                pack.role == Role::PackAndPublish
                    && pack.parameters["previous_corpus"] == serde_json::to_value(&corpus)?,
                "predecessor corpus lineage changed"
            );
            let (pack_ref, packed) = previous
                .result(graph, client, &pack)
                .await?
                .context("predecessor publication absent")?;
            let published = native_genome::load_dataset_receipt(&self.root, &packed.output)?;
            ensure!(
                published["local_corpus_committed"] == true
                    && (!previous.policy.require_remote_publication
                        || published["remote_verified"] == true),
                "predecessor corpus publication not verified"
            );
            let next_corpus: Artifact = serde_json::from_value(published["corpus_index"].clone())?;
            let train = previous.original_request(generation, &format!("g{generation}-train"))?;
            let initial = previous.config.initial_optimizer_step + generation * 3000;
            ensure!(
                train.role == Role::Train
                    && train.parameters["checkpoint"] == serde_json::to_value(&parent)?
                    && train.parameters["corpus"] == serde_json::to_value(&next_corpus)?
                    && train.parameters["initial_step"] == initial
                    && train.parameters["requested_increment"] == 3000
                    && train.parameters["target_step"] == initial + 3000
                    && train.parameters["global_batch"] == 2048
                    && train.predecessors == BTreeMap::from([(packed.task_id, pack_ref)]),
                "predecessor original optimizer/publication edge differs"
            );
            let (train_ref, trained) = previous
                .result(graph, client, &train)
                .await?
                .context("predecessor Train result absent")?;
            let numerical: Value = native_genome::load(&self.root, &trained.output)?;
            let checkpoint_path = PathBuf::from(
                numerical["latest"]["checkpoint"]
                    .as_str()
                    .context("predecessor checkpoint absent")?,
            );
            let next_parent = artifact(
                &self.root,
                &self.root.join(checkpoint_path).join("state.json"),
            )?;
            let comparison = previous
                .prove_comparison(graph, client, generation, &parent, &corpus, &acceptance)
                .await?;
            let decision_ref: Artifact = serde_json::from_slice(&read(
                &old_state
                    .join("parent-decisions")
                    .join(format!("g{generation}.json")),
            )?)?;
            let decision: ProposalParentDecision = native_genome::load(&self.root, &decision_ref)?;
            let assessment: Value = native_genome::load(&self.root, &decision.assessment)?;
            verify_parent_decision(
                &decision,
                &assessment,
                &completed.run,
                generation,
                &next_parent,
                &parent,
                &train_ref,
                &comparison,
            )?;
            ensure!(decision.accepted, "predecessor checkpoint was not accepted");
            let state: Value = native_genome::load(&self.root, &next_parent)?;
            ensure!(
                state["step"] == initial + 3000 && state["corpus_sha256"] == next_corpus.sha256,
                "predecessor final optimizer/corpus bytes differ"
            );
            parent = next_parent;
            corpus = next_corpus;
            acceptance = decision_ref;
        }
        ensure!(
            parent == completed.checkpoint
                && corpus == completed.corpus
                && acceptance == completed.accepted_parent_decision,
            "successor completion does not derive from actual final Train/Pack"
        );
        for (arm, reference) in &completed.final_proposals {
            let id = format!(
                "g2-proposal-{}",
                if *arm == Arm::Model {
                    "model"
                } else {
                    "shuffled"
                }
            );
            let request = previous.original_request(2, &id)?;
            ensure!(
                request.parameters["checkpoint"] == serde_json::to_value(&parent)?
                    && request.parameters["corpus"] == serde_json::to_value(&corpus)?
                    && request.parameters["accepted_parent_decision"]
                        == serde_json::to_value(&acceptance)?
                    && request.parameters["proposal_seed"]
                        == generation_seed(&completed.run, 2, &parent.sha256)?,
                "final successor proposal parent changed"
            );
            let (actual, _) = previous
                .result(graph, client, &request)
                .await?
                .context("final predecessor proposal absent")?;
            ensure!(
                actual == *reference,
                "successor proposal differs from original bound result"
            );
        }
        Ok(())
    }
    pub async fn tick(&self, graph: &Graph, client: &Client) -> Result<Progress> {
        ensure!(!self.read_only, "read-only loop cannot dispatch work");
        self.config.validate()?;
        if let Some(a) = &self.config.preparation_overlay {
            crate::preparation_overlay::load(&self.root, a, &self.config.run)?
                .require_live(now()?, self.config.task_deadline.as_unix_millis())?;
        }
        let approval = self.verify_run_approval()?;
        graph.verify_transport(client, &approval)?;
        ensure!(
            self.config.task_deadline.as_unix_millis() > now()?,
            "original loop deadline elapsed"
        );
        if let Some(previous) = &self.config.predecessor_run {
            let completed: RunCompletion = native_genome::load(&self.root, previous)?;
            let previous_config: Config = native_genome::load(&self.root, &completed.config)?;
            verify_successor(&self.config, &previous_config, &completed)?;
            self.verify_completed_predecessor(graph, client, &completed)
                .await?;
        }
        let mut p = Progress::default();
        let mut checkpoint = self.config.accepted_checkpoint.clone();
        let mut corpus = self.config.initial_corpus.clone();
        let mut parent_predecessors = BTreeMap::new();
        let mut parent_acceptance = self.config.accepted_decision.clone();
        let mut all_targets = Vec::new();
        for generation in 0..=self.config.cycles {
            let mut proposals = Vec::new();
            for (arm, role) in [
                (Arm::Model, Role::ProposalModel),
                (Arm::Shuffled, Role::ProposalShuffled),
            ] {
                let id = format!(
                    "g{generation}-proposal-{}",
                    if arm == Arm::Model {
                        "model"
                    } else {
                        "shuffled"
                    }
                );
                let seed = generation_seed(&self.config.run, generation, &checkpoint.sha256)?;
                let (_,r)=self.request(generation,id,role,json!({"checkpoint":checkpoint,"corpus":corpus,"proposal_seed":seed,"scientific_policy":self.config.scientific_policy,"accepted_parent_decision":parent_acceptance}),parent_predecessors.clone())?;
                if let Some(result) = self.result(graph, client, &r).await? {
                    proposals.push((arm, result));
                } else {
                    self.offer(graph, client, &r, &mut p).await?;
                    p.waiting_execution += 1;
                }
            }
            if proposals.len() != 2 {
                break;
            }
            if generation == self.config.cycles {
                let completed = RunCompletion {
                    schema: "jope.native_loop_completion.v1".into(),
                    run: self.config.run.clone(),
                    config: artifact(&self.root, &self.state.join("config.json"))?,
                    initial_optimizer_step: self.config.initial_optimizer_step,
                    target_optimizer_step: self.config.initial_optimizer_step + 6000,
                    checkpoint: checkpoint.clone(),
                    corpus: corpus.clone(),
                    accepted_parent_decision: parent_acceptance.clone(),
                    final_proposals: proposals
                        .iter()
                        .map(|(arm, (a, _))| (*arm, a.clone()))
                        .collect(),
                };
                frozen(&self.root, &self.state.join("completion.json"), &completed)?;
                p.complete = true;
                break;
            }
            let mut pending_direct = Vec::new();
            let mut pending_continuation = Vec::new();
            let mut inflight_direct = 0usize;
            let mut inflight_continuation = 0usize;
            let mut targets = Vec::new();
            let mut target_predecessors = BTreeMap::new();
            let mut expected_targets = 0usize;
            let mut refused = BTreeMap::new();
            let mut comparison_candidates = Vec::new();
            let vocabulary = artifact(
                &self.root,
                &self
                    .root
                    .join(corpus.path.parent().context("corpus parent")?)
                    .join("vocabulary.json"),
            )?;
            for (arm, (proposal_ref, proposal_result)) in &proposals {
                for candidate in self.candidates(
                    generation,
                    *arm,
                    proposal_ref,
                    proposal_result,
                    &checkpoint,
                    &corpus,
                )? {
                    comparison_candidates.push(candidate.clone());
                    expected_targets += 1;
                    let predecessors =
                        BTreeMap::from([(proposal_result.task_id.clone(), proposal_ref.clone())]);
                    let direct = self.freeze_genome(
                        generation,
                        &candidate,
                        None,
                        predecessors,
                        vocabulary.clone(),
                    )?;
                    let Some((direct_ref, direct_result)) =
                        self.result(graph, client, &direct).await?
                    else {
                        if let Some(failure) = self.failed(graph, client, &direct).await? {
                            refused.insert(candidate.sha256.clone(), failure);
                            continue;
                        }
                        if self.submitted(&direct)? {
                            inflight_direct += 1
                        } else {
                            pending_direct.push(direct);
                        }
                        continue;
                    };
                    let continuation = self.freeze_genome(
                        generation,
                        &candidate,
                        Some(direct_result.output),
                        BTreeMap::from([(direct_result.task_id, direct_ref)]),
                        vocabulary.clone(),
                    )?;
                    if let Some((result_ref, result)) =
                        self.result(graph, client, &continuation).await?
                    {
                        let target = native_genome::target(&self.root, &result.output)?;
                        let a = frozen(
                            &self.root,
                            &self
                                .state
                                .join("targets")
                                .join(format!("{}.json", hash(&serde_json::to_vec(&target)?))),
                            &target,
                        )?;
                        targets.push(a);
                        target_predecessors.insert(result.task_id, result_ref);
                    } else if let Some(failure) = self.failed(graph, client, &continuation).await? {
                        refused.insert(candidate.sha256.clone(), failure);
                    } else if self.submitted(&continuation)? {
                        inflight_continuation += 1
                    } else {
                        pending_continuation.push(continuation);
                    }
                }
            }
            let comparisons = matched_comparisons(&self.root, generation, &comparison_candidates)?;
            frozen(
                &self.root,
                &self
                    .state
                    .join("comparisons")
                    .join(format!("g{generation}.json")),
                &comparisons,
            )?;
            p.pending_direct = pending_direct.len() + inflight_direct;
            p.pending_continuation = pending_continuation.len() + inflight_continuation;
            let selected = replenish(
                self.config.workers,
                inflight_direct,
                inflight_continuation,
                pending_direct,
                pending_continuation,
            );
            for request in selected {
                self.offer(graph, client, &request, &mut p).await?;
            }
            if targets.len() + refused.len() != expected_targets {
                p.idle_reasons
                    .insert("awaiting_original_direct_and_matched_continuation_outcomes".into());
                break;
            }
            let refusal_ref = frozen(
                &self.root,
                &self
                    .state
                    .join("refusals")
                    .join(format!("g{generation}.json")),
                &json!({"schema":"jope.native_generation_refusals.v1","run":self.config.run,"generation":generation,
                    "refusals":refused,"original_failure_receipts_retained":true}),
            )?;
            let comparison = frozen(
                &self.root,
                &self
                    .state
                    .join("comparisons")
                    .join(format!("g{generation}-outcomes.json")),
                &comparison_outcomes(
                    &self.root,
                    generation,
                    &targets,
                    &comparison_candidates,
                    &refusal_ref,
                )?,
            )?;
            all_targets.extend(targets);
            all_targets.sort_by(|a, b| a.sha256.cmp(&b.sha256));
            all_targets.dedup();
            let set = crate::continuation_corpus::TargetSet {
                preparation_overlay: self.config.preparation_overlay.clone(),
                schema: "jope.continuation_target_set.v1".into(),
                run: self.config.run.clone(),
                generation,
                source_graph_state: self.graph_state.strip_prefix(&self.root)?.into(),
                qualification: self.config.qualification,
                parent_corpus: corpus.clone(),
                vocabulary,
                admission_policy: self.config.label_admission_policy.clone(),
                frozen_scientific_policy: self.config.scientific_policy.clone(),
                targets: all_targets.clone(),
                training_arms: self.policy.training_arms.clone(),
            };
            let set_ref = frozen(
                &self.root,
                &self
                    .state
                    .join("corpora")
                    .join(format!("g{generation}-targets.json")),
                &set,
            )?;
            let mut source_closure =
                crate::continuation_corpus::input_closure(&self.root, &set_ref, &self.graph_state)?;
            source_closure.insert(refusal_ref.path.clone(), refusal_ref.sha256.clone());
            source_closure.insert(comparison.path.clone(), comparison.sha256.clone());
            for failure in refused.values() {
                source_closure.insert(failure.path.clone(), failure.sha256.clone());
            }
            let mut closure_bytes = Vec::new();
            for (path, sha256) in source_closure {
                serde_json::to_writer(&mut closure_bytes, &Artifact { path, sha256 })?;
                closure_bytes.push(b'\n');
            }
            let closure_ref = immutable(
                &self.root,
                &self
                    .state
                    .join("corpora")
                    .join(format!("g{generation}-source-closure.jsonl")),
                &closure_bytes,
            )?;
            let Some(target_predecessors) = self
                .aggregate(graph, client, generation, target_predecessors, &mut p)
                .await?
            else {
                break;
            };
            let (_, pack) = self.request(
                generation,
                format!("g{generation}-pack-and-publish"),
                Role::PackAndPublish,
                json!({"target_set":set_ref,"previous_corpus":corpus,"closure":closure_ref}),
                target_predecessors,
            )?;
            let Some((pack_ref, pack_result)) = self.result(graph, client, &pack).await? else {
                self.offer(graph, client, &pack, &mut p).await?;
                p.waiting_execution += 1;
                break;
            };
            let publication: Value =
                native_genome::load_dataset_receipt(&self.root, &pack_result.output)?;
            ensure!(
                publication["local_corpus_committed"] == true,
                "changed corpus not locally committed before training"
            );
            if self.policy.require_remote_publication {
                ensure!(
                    publication["remote_verified"] == true,
                    "changed corpus remote restoration not verified"
                );
            }
            let next_corpus: Artifact = serde_json::from_value(publication["corpus_index"].clone())
                .context("published corpus exact index receipt absent")?;
            let _: Value = native_genome::load_corpus_index(&self.root, &next_corpus)?;
            ensure!(
                next_corpus.sha256 != corpus.sha256,
                "training requires a changed committed corpus"
            );
            let predecessors = BTreeMap::from([(pack_result.task_id, pack_ref)]);
            let (_,train)=self.request(generation,format!("g{generation}-train"),Role::Train,json!({"checkpoint":checkpoint,"corpus":next_corpus,"initial_step":self.config.initial_optimizer_step+generation*3000,"requested_increment":3000,"target_step":self.config.initial_optimizer_step+(generation+1)*3000,"global_batch":2048}),predecessors)?;
            let Some((train_ref, train_result)) = self.result(graph, client, &train).await? else {
                self.offer(graph, client, &train, &mut p).await?;
                p.waiting_execution += 1;
                break;
            };
            let output: Value = native_genome::load(&self.root, &train_result.output)?;
            let cp = PathBuf::from(
                output["latest"]["checkpoint"]
                    .as_str()
                    .context("trained checkpoint path absent")?,
            );
            let candidate_checkpoint =
                artifact(&self.root, &self.root.join(cp).join("state.json"))?;
            let state: Value = native_genome::load(&self.root, &candidate_checkpoint)?;
            ensure!(
                state["step"] == self.config.initial_optimizer_step + (generation + 1) * 3000
                    && state["corpus_sha256"] == next_corpus.sha256,
                "optimizer window or changed corpus differs"
            );
            let decision_path = self
                .state
                .join("parent-decisions")
                .join(format!("g{generation}.json"));
            if !decision_path.exists() {
                p.idle_reasons
                    .insert("awaiting_independent_proposal_parent_quality_decision".into());
                break;
            }
            let decision_ref: Artifact = serde_json::from_slice(&read(&decision_path)?)?;
            let decision: ProposalParentDecision = native_genome::load(&self.root, &decision_ref)?;
            let assessment: Value = native_genome::load(&self.root, &decision.assessment)?;
            verify_parent_decision(
                &decision,
                &assessment,
                &self.config.run,
                generation,
                &candidate_checkpoint,
                &checkpoint,
                &train_ref,
                &comparison,
            )?;
            if !decision.accepted {
                p.idle_reasons
                    .insert("trained_checkpoint_rejected_as_proposal_parent".into());
                break;
            }
            checkpoint = candidate_checkpoint;
            parent_acceptance = decision_ref;
            corpus = next_corpus;
            parent_predecessors = BTreeMap::from([(train_result.task_id, train_ref)]);
            p.completed_transitions += 1;
        }
        let status = json!({"schema":"jope.native_loop_health.v1","timestamp_unix_ms":now()?,"stale_after_ms":30000,"progress":p,"model_improvement_claimed":false});
        crate::manifest::atomic_write(
            &self.state.join("health.json"),
            &serde_json::to_vec(&status)?,
        )?;
        Ok(p)
    }
    fn submitted(&self, r: &Request) -> Result<bool> {
        let key = identity(&r.run, r.generation, &r.id)?;
        Ok(self
            .graph_state
            .join("tasks")
            .join(key)
            .join("submitted.json")
            .exists())
    }
}
/// Pair only actual same-cut/slot observations; unmatched valid outputs remain explicit.
fn matched_comparisons(root: &Path, generation: u64, artifacts: &[Artifact]) -> Result<Value> {
    let mut slots = BTreeMap::<(String, u64), BTreeMap<Arm, Artifact>>::new();
    for a in artifacts {
        let c: Candidate = native_genome::load(root, a)?;
        ensure!(c.generation == generation, "comparison generation differs");
        let key = (hash(&serde_json::to_vec(&c.context)?), c.matched_slot);
        ensure!(
            slots
                .entry(key)
                .or_default()
                .insert(c.arm, a.clone())
                .is_none(),
            "duplicate comparison arm/slot"
        );
    }
    let mut paired = Vec::new();
    let mut unmatched = Vec::new();
    for ((context, slot), arms) in slots {
        if arms.len() == 3 {
            paired.push(json!({"context_sha256":context,"slot":slot,"arms":arms}));
        } else {
            unmatched.push(json!({"context_sha256":context,"slot":slot,"arms":arms,"reason":"unequal_valid_output_counts","excluded_from_matched_statistics":true}));
        }
    }
    Ok(
        json!({"schema":"jope.native_matched_comparisons.v1","generation":generation,"matching":"same_original_cut_slot_and_numerical_budget","paired":paired,"unmatched":unmatched,"all_valid_candidates_preserved":true}),
    )
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalParentDecision {
    pub schema: String,
    pub run: String,
    pub generation: u64,
    pub checkpoint: Artifact,
    pub previous_parent: Artifact,
    pub training_result: Artifact,
    pub comparison: Artifact,
    pub assessment: Artifact,
    pub accepted: bool,
}
fn verify_result_binding(
    request: &Request,
    admission: &Admission,
    result: &VerifiedResult,
) -> Result<()> {
    ensure!(
        admission.predecessors == request.predecessors
            && result.admission_sha256 == hash(&serde_json::to_vec(admission)?)
            && result.task_manifest_sha256 == admission.manifest.sha256
            && result.run == request.run
            && result.generation == request.generation
            && result.task_id == request.id,
        "graph result differs from exact original loop admission/predecessors"
    );
    Ok(())
}
fn verify_parent_decision(
    d: &ProposalParentDecision,
    a: &Value,
    run: &str,
    g: u64,
    cp: &Artifact,
    previous: &Artifact,
    train: &Artifact,
    comparison: &Artifact,
) -> Result<()> {
    ensure!(
        d.schema == "jope.proposal_parent_decision.v1"
            && d.run == run
            && d.generation == g
            && &d.checkpoint == cp
            && &d.previous_parent == previous
            && &d.training_result == train
            && &d.comparison == comparison,
        "quality decision does not bind this original committed checkpoint and evidence"
    );
    ensure!(
        a["schema"] == "jope.proposal_parent_quality_review.v1"
            && a["checkpoint"] == serde_json::to_value(cp)?
            && a["previous_parent"] == serde_json::to_value(previous)?
            && a["training_result"] == serde_json::to_value(train)?
            && a["comparison"] == serde_json::to_value(comparison)?
            && a["accepted"] == d.accepted
            && a["sealed_test_used"] == false
            && a["serving_promotion"] == false
            && a["reviewer"].as_str().is_some_and(|s| !s.trim().is_empty())
            && a["rationale"]
                .as_str()
                .is_some_and(|s| !s.trim().is_empty()),
        "independent quality review absent or mixed with serving/sealed-test evidence"
    );
    Ok(())
}
fn comparison_outcomes(
    root: &Path,
    g: u64,
    targets: &[Artifact],
    candidates: &[Artifact],
    refusals: &Artifact,
) -> Result<Value> {
    let mut outcomes = BTreeMap::new();
    for target in targets {
        let value: crate::native_genome::ContinuationTarget = native_genome::load(root, target)?;
        ensure!(
            outcomes
                .insert(
                    value.candidate.sha256.clone(),
                    (target.clone(), value.clone())
                )
                .is_none(),
            "duplicate candidate outcome"
        );
    }
    let mut report = matched_comparisons(root, g, candidates)?;
    let mut comparisons = serde_json::Map::new();
    for opponent in ["generic", "shuffled"] {
        let (mut wins, mut losses, mut ties, mut unavailable) = (0u64, 0u64, 0u64, 0u64);
        for pair in report["paired"]
            .as_array()
            .context("paired comparison set")?
        {
            let model: Artifact = serde_json::from_value(pair["arms"]["model"].clone())?;
            let other: Artifact = serde_json::from_value(pair["arms"][opponent].clone())?;
            let left = outcomes
                .get(&model.sha256)
                .filter(|(_, target)| target.training_admission == "admitted")
                .and_then(|(_, target)| target.quality_rmse_cut_normalized);
            let right = outcomes
                .get(&other.sha256)
                .filter(|(_, target)| target.training_admission == "admitted")
                .and_then(|(_, target)| target.quality_rmse_cut_normalized);
            match (left, right) {
                (Some(a), Some(b)) if a < b => wins += 1,
                (Some(a), Some(b)) if a > b => losses += 1,
                (Some(_), Some(_)) => ties += 1,
                _ => unavailable += 1,
            }
        }
        let n = wins + losses;
        let interval = if n == 0 {
            Value::Null
        } else {
            let nf = n as f64;
            let p = wins as f64 / nf;
            let z = 1.959963984540054;
            let den = 1.0 + z * z / nf;
            let center = (p + z * z / (2.0 * nf)) / den;
            let radius = z * ((p * (1.0 - p) / nf + z * z / (4.0 * nf * nf)).sqrt()) / den;
            json!([center - radius, center + radius])
        };
        comparisons.insert(opponent.into(),json!({"wins":wins,"losses":losses,"ties":ties,"unavailable":unavailable,
            "decisive_pairs":n,"win_rate":if n==0 {Value::Null}else{json!(wins as f64/n as f64)},
            "nominal_unclustered_wilson_95_interval":interval,"independent_sampling_units_validated":false,"ties_excluded_from_decisive_rate":true,"independent_final_test":false}));
    }
    report["refusals"] = serde_json::to_value(refusals)?;
    report["outcomes"] = serde_json::to_value(outcomes)?;
    report["model_comparisons"] = Value::Object(comparisons);
    report["operational_progress_is_not_model_improvement"] = json!(true);
    Ok(report)
}

/// Eligibility replenishment, not resource scheduling. Native queued and running work both count.
/// Failures remain counted/visible until original recovery is resolved; no competing pool is started.
pub fn replenish(
    workers: usize,
    inflight_direct: usize,
    inflight_continuation: usize,
    mut direct: Vec<Request>,
    mut continuation: Vec<Request>,
) -> Vec<Request> {
    if inflight_direct.saturating_add(inflight_continuation) >= workers.saturating_mul(2) {
        return vec![];
    }
    direct.sort_by(|a, b| (a.generation, &a.id).cmp(&(b.generation, &b.id)));
    continuation.sort_by(|a, b| (a.generation, &a.id).cmp(&(b.generation, &b.id)));
    let slots = workers
        .saturating_mul(3)
        .saturating_sub(inflight_direct + inflight_continuation);
    let cap = if !direct.is_empty() && !continuation.is_empty() {
        (workers / 4).saturating_sub(inflight_continuation)
    } else {
        slots
    };
    let mut chosen: Vec<Request> = continuation.into_iter().take(cap.min(slots)).collect();
    chosen.extend(direct.into_iter().take(slots - chosen.len()));
    chosen
}
#[cfg(test)]
mod tests {
    use super::*;
    fn r(g: u64, n: &str, role: Role) -> Request {
        Request {
            schema: "jope.native_loop_request.v1".into(),
            run: "r".into(),
            generation: g,
            id: n.into(),
            role,
            parameters: json!({}),
            predecessors: BTreeMap::new(),
        }
    }
    #[test]
    fn independently_valid_graph_admission_with_same_ids_cannot_replace_loop_binding() {
        let request = r(0, "same", Role::Aggregate);
        let a = Artifact {
            path: "task.json".into(),
            sha256: "1".repeat(64),
        };
        let admission = Admission {
            schema: "jope.native_graph_admission.v1".into(),
            manifest: a.clone(),
            predecessors: BTreeMap::new(),
            delivery: Delivery::NativeJob,
        };
        let attempt: jain_smartcluster::FencedAttempt = serde_json::from_value(
            json!({"job_id":jain_smartcluster::JobId::new(),"number":1,"fence":1}),
        )
        .unwrap();
        let mut result = VerifiedResult {
            schema: "jope.native_graph_result.v1".into(),
            admission_sha256: hash(&serde_json::to_vec(&admission).unwrap()),
            task_manifest_sha256: a.sha256.clone(),
            task_id: request.id.clone(),
            run: request.run.clone(),
            generation: 0,
            receipt_id: "original".into(),
            native_attempt: attempt,
            worker_completion: a.clone(),
            output: a.clone(),
        };
        verify_result_binding(&request, &admission, &result).unwrap();
        let mut different = admission.clone();
        different.manifest.sha256 = "2".repeat(64);
        // This digest correctly authenticates a different real graph admission,
        // but names/run/generation alone must not make it the loop's task.
        result.admission_sha256 = hash(&serde_json::to_vec(&different).unwrap());
        result.task_manifest_sha256 = different.manifest.sha256;
        assert!(verify_result_binding(&request, &admission, &result).is_err());
    }
    #[test]
    fn committed_training_is_not_a_quality_or_serving_acceptance() {
        let a = Artifact {
            path: "checkpoint.json".into(),
            sha256: "1".repeat(64),
        };
        let mut decision = ProposalParentDecision {
            schema: "jope.proposal_parent_decision.v1".into(),
            run: "r".into(),
            generation: 0,
            checkpoint: a.clone(),
            previous_parent: a.clone(),
            training_result: a.clone(),
            comparison: a.clone(),
            assessment: a.clone(),
            accepted: true,
        };
        assert!(verify_parent_decision(
            &decision,
            &json!({"committed":true}),
            "r",
            0,
            &a,
            &a,
            &a,
            &a
        )
        .is_err());
        let review = json!({"schema":"jope.proposal_parent_quality_review.v1","checkpoint":a,"previous_parent":a,"training_result":a,"comparison":a,
            "accepted":true,"sealed_test_used":false,"serving_promotion":false,"reviewer":"unit-test reviewer","rationale":"unit-test evidence decision"});
        verify_parent_decision(&decision, &review, "r", 0, &a, &a, &a, &a).unwrap();
        decision.checkpoint.sha256 = "2".repeat(64);
        assert!(verify_parent_decision(&decision, &review, "r", 0, &a, &a, &a, &a).is_err());
    }
    #[test]
    fn unequal_valid_arms_remain_preserved_but_do_not_invent_matches() {
        let t = tempfile::tempdir().unwrap();
        let mut refs = Vec::new();
        let reference = Artifact {
            path: "source.json".into(),
            sha256: "a".repeat(64),
        };
        for (i, arm, slot) in [
            (0, Arm::Model, 0),
            (1, Arm::Generic, 0),
            (2, Arm::Shuffled, 0),
            (3, Arm::Model, 1),
            (4, Arm::Generic, 1),
        ] {
            let c = Candidate {
                schema: "jope.native_candidate.v1".into(),
                run: "r".into(),
                generation: 0,
                arm,
                matched_slot: slot,
                context: crate::corpus::ContextRow {
                    dataset: "d".into(),
                    family: "f".into(),
                    cut: 0,
                    partition: "meta_train".into(),
                    path: "context".into(),
                    sha256: "b".repeat(64),
                    prepared: "prepared".into(),
                    prepared_sha256: "c".repeat(64),
                },
                genome: "(pipe X (ridge))".into(),
                genome_sha256: "d".repeat(64),
                proposal: reference.clone(),
                parent_checkpoint: reference.clone(),
                sampling_seed: 1,
                origin_capture: reference.clone(),
                origin_row: 0,
                generic_draw: None,
            };
            refs.push(
                frozen(t.path(), &t.path().join(format!("candidates/{i}.json")), &c).unwrap(),
            );
        }
        let compared = matched_comparisons(t.path(), 0, &refs).unwrap();
        assert_eq!(compared["paired"].as_array().unwrap().len(), 1);
        assert_eq!(compared["unmatched"].as_array().unwrap().len(), 1);
        assert_eq!(
            compared["unmatched"][0]["arms"].as_object().unwrap().len(),
            2
        );
        refs.push(refs[0].clone());
        assert!(matched_comparisons(t.path(), 0, &refs).is_err());
    }
    #[test]
    fn shared_backlog_counts_and_continuation_quarter_cap() {
        let direct = (0..500)
            .map(|i| r(2, &i.to_string(), Role::Direct))
            .collect();
        let cont = (0..500)
            .map(|i| r(1, &i.to_string(), Role::Continuation))
            .collect();
        let out = replenish(48, 20, 4, direct, cont);
        assert_eq!(out.len(), 120);
        assert_eq!(
            out.iter().filter(|r| r.role == Role::Continuation).count(),
            8
        );
        assert!(replenish(48, 90, 6, vec![r(0, "x", Role::Direct)], vec![]).is_empty());
    }
    #[test]
    fn generations_not_newest_first_and_no_continuation_starvation() {
        let out = replenish(
            48,
            0,
            0,
            vec![r(3, "new", Role::Direct), r(0, "old", Role::Direct)],
            vec![r(1, "continuation", Role::Continuation)],
        );
        assert_eq!(out[0].role, Role::Continuation);
        assert_eq!(out[1].generation, 0);
        assert_eq!(out[2].generation, 3);
    }
    #[test]
    fn generation_seed_replays_but_changes_on_parent_or_generation() {
        let s = generation_seed("r", 0, "a").unwrap();
        assert_eq!(s, generation_seed("r", 0, "a").unwrap());
        assert_ne!(s, generation_seed("r", 1, "a").unwrap());
        assert_ne!(s, generation_seed("r", 0, "b").unwrap());
    }
    #[test]
    fn conflicting_durable_intent_cannot_extend_original_window() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("state/window.json");
        frozen(t.path(), &p, &json!({"initial":60000,"target":63000})).unwrap();
        assert!(frozen(t.path(), &p, &json!({"initial":60000,"target":66000})).is_err());
    }
    #[test]
    fn successor_requires_exact_completed_window_and_new_run_identity() {
        let a = Artifact {
            path: "a.json".into(),
            sha256: "a".repeat(64),
        };
        let previous = Config {
            preparation_overlay: None,
            schema: "jope.native_loop.v1".into(),
            run: "original".into(),
            accepted_checkpoint: a.clone(),
            initial_corpus: a.clone(),
            accepted_decision: a.clone(),
            scientific_policy: a.clone(),
            label_admission_policy: a.clone(),
            fit_policy: "policy.json".into(),
            run_approval: "/etc/jope-dime/test-run-approval.json".into(),
            resident_job: JobId::new(),
            workers: 48,
            cycles: 2,
            initial_optimizer_step: 60000,
            predecessor_run: None,
            updates_per_block: 3000,
            global_batch: 2048,
            microbatch: 256,
            direct_seconds: 30,
            continuation_evaluations: 16,
            continuation_seconds: 60,
            cpu_resources: ResourceRequest::cpu_and_ram(1000, 1024 * 1024 * 1024),
            task_deadline: Timestamp::from_unix_millis(u64::MAX),
            qualification: false,
            cuts: 1,
            cut_start: 0,
            materialization: None,
        };
        previous.validate().unwrap();
        let mut scoped = previous.clone();
        scoped.preparation_overlay = Some(a.clone());
        assert!(scoped.validate().is_err());
        scoped.continuation_evaluations = 24;
        scoped.continuation_seconds = 54;
        scoped.validate().unwrap();
        scoped.continuation_seconds = 53;
        assert!(scoped.validate().is_err());
        scoped.continuation_seconds = 54;
        scoped.continuation_evaluations = 23;
        assert!(scoped.validate().is_err());
        let completed = RunCompletion {
            schema: "jope.native_loop_completion.v1".into(),
            run: previous.run.clone(),
            config: a.clone(),
            initial_optimizer_step: 60000,
            target_optimizer_step: 66000,
            checkpoint: a.clone(),
            corpus: a.clone(),
            accepted_parent_decision: a.clone(),
            final_proposals: BTreeMap::from([(Arm::Model, a.clone()), (Arm::Shuffled, a.clone())]),
        };
        let mut next = previous.clone();
        next.run = "successor".into();
        next.initial_optimizer_step = 66000;
        next.predecessor_run = Some(a);
        next.validate().unwrap();
        verify_successor(&next, &previous, &completed).unwrap();
        next.initial_optimizer_step = 69000;
        assert!(next.validate().is_err());
        assert!(verify_successor(&next, &previous, &completed).is_err());
        next.initial_optimizer_step = 66000;
        next.run = previous.run.clone();
        assert!(verify_successor(&next, &previous, &completed).is_err());
        next.run = "successor".into();
        next.predecessor_run = None;
        assert!(next.validate().is_err());
        assert_eq!(previous.initial_optimizer_step, 60000);
    }
}
