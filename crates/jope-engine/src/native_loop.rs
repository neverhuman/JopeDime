//! Finite scientific eligibility controller. SmartCluster alone allocates and executes jobs.
//! Frozen requests are reviewable artifacts. External signed authority grants remain mandatory.
pub mod bounded;
pub mod cli;
mod grant_proof;
pub mod lineage;
pub(crate) mod materialize;
mod run_approval;
pub mod warm_start;
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
    os::unix::fs::OpenOptionsExt,
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
    ensure!(
        next.schema == warm_start::LOOP_V1,
        "successor version requires explicit lineage evidence"
    );
    ensure!(
        previous.schema == warm_start::LOOP_V1,
        "a {} weight-warm-start run has no successor protocol",
        warm_start::LOOP_V2
    );
    ensure!(completion.schema == "jope.native_loop_completion.v1"
        && completion.run == previous.run && next.run != previous.run
        && completion.initial_optimizer_step == previous.initial_optimizer_step
        && completion.target_optimizer_step == previous.initial_optimizer_step + 6000
        && next.initial_optimizer_step == completion.target_optimizer_step
        && next.accepted_checkpoint == completion.checkpoint && next.initial_corpus == completion.corpus
        && completion.final_proposals.keys().copied().collect::<BTreeSet<_>>() == BTreeSet::from([Arm::Model, Arm::Shuffled]),
        "successor must preserve a different completed run and its exact accepted optimizer/corpus window");
    verify_scientific_successor(next, previous)
}
fn verify_scientific_successor(next: &Config, previous: &Config) -> Result<()> {
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

/// Native entrypoint shared with the controller. A v3 successor never obtains
/// continuity from its numeric step alone. I13 supplies committed receipt bindings.
pub fn verify_versioned_successor(
    next: &Config,
    previous: &Config,
    completion: &RunCompletion,
    bundle: Option<&lineage::LineageBundle>,
    prior_lineage: Option<&lineage::VerifiedLineage>,
) -> Result<Option<lineage::VerifiedLineage>> {
    next.validate()?;
    if next.schema != warm_start::LOOP_V3 {
        ensure!(
            bundle.is_none() && prior_lineage.is_none(),
            "legacy successor cannot carry lineage reinterpretation"
        );
        verify_successor(next, previous, completion)?;
        return Ok(None);
    }
    previous.validate()?;
    let expected_schema = match previous.schema.as_str() {
        warm_start::LOOP_V2 => warm_start::COMPLETION_V2,
        warm_start::LOOP_V3 => warm_start::COMPLETION_V3,
        _ => anyhow::bail!("v3 requires a completed reset-epoch descendant"),
    };
    ensure!(
        completion.schema == expected_schema
            && completion.run == previous.run
            && next.run != previous.run
            && completion.initial_optimizer_step == previous.train_window(0)?.initial
            && completion.target_optimizer_step == previous.train_window(1)?.target
            && next.initial_optimizer_step == completion.target_optimizer_step
            && next.accepted_checkpoint == completion.checkpoint
            && next.initial_corpus == completion.corpus
            && completion
                .final_proposals
                .keys()
                .copied()
                .collect::<BTreeSet<_>>()
                == BTreeSet::from([Arm::Model, Arm::Shuffled]),
        "v3 successor requires a separate exact two-block completion"
    );
    verify_scientific_successor(next, previous)?;
    let verified = bundle
        .context("v3 committed lineage evidence absent")?
        .verify()?;
    let history = verified.lineage();
    let prefix = if previous.schema == warm_start::LOOP_V2 {
        ensure!(
            prior_lineage.is_none()
                && history.anchor.checkpoint_sha256 == previous.accepted_checkpoint.sha256
                && history.anchor.optimizer_step == previous.initial_optimizer_step,
            "v3 reset anchor differs from accepted parent"
        );
        0
    } else {
        let verified_prior = prior_lineage.context("previous v3 lineage identity absent")?;
        ensure!(
            verified_prior.tip().checkpoint_sha256 == previous.accepted_checkpoint.sha256
                && verified_prior.local_step() == previous.initial_optimizer_step,
            "previous v3 lineage does not bind accepted checkpoint/epoch"
        );
        let prior = verified_prior.lineage();
        ensure!(
            history.anchor == prior.anchor
                && history.protocol == prior.protocol
                && history.edges.starts_with(&prior.edges),
            "v3 successor changed immutable ancestry"
        );
        prior.edges.len()
    };
    let suffix = verified
        .edges()
        .get(prefix..)
        .context("missing successor ancestry")?;
    ensure!(
        suffix.len() == 2
            && suffix[0].parent_checkpoint_sha256 == previous.accepted_checkpoint.sha256
            && suffix[0].intent.initial_step == completion.initial_optimizer_step
            && suffix[0].intent.target_step == suffix[1].intent.initial_step
            && suffix
                .iter()
                .all(|e| e.intent.target_step.checked_sub(e.intent.initial_step)
                    == Some(warm_start::BLOCK_UPDATES))
            && verified.local_step() == completion.target_optimizer_step
            && verified.tip().checkpoint_sha256 == completion.checkpoint.sha256
            && verified.tip().intent.corpus_sha256 == completion.corpus.sha256,
        "v3 completion lacks both committed block edges"
    );
    Ok(Some(verified))
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
pub enum NativeLoopRole {
    ProposalModel,
    ProposalShuffled,
    Direct,
    Continuation,
    PackAndPublish,
    Aggregate,
    Train,
}

/// Compatibility alias for the domain-specific `NativeLoopRole` type.
pub type Role = NativeLoopRole;
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema: String,
    pub run: String,
    pub generation: u64,
    pub id: String,
    pub role: NativeLoopRole,
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
    /// The optimizer window of `generation`'s training block (`warm_start::window`).
    pub(crate) fn train_window(&self, generation: u64) -> Result<warm_start::Window> {
        warm_start::window(&self.schema, self.initial_optimizer_step, generation)
    }
    fn validate(&self) -> Result<()> {
        // v1: the original accepted step-60000 parent or an explicit completed predecessor.
        // v2 (a weight warm start, OWNER-DECISIONS.md D12): the accepted step-60000 parent only.
        let window = match self.schema.as_str() {
            warm_start::LOOP_V1 => {
                (self.predecessor_run.is_none() && self.initial_optimizer_step == 60000)
                    || (self.predecessor_run.is_some()
                        && self.initial_optimizer_step > 60000
                        && (self.initial_optimizer_step - 60000).is_multiple_of(6000))
            }
            warm_start::LOOP_V2 => {
                self.predecessor_run.is_none() && self.initial_optimizer_step == 60000
            }
            warm_start::LOOP_V3 => {
                self.predecessor_run.is_some() && self.initial_optimizer_step > 0
            }
            _ => false,
        };
        ensure!(
            window
                && !self.run.is_empty()
                && self.cycles == 2
                && self.initial_optimizer_step.checked_add(6000).is_some()
                && self.updates_per_block == 3000
                && self.global_batch == 2048,
            "native loop requires original60000 or an explicit completed predecessor and two3000-update blocks"
        );
        ensure!(
            [48, 72, 96].contains(&self.workers)
                && [64, 128, 256].contains(&self.microbatch)
                && self.global_batch.is_multiple_of(self.microbatch)
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
mod candidates;
mod decisions;
mod offers;
mod open;
mod tick;
use decisions::{
    comparison_outcomes, matched_comparisons, verify_parent_decision, verify_result_binding,
    ParentEvidence,
};
pub use decisions::{replenish, ProposalParentDecision};

#[cfg(test)]
mod tests;
