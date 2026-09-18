//! Epoch 6 operational contracts shared by recovery, Encoder V2, and Search V3.
//!
//! These are new-schema authorities. Frozen V1/V2 decoders remain in their
//! original modules and may read old work, but they cannot construct these
//! identities without satisfying the validations here.

use crate::paths;
use crate::selfplay_data::canonical_json_sha256;
use crate::selfplay_shadow::Sha256Digest;
use crate::task::{verify_embedded_task_semantics, Task, TASK_SEMANTICS_SHA256};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

pub const REPLAY_PROTOCOL_SCHEMA: &str = "ReplayProtocol.v2";
pub const LANE_ACTIVATION_SCHEMA: &str = "LaneActivationSet.v1";
pub const TASK_LANE_PROFILE_SCHEMA: &str = "TaskLaneProfile.v1";
pub const COHORT_SIGNAL_SNAPSHOT_SCHEMA: &str = "CohortSignalSnapshot.v1";
pub const COHORT_PLAN_SCHEMA: &str = "CohortPlan.v2";
pub const WAVE_COMMIT_SCHEMA: &str = "WaveCommit.v2";
pub const ARTIFACT_REF_SCHEMA: &str = "ArtifactRef.v1";
pub const ARTIFACT_MANIFEST_SCHEMA: &str = "JopeArtifactObjectManifest.v1";
pub const EVALUATOR_REF_SCHEMA: &str = "EvaluatorRef.v2";
pub const ARTIFACT_GC_ROOTS_SCHEMA: &str = "JopeArtifactGcRoots.v1";
pub const SUPPLY_INTENT_SCHEMA: &str = "SupplyIntent.v2";
pub const PIPELINE_EDGE_SCHEMA: &str = "PipelineEdge.v1";
pub const WORK_INTENT_SCHEMA: &str = "WorkIntent.v1";
pub const CONSUMER_HEARTBEAT_SCHEMA: &str = "ConsumerHeartbeat.v1";
pub const PIPELINE_STATUS_SCHEMA: &str = "JopePipelineStatus.v2";
jope_paths::once_str!(
    pub fn general_artifact_cas_root_str => paths::artifact_cas_root()
);

fn sha256(value: &str, field: &str) -> Result<()> {
    Sha256Digest::parse(value.to_owned())
        .with_context(|| format!("{field} is not a lowercase SHA-256"))?;
    Ok(())
}

fn sorted_unique<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn canonical_task_order(tasks: &[Task]) -> bool {
    let positions = tasks
        .iter()
        .map(|task| {
            Task::CANONICAL_ORDER
                .iter()
                .position(|candidate| candidate == task)
                .expect("Task::CANONICAL_ORDER is exhaustive")
        })
        .collect::<Vec<_>>();
    sorted_unique(&positions)
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneMaturity {
    Bootstrap,
    Mature,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneRole {
    StableAnchor,
    LeastObserved,
    HardUncertain,
    GrammarImmigrant,
    GpSeed,
    DlProposal,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskLaneProfileV1 {
    pub schema: String,
    pub task: Task,
    pub maturity: LaneMaturity,
    pub roles: Vec<LaneRole>,
}

impl TaskLaneProfileV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == TASK_LANE_PROFILE_SCHEMA,
            "invalid task-lane profile schema"
        );
        ensure!(
            !self.roles.is_empty() && sorted_unique(&self.roles),
            "lane roles are empty, duplicated, or unsorted"
        );
        match self.maturity {
            LaneMaturity::Bootstrap => ensure!(
                self.roles.contains(&LaneRole::GrammarImmigrant),
                "bootstrap lane lacks grammar immigrants"
            ),
            LaneMaturity::Mature => ensure!(
                self.roles.contains(&LaneRole::StableAnchor)
                    && self.roles.contains(&LaneRole::LeastObserved),
                "mature lane lacks stable and least-observed roles"
            ),
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskActivationV1 {
    pub task: Task,
    pub activation_epoch: String,
    pub activation_wave: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaneActivationSetV1 {
    pub schema: String,
    pub activations: Vec<TaskActivationV1>,
}

impl LaneActivationSetV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == LANE_ACTIVATION_SCHEMA,
            "invalid lane-activation schema"
        );
        let tasks = self
            .activations
            .iter()
            .map(|row| row.task)
            .collect::<Vec<_>>();
        ensure!(
            !tasks.is_empty() && canonical_task_order(&tasks),
            "activations are empty, duplicated, or noncanonical"
        );
        for activation in &self.activations {
            ensure!(
                !activation.activation_epoch.trim().is_empty(),
                "activation epoch is empty"
            );
        }
        Ok(())
    }

    pub fn is_active(&self, task: Task, epoch: &str, wave: u64) -> bool {
        self.activations.iter().any(|activation| {
            activation.task == task
                && activation.activation_epoch == epoch
                && wave >= activation.activation_wave
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateSlotArithmeticV2 {
    pub slots_per_dataset: u32,
    pub common_winner_slots: Vec<u32>,
    pub checkpoint_proposal_slots: Vec<u32>,
}

impl CandidateSlotArithmeticV2 {
    fn validate(&self) -> Result<()> {
        ensure!(self.slots_per_dataset > 0, "candidate slot count is zero");
        let validate = |slots: &[u32], name: &str| -> Result<()> {
            ensure!(
                sorted_unique(slots),
                "{name} slots are duplicated or unsorted"
            );
            ensure!(
                slots.iter().all(|slot| *slot < self.slots_per_dataset),
                "{name} slot is out of range"
            );
            Ok(())
        };
        validate(&self.common_winner_slots, "common winner")?;
        validate(&self.checkpoint_proposal_slots, "checkpoint proposal")?;
        let common = self
            .common_winner_slots
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        ensure!(
            self.checkpoint_proposal_slots
                .iter()
                .all(|slot| !common.contains(slot)),
            "candidate slot groups overlap"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorModulusV2 {
    pub operator: String,
    pub modulus: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DlSlotPolicyV2 {
    pub maximum_slots: u32,
    pub replacement_only: bool,
    pub candidate_evaluation_calls: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RngAlgorithmV2 {
    Sha256Counter,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayProtocolV2 {
    pub schema: String,
    pub task_order: Vec<Task>,
    pub task_lane_sizes: BTreeMap<Task, u32>,
    pub selector_versions: BTreeMap<Task, String>,
    pub frozen_lane_profiles: BTreeMap<Task, TaskLaneProfileV1>,
    pub candidate_slots: CandidateSlotArithmeticV2,
    pub gp_seed_slots: Vec<u32>,
    pub operator_vocabulary: Vec<OperatorModulusV2>,
    pub dl_slot_policy: DlSlotPolicyV2,
    pub proposal_sampling_policy: String,
    pub rng_algorithm: RngAlgorithmV2,
    pub evaluator_protocol_sha256: String,
    pub task_semantics_sha256: String,
    pub encoding_protocol_sha256: String,
}

impl ReplayProtocolV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == REPLAY_PROTOCOL_SCHEMA,
            "invalid replay protocol schema"
        );
        ensure!(
            self.task_order == Task::CANONICAL_ORDER,
            "replay task order is not canonical"
        );
        let expected = Task::CANONICAL_ORDER.into_iter().collect::<BTreeSet<_>>();
        for (actual, field) in [
            (
                self.task_lane_sizes
                    .keys()
                    .copied()
                    .collect::<BTreeSet<_>>(),
                "lane sizes",
            ),
            (
                self.selector_versions
                    .keys()
                    .copied()
                    .collect::<BTreeSet<_>>(),
                "selector versions",
            ),
            (
                self.frozen_lane_profiles
                    .keys()
                    .copied()
                    .collect::<BTreeSet<_>>(),
                "lane profiles",
            ),
        ] {
            ensure!(actual == expected, "replay {field} is not exhaustive");
        }
        ensure!(
            self.task_lane_sizes.values().all(|size| *size > 0),
            "replay lane size is zero"
        );
        ensure!(
            self.selector_versions
                .values()
                .all(|version| !version.trim().is_empty()),
            "selector version is empty"
        );
        for (task, profile) in &self.frozen_lane_profiles {
            profile.validate()?;
            ensure!(profile.task == *task, "lane profile map key mismatch");
        }
        self.candidate_slots.validate()?;
        ensure!(
            sorted_unique(&self.gp_seed_slots),
            "GP seed slots are duplicated or unsorted"
        );
        ensure!(
            self.gp_seed_slots
                .iter()
                .all(|slot| *slot < self.candidate_slots.slots_per_dataset),
            "GP seed slot is out of range"
        );
        ensure!(
            !self.operator_vocabulary.is_empty(),
            "operator vocabulary is empty"
        );
        let operators = self
            .operator_vocabulary
            .iter()
            .map(|row| row.operator.as_str())
            .collect::<Vec<_>>();
        ensure!(
            sorted_unique(&operators),
            "operator vocabulary is duplicated or unsorted"
        );
        ensure!(
            self.operator_vocabulary.iter().all(|row| row.modulus > 0),
            "operator modulus is zero"
        );
        ensure!(
            self.dl_slot_policy.maximum_slots <= self.candidate_slots.slots_per_dataset,
            "DL slot limit exceeds the candidate lane"
        );
        ensure!(
            self.dl_slot_policy.replacement_only
                && self.dl_slot_policy.candidate_evaluation_calls == 0,
            "DL policy permits additional scientific work"
        );
        ensure!(
            !self.proposal_sampling_policy.trim().is_empty(),
            "proposal sampling policy is empty"
        );
        sha256(&self.evaluator_protocol_sha256, "evaluator protocol")?;
        sha256(&self.encoding_protocol_sha256, "encoding protocol")?;
        verify_embedded_task_semantics()?;
        ensure!(
            self.task_semantics_sha256 == TASK_SEMANTICS_SHA256,
            "replay task-semantics digest drift"
        );
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn derive_item_seed(
        &self,
        wave: u64,
        task: Task,
        dataset_sha256: &str,
        slot_group: &str,
        slot: u32,
        operator: &str,
        local_counter: u64,
    ) -> Result<Sha256Digest> {
        let protocol_sha256 = self.digest()?;
        sha256(dataset_sha256, "dataset")?;
        ensure!(
            self.task_order.contains(&task),
            "seed task is absent from replay protocol"
        );
        ensure!(
            slot < self.candidate_slots.slots_per_dataset,
            "seed slot is out of range"
        );
        ensure!(
            self.operator_vocabulary
                .iter()
                .any(|row| row.operator == operator),
            "seed operator is absent from replay protocol"
        );
        for (field, name) in [(slot_group, "slot group"), (operator, "operator")] {
            ensure!(!field.trim().is_empty(), "seed {name} is empty");
        }
        let value = serde_json::json!({
            "domain": "jope-replay-protocol-v2-item-seed",
            "protocol_sha256": protocol_sha256,
            "wave": wave,
            "task": task,
            "dataset_sha256": dataset_sha256,
            "slot_group": slot_group,
            "slot": slot,
            "operator": operator,
            "local_counter": local_counter,
        });
        Sha256Digest::parse(canonical_json_sha256(&value)?)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CohortSignalSnapshotV1 {
    pub schema: String,
    pub catalog_sha256: String,
    pub observations_sha256: String,
    pub stale_wave_input_sha256: String,
    pub hardness_input_sha256: String,
}

impl CohortSignalSnapshotV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == COHORT_SIGNAL_SNAPSHOT_SCHEMA,
            "invalid cohort-signal schema"
        );
        for (value, field) in [
            (&self.catalog_sha256, "catalog"),
            (&self.observations_sha256, "observations"),
            (&self.stale_wave_input_sha256, "stale-wave input"),
            (&self.hardness_input_sha256, "hardness input"),
        ] {
            sha256(value, field)?;
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CohortPlanV2 {
    pub schema: String,
    pub activation_sha256: String,
    pub replay_protocol_sha256: String,
    pub catalog_sha256: String,
    pub observation_sha256: String,
    pub signal_snapshot_sha256: String,
}

impl CohortPlanV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == COHORT_PLAN_SCHEMA,
            "invalid cohort-plan schema"
        );
        for (value, field) in [
            (&self.activation_sha256, "activation"),
            (&self.replay_protocol_sha256, "replay protocol"),
            (&self.catalog_sha256, "catalog"),
            (&self.observation_sha256, "observation"),
            (&self.signal_snapshot_sha256, "signal snapshot"),
        ] {
            sha256(value, field)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaveCommitV2 {
    pub schema: String,
    pub epoch: String,
    pub wave: u64,
    pub cohort_plan_sha256: String,
    pub task_counts: BTreeMap<Task, u32>,
    pub manifest_count: u32,
    pub manifest_set_sha256: String,
}

impl WaveCommitV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == WAVE_COMMIT_SCHEMA,
            "invalid wave-commit schema"
        );
        ensure!(!self.epoch.trim().is_empty(), "wave epoch is empty");
        sha256(&self.cohort_plan_sha256, "cohort plan")?;
        sha256(&self.manifest_set_sha256, "manifest set")?;
        ensure!(!self.task_counts.is_empty(), "wave has no task counts");
        let actual = self.task_counts.values().try_fold(0_u32, |sum, value| {
            sum.checked_add(*value).context("wave task-count overflow")
        })?;
        ensure!(
            actual == self.manifest_count,
            "wave task counts do not equal actual manifest count"
        );
        ensure!(
            matches!(actual, 128 | 192 | 256),
            "wave manifest count is not an admitted shape"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Evaluator,
    Checkpoint,
    Panel,
    FrozenCatalog,
    RuntimeHelper,
    ModelBundle,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRefV1 {
    pub schema: String,
    pub kind: ArtifactKind,
    pub object_sha256: String,
    pub object_manifest_sha256: String,
    pub protocol_sha256: String,
    pub byte_length: u64,
}

impl ArtifactRefV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == ARTIFACT_REF_SCHEMA,
            "invalid artifact-ref schema"
        );
        sha256(&self.object_sha256, "artifact object")?;
        sha256(&self.object_manifest_sha256, "artifact object manifest")?;
        sha256(&self.protocol_sha256, "artifact protocol")?;
        ensure!(self.byte_length > 0, "artifact is empty");
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ArtifactObjectManifestV1 {
    schema: String,
    kind: ArtifactKind,
    object_sha256: String,
    protocol_sha256: String,
    byte_length: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorRefV2 {
    pub schema: String,
    pub artifact: ArtifactRefV1,
    pub build_identity_sha256: String,
    pub evaluator_protocol_sha256: String,
}

impl EvaluatorRefV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == EVALUATOR_REF_SCHEMA,
            "invalid evaluator-ref schema"
        );
        self.artifact.validate()?;
        ensure!(
            self.artifact.kind == ArtifactKind::Evaluator,
            "evaluator ref names a non-evaluator artifact"
        );
        sha256(&self.build_identity_sha256, "evaluator build identity")?;
        sha256(&self.evaluator_protocol_sha256, "evaluator protocol")?;
        ensure!(
            self.artifact.protocol_sha256 == self.evaluator_protocol_sha256,
            "evaluator protocol/ref mismatch"
        );
        Ok(())
    }
}

pub struct ArtifactCas {
    root: PathBuf,
}

impl ArtifactCas {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn publish_bytes(
        &self,
        kind: ArtifactKind,
        protocol_sha256: &str,
        bytes: &[u8],
    ) -> Result<ArtifactRefV1> {
        sha256(protocol_sha256, "artifact protocol")?;
        ensure!(!bytes.is_empty(), "refusing to publish an empty artifact");
        let object_sha256 = hex::encode(Sha256::digest(bytes));
        let manifest = ArtifactObjectManifestV1 {
            schema: ARTIFACT_MANIFEST_SCHEMA.to_owned(),
            kind,
            object_sha256: object_sha256.clone(),
            protocol_sha256: protocol_sha256.to_owned(),
            byte_length: bytes.len() as u64,
        };
        let manifest_bytes = serde_json::to_vec(&manifest)?;
        let manifest_sha256 = hex::encode(Sha256::digest(&manifest_bytes));
        publish_cas_file(&self.object_path(&object_sha256), bytes)?;
        publish_cas_file(&self.manifest_path(&manifest_sha256), &manifest_bytes)?;
        let reference = ArtifactRefV1 {
            schema: ARTIFACT_REF_SCHEMA.to_owned(),
            kind,
            object_sha256,
            object_manifest_sha256: manifest_sha256,
            protocol_sha256: protocol_sha256.to_owned(),
            byte_length: bytes.len() as u64,
        };
        self.verify(&reference)?;
        Ok(reference)
    }

    pub fn publish_file(
        &self,
        kind: ArtifactKind,
        protocol_sha256: &str,
        path: &Path,
    ) -> Result<ArtifactRefV1> {
        ensure!(path.is_file(), "artifact source is not a regular file");
        self.publish_bytes(kind, protocol_sha256, &fs::read(path)?)
    }

    pub fn verify(&self, reference: &ArtifactRefV1) -> Result<()> {
        reference.validate()?;
        let object = fs::read(self.object_path(&reference.object_sha256))
            .context("read artifact CAS object")?;
        ensure!(
            object.len() as u64 == reference.byte_length,
            "artifact CAS object length drift"
        );
        ensure!(
            hex::encode(Sha256::digest(&object)) == reference.object_sha256,
            "artifact CAS object digest drift"
        );
        let manifest_bytes = fs::read(self.manifest_path(&reference.object_manifest_sha256))
            .context("read artifact CAS manifest")?;
        ensure!(
            hex::encode(Sha256::digest(&manifest_bytes)) == reference.object_manifest_sha256,
            "artifact CAS manifest digest drift"
        );
        let manifest: ArtifactObjectManifestV1 = serde_json::from_slice(&manifest_bytes)?;
        ensure!(
            manifest.schema == ARTIFACT_MANIFEST_SCHEMA,
            "artifact CAS manifest schema drift"
        );
        ensure!(
            manifest.kind == reference.kind
                && manifest.object_sha256 == reference.object_sha256
                && manifest.protocol_sha256 == reference.protocol_sha256
                && manifest.byte_length == reference.byte_length,
            "artifact CAS manifest/reference mismatch"
        );
        Ok(())
    }

    pub fn materialize(&self, reference: &ArtifactRefV1, destination: &Path) -> Result<()> {
        self.verify(reference)?;
        validate_materialization_path(destination)?;
        ensure!(
            !destination.exists(),
            "artifact materialization destination exists"
        );
        let parent = destination
            .parent()
            .context("artifact destination has no parent")?;
        fs::create_dir_all(parent)?;
        let mut input = File::open(self.object_path(&reference.object_sha256))?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o444)
            .open(destination)?;
        std::io::copy(&mut input, &mut output)?;
        output.sync_all()?;
        fs::set_permissions(destination, fs::Permissions::from_mode(0o444))?;
        sync_directory(parent)
    }

    fn object_path(&self, digest: &str) -> PathBuf {
        self.root
            .join("objects/sha256")
            .join(&digest[..2])
            .join(digest)
    }

    fn manifest_path(&self, digest: &str) -> PathBuf {
        self.root
            .join("manifests/sha256")
            .join(&digest[..2])
            .join(format!("{digest}.json"))
    }
}

fn validate_materialization_path(path: &Path) -> Result<()> {
    ensure!(
        path.is_absolute(),
        "artifact materialization destination is not absolute"
    );
    ensure!(
        path.components()
            .all(|part| !matches!(part, Component::ParentDir | Component::CurDir)),
        "artifact materialization destination is not normalized"
    );
    Ok(())
}

fn publish_cas_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("CAS path has no parent")?;
    fs::create_dir_all(parent)?;
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o444)
        .open(path)
    {
        Ok(mut file) => {
            file.write_all(bytes)?;
            file.sync_all()?;
            fs::set_permissions(path, fs::Permissions::from_mode(0o444))?;
            sync_directory(parent)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let mut existing = Vec::new();
            File::open(path)?.read_to_end(&mut existing)?;
            ensure!(existing == bytes, "content-addressed artifact conflict");
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactGcRootClass {
    Queues,
    Observations,
    Repairs,
    Canaries,
    Promotions,
    BenchmarkPlans,
    LiftManifests,
    Quarantine,
    RollbackEvidence,
    ModelBundles,
}

impl ArtifactGcRootClass {
    pub const ALL: [Self; 10] = [
        Self::Queues,
        Self::Observations,
        Self::Repairs,
        Self::Canaries,
        Self::Promotions,
        Self::BenchmarkPlans,
        Self::LiftManifests,
        Self::Quarantine,
        Self::RollbackEvidence,
        Self::ModelBundles,
    ];
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactGcRootV1 {
    pub class: ArtifactGcRootClass,
    pub readable: bool,
    pub artifacts: Vec<ArtifactRefV1>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactGcRootsV1 {
    pub schema: String,
    pub roots: Vec<ArtifactGcRootV1>,
}

impl ArtifactGcRootsV1 {
    pub fn validate_for_sweep(&self) -> Result<BTreeSet<String>> {
        ensure!(
            self.schema == ARTIFACT_GC_ROOTS_SCHEMA,
            "invalid artifact GC-root schema"
        );
        ensure!(
            self.roots.len() == ArtifactGcRootClass::ALL.len(),
            "artifact GC root classes are incomplete"
        );
        let classes = self.roots.iter().map(|root| root.class).collect::<Vec<_>>();
        ensure!(
            classes == ArtifactGcRootClass::ALL,
            "artifact GC roots are absent, duplicated, or noncanonical"
        );
        let mut reachable = BTreeSet::new();
        for root in &self.roots {
            ensure!(
                root.readable,
                "artifact GC root {:?} is unreadable",
                root.class
            );
            for artifact in &root.artifacts {
                artifact.validate()?;
                reachable.insert(artifact.object_sha256.clone());
            }
        }
        Ok(reachable)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SupplyIntentStateV2 {
    Planned,
    Proposing,
    Built,
    Published,
    KeeperAcked,
    Superseded,
    FailedRetryable,
    FailedPermanent,
}

impl SupplyIntentStateV2 {
    pub fn can_transition_to(self, next: Self) -> bool {
        match self {
            Self::Planned => matches!(
                next,
                Self::Proposing | Self::Superseded | Self::FailedPermanent
            ),
            Self::Proposing => matches!(
                next,
                Self::Built | Self::FailedRetryable | Self::FailedPermanent | Self::Superseded
            ),
            Self::Built => matches!(
                next,
                Self::Published | Self::FailedRetryable | Self::FailedPermanent | Self::Superseded
            ),
            Self::Published => matches!(
                next,
                Self::KeeperAcked | Self::FailedRetryable | Self::FailedPermanent
            ),
            Self::FailedRetryable => matches!(
                next,
                Self::Proposing
                    | Self::Built
                    | Self::Published
                    | Self::FailedPermanent
                    | Self::Superseded
            ),
            Self::KeeperAcked | Self::Superseded | Self::FailedPermanent => false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SupplyIntentV2 {
    pub schema: String,
    pub checkpoint_sha256: String,
    pub cohort_sha256: String,
    pub catalog_sha256: String,
    pub replay_protocol_sha256: String,
    pub task_activation_sha256: String,
    pub proposal_identity_sha256: String,
    pub seed_namespace_sha256: String,
    pub expected_manifest_count: u32,
    pub admission_budget: u32,
    pub state: SupplyIntentStateV2,
}

impl SupplyIntentV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SUPPLY_INTENT_SCHEMA,
            "invalid supply-intent schema"
        );
        for (value, field) in [
            (&self.checkpoint_sha256, "checkpoint"),
            (&self.cohort_sha256, "cohort"),
            (&self.catalog_sha256, "catalog"),
            (&self.replay_protocol_sha256, "replay protocol"),
            (&self.task_activation_sha256, "task activation"),
            (&self.proposal_identity_sha256, "proposal identity"),
            (&self.seed_namespace_sha256, "seed namespace"),
        ] {
            sha256(value, field)?;
        }
        ensure!(
            matches!(self.expected_manifest_count, 128 | 192 | 256),
            "supply intent has an unsupported manifest count"
        );
        ensure!(
            self.admission_budget >= self.expected_manifest_count,
            "supply admission budget is below its manifested work"
        );
        Ok(())
    }

    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::json!({
            "domain": "jope-supply-intent-v2",
            "checkpoint_sha256": self.checkpoint_sha256,
            "cohort_sha256": self.cohort_sha256,
            "catalog_sha256": self.catalog_sha256,
            "replay_protocol_sha256": self.replay_protocol_sha256,
            "task_activation_sha256": self.task_activation_sha256,
            "proposal_identity_sha256": self.proposal_identity_sha256,
            "seed_namespace_sha256": self.seed_namespace_sha256,
            "expected_manifest_count": self.expected_manifest_count,
            "admission_budget": self.admission_budget,
        }))
    }

    pub fn transition(&mut self, next: SupplyIntentStateV2) -> Result<()> {
        ensure!(
            self.state.can_transition_to(next),
            "invalid supply-intent transition {:?} -> {:?}",
            self.state,
            next
        );
        self.state = next;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineEdgeV1 {
    pub schema: String,
    pub edge_id: String,
    pub producer: String,
    pub consumer: String,
    pub accepted_schemas: Vec<String>,
    pub claim_ttl_seconds: u64,
    pub runnable_slo_seconds: u64,
    pub progress_slo_seconds: u64,
    pub terminal_result_schemas: Vec<String>,
}

impl PipelineEdgeV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == PIPELINE_EDGE_SCHEMA,
            "invalid pipeline-edge schema"
        );
        for (value, field) in [
            (&self.edge_id, "edge ID"),
            (&self.producer, "producer"),
            (&self.consumer, "consumer"),
        ] {
            ensure!(!value.trim().is_empty(), "pipeline {field} is empty");
        }
        ensure!(
            !self.accepted_schemas.is_empty() && sorted_unique(&self.accepted_schemas),
            "accepted schemas are empty, duplicated, or unsorted"
        );
        ensure!(
            !self.terminal_result_schemas.is_empty()
                && sorted_unique(&self.terminal_result_schemas),
            "terminal result schemas are empty, duplicated, or unsorted"
        );
        ensure!(
            self.claim_ttl_seconds > 0
                && self.runnable_slo_seconds > 0
                && self.progress_slo_seconds > 0,
            "pipeline edge has a zero SLO"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkIntentV1 {
    pub schema: String,
    pub intent_sha256: String,
    pub edge_id: String,
    pub payload_schema: String,
    pub payload_sha256: String,
    pub created_at_unix_ms: u64,
}

impl WorkIntentV1 {
    pub fn validate_for(&self, edge: &PipelineEdgeV1) -> Result<()> {
        ensure!(
            self.schema == WORK_INTENT_SCHEMA,
            "invalid work-intent schema"
        );
        sha256(&self.intent_sha256, "work intent")?;
        sha256(&self.payload_sha256, "work payload")?;
        ensure!(
            self.created_at_unix_ms > 0,
            "work intent creation time is absent"
        );
        ensure!(
            self.edge_id == edge.edge_id && edge.accepted_schemas.contains(&self.payload_schema),
            "work intent is incompatible with its edge"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum PipelineStageStateV2 {
    Running,
    Idle,
    Waiting {
        dependency: String,
        deadline_unix_ms: u64,
    },
    Paused {
        policy: String,
        unblock_condition: String,
    },
    BlockedCapacity,
    NoCompatibleConsumer,
    ConsumerUnavailable,
    ClaimStalled,
    FailedRetryable,
    FailedPermanent,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConsumerHeartbeatV1 {
    pub schema: String,
    pub edge_id: String,
    pub consumer: String,
    pub state: PipelineStageStateV2,
    pub process_heartbeat_unix_ms: u64,
    pub last_state_transition_unix_ms: u64,
    pub last_semantic_progress_unix_ms: u64,
    pub oldest_runnable_dependency_unix_ms: Option<u64>,
    /// Age of the oldest unresolved dependency at the process heartbeat.
    /// This is explicit so consumers never mistake a heartbeat for progress.
    pub oldest_unresolved_dependency_age_ms: Option<u64>,
    pub claimed_intent_sha256: Option<String>,
    pub claim_expires_unix_ms: Option<u64>,
}

impl ConsumerHeartbeatV1 {
    pub fn validate_at(&self, edge: &PipelineEdgeV1, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema == CONSUMER_HEARTBEAT_SCHEMA,
            "invalid consumer-heartbeat schema"
        );
        ensure!(
            self.edge_id == edge.edge_id && self.consumer == edge.consumer,
            "consumer heartbeat/edge mismatch"
        );
        ensure!(
            self.process_heartbeat_unix_ms <= now_unix_ms
                && self.last_state_transition_unix_ms <= now_unix_ms
                && self.last_semantic_progress_unix_ms <= now_unix_ms,
            "consumer heartbeat contains a future clock"
        );
        ensure!(
            self.last_state_transition_unix_ms <= self.process_heartbeat_unix_ms
                && self.last_semantic_progress_unix_ms <= self.process_heartbeat_unix_ms,
            "state transition or semantic progress is newer than process heartbeat"
        );
        match (
            self.oldest_runnable_dependency_unix_ms,
            self.oldest_unresolved_dependency_age_ms,
        ) {
            (Some(oldest), Some(age)) => ensure!(
                oldest <= self.process_heartbeat_unix_ms
                    && age == self.process_heartbeat_unix_ms.saturating_sub(oldest),
                "oldest unresolved dependency age is inconsistent"
            ),
            (None, None) => {}
            _ => bail!("oldest unresolved dependency timestamp and age are not paired"),
        }
        match (&self.claimed_intent_sha256, self.claim_expires_unix_ms) {
            (Some(intent), Some(expires)) => {
                sha256(intent, "claimed work intent")?;
                if matches!(self.state, PipelineStageStateV2::Running) {
                    ensure!(
                        expires >= now_unix_ms,
                        "running consumer has an expired claim"
                    );
                    ensure!(
                        now_unix_ms.saturating_sub(self.last_semantic_progress_unix_ms)
                            <= edge.progress_slo_seconds.saturating_mul(1_000),
                        "running consumer has stale semantic progress"
                    );
                }
            }
            (None, None) => ensure!(
                !matches!(
                    self.state,
                    PipelineStageStateV2::Running | PipelineStageStateV2::ClaimStalled
                ),
                "running or stalled consumer lacks a claim"
            ),
            _ => bail!("consumer claim identity and expiry are not paired"),
        }
        if let PipelineStageStateV2::Waiting {
            dependency,
            deadline_unix_ms,
        } = &self.state
        {
            ensure!(
                !dependency.trim().is_empty() && *deadline_unix_ms >= now_unix_ms,
                "waiting state lacks a live dependency deadline"
            );
        }
        if let PipelineStageStateV2::Paused {
            policy,
            unblock_condition,
        } = &self.state
        {
            ensure!(
                !policy.trim().is_empty() && !unblock_condition.trim().is_empty(),
                "paused state lacks policy or unblock condition"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JopePipelineStatusV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub valid_until_unix_ms: u64,
    pub protected_complete_wave_count: u64,
    pub manifest_count: u64,
    pub candidate_evaluation_count: u64,
    pub predicted_cpu_seconds: u64,
    pub edges: Vec<PipelineEdgeV1>,
    pub consumers: Vec<ConsumerHeartbeatV1>,
}

impl JopePipelineStatusV2 {
    pub fn validate_at(&self, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema == PIPELINE_STATUS_SCHEMA,
            "invalid pipeline-status schema"
        );
        ensure!(
            self.generated_at_unix_ms > 0 && self.generated_at_unix_ms <= now_unix_ms,
            "pipeline status timestamp is absent or future-dated"
        );
        ensure!(
            self.valid_until_unix_ms > self.generated_at_unix_ms
                && now_unix_ms <= self.valid_until_unix_ms,
            "pipeline status is stale or has an invalid freshness window"
        );
        ensure!(
            (self.manifest_count == 0
                && self.candidate_evaluation_count == 0
                && self.predicted_cpu_seconds == 0
                && self.protected_complete_wave_count == 0)
                || (self.manifest_count > 0
                    && self.candidate_evaluation_count >= self.manifest_count
                    && self.predicted_cpu_seconds > 0),
            "pipeline work accounting is inconsistent"
        );
        ensure!(!self.edges.is_empty(), "pipeline has no edges");
        let mut edge_ids = BTreeSet::new();
        for edge in &self.edges {
            edge.validate()?;
            ensure!(
                edge_ids.insert(edge.edge_id.as_str()),
                "pipeline repeats an edge ID"
            );
            let matches = self
                .consumers
                .iter()
                .filter(|heartbeat| heartbeat.edge_id == edge.edge_id)
                .collect::<Vec<_>>();
            ensure!(
                matches.len() == 1,
                "pipeline edge does not have exactly one consumer heartbeat"
            );
            ensure!(
                matches[0].process_heartbeat_unix_ms <= self.generated_at_unix_ms,
                "consumer heartbeat is newer than its pipeline status"
            );
            matches[0].validate_at(edge, now_unix_ms)?;
        }
        ensure!(
            self.consumers.len() == self.edges.len(),
            "pipeline has orphan consumer heartbeats"
        );
        Ok(())
    }

    /// Preflight a producer-to-consumer route before production work exists.
    pub fn preflight_route(
        &self,
        producer: &str,
        consumer: &str,
        now_unix_ms: u64,
    ) -> Result<Vec<String>> {
        self.validate_at(now_unix_ms)?;
        let mut queue = VecDeque::from([(producer.to_owned(), Vec::<String>::new())]);
        let mut visited = BTreeSet::new();
        while let Some((stage, path)) = queue.pop_front() {
            if stage == consumer {
                return Ok(path);
            }
            if !visited.insert(stage.clone()) {
                continue;
            }
            for edge in self.edges.iter().filter(|edge| edge.producer == stage) {
                let heartbeat = self
                    .consumers
                    .iter()
                    .find(|row| row.edge_id == edge.edge_id)
                    .expect("validated one heartbeat per edge");
                ensure!(
                    !matches!(
                        heartbeat.state,
                        PipelineStageStateV2::NoCompatibleConsumer
                            | PipelineStageStateV2::ConsumerUnavailable
                            | PipelineStageStateV2::FailedPermanent
                    ),
                    "pipeline route is unavailable at {}",
                    edge.edge_id
                );
                let mut next_path = path.clone();
                next_path.push(edge.edge_id.clone());
                queue.push_back((edge.consumer.clone(), next_path));
            }
        }
        bail!("no complete pipeline route from {producer} to {consumer}")
    }
}
