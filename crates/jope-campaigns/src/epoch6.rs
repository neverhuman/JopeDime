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
mod pipeline_intents;
pub use pipeline_intents::{
    ConsumerHeartbeatV1, JopePipelineStatusV2, PipelineEdgeV1, PipelineStageStateV2,
    SupplyIntentStateV2, SupplyIntentV2, WorkIntentV1,
};
