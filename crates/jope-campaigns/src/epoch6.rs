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

mod activation;
mod artifacts;
mod cohort;
mod replay;

pub use activation::{
    LaneActivationSetV1, LaneMaturity, LaneRole, TaskActivationV1, TaskLaneProfileV1,
};
pub use artifacts::{
    ArtifactCas, ArtifactGcRootClass, ArtifactGcRootV1, ArtifactGcRootsV1, ArtifactKind,
    ArtifactRefV1, EvaluatorRefV2,
};
pub use cohort::{CohortPlanV2, CohortSignalSnapshotV1, WaveCommitV2};
pub use replay::{
    CandidateSlotArithmeticV2, DlSlotPolicyV2, OperatorModulusV2, ReplayProtocolV2, RngAlgorithmV2,
};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
mod pipeline_intents;
pub use pipeline_intents::{
    ConsumerHeartbeatV1, JopePipelineStatusV2, PipelineEdgeV1, PipelineStageStateV2,
    SupplyIntentStateV2, SupplyIntentV2, WorkIntentV1,
};
