use super::*;

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
