//! Fixed-schedule curriculum snapshots with explicit exposure ancestry.
//!
//! This records a bounded training candidate only. Synthetic outcomes never
//! become real-family qualification or policy-promotion evidence.

use sha2::{Digest, Sha256};
pub const CURRICULUM_SCHEMA: &str = "jope.selfplay-curriculum-snapshot/v1";
pub const MAX_SCHEDULE_TASKS: usize = 256;
pub const MAX_EXPOSURES: usize = 16_384;
pub const MAX_EXPOSURES_PER_TASK: usize = 4_096;
pub const MAX_ID_BYTES: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlantedTask {
    pub task_id: String,
    pub family_id: String,
    pub synthetic: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CurriculumExposure {
    pub exposure_id: String,
    pub task_id: String,
    pub family_id: String,
    pub schedule_position: usize,
    pub parent_exposure_id: Option<String>,
    pub synthetic: bool,
    pub success: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualificationDisposition {
    NotEvaluated,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CurriculumSnapshot {
    pub schema: &'static str,
    pub snapshot_id: String,
    pub schedule_id: String,
    pub policy_generation: String,
    pub replay_limit: usize,
    pub qualification: QualificationDisposition,
    pub schedule: Vec<PlantedTask>,
    pub exposures: Vec<CurriculumExposure>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CurriculumError {
    MissingIdentity,
    InvalidSchedule,
    InvalidReplayLimit,
    ExposureLimit,
    WrongScheduledTask,
    WrongParent,
    DuplicateExposure,
    TaskReplayLimit,
}

impl CurriculumSnapshot {
    pub fn new(
        schedule_id: impl Into<String>,
        policy_generation: impl Into<String>,
        schedule: Vec<PlantedTask>,
        replay_limit: usize,
    ) -> Result<Self, CurriculumError> {
        let schedule_id = schedule_id.into();
        let policy_generation = policy_generation.into();
        if schedule_id.trim().is_empty()
            || schedule_id.len() > MAX_ID_BYTES
            || policy_generation.trim().is_empty()
            || policy_generation.len() > MAX_ID_BYTES
        {
            return Err(CurriculumError::MissingIdentity);
        }
        if schedule.is_empty()
            || schedule.len() > MAX_SCHEDULE_TASKS
            || schedule.iter().any(|task| {
                task.task_id.trim().is_empty()
                    || task.task_id.len() > MAX_ID_BYTES
                    || task.family_id.trim().is_empty()
                    || task.family_id.len() > MAX_ID_BYTES
            })
        {
            return Err(CurriculumError::InvalidSchedule);
        }
        if replay_limit == 0 || replay_limit > MAX_EXPOSURES {
            return Err(CurriculumError::InvalidReplayLimit);
        }
        let snapshot_id =
            snapshot_digest(&schedule_id, &policy_generation, replay_limit, &schedule);
        Ok(Self {
            schema: CURRICULUM_SCHEMA,
            snapshot_id,
            schedule_id,
            policy_generation,
            replay_limit,
            qualification: QualificationDisposition::NotEvaluated,
            schedule,
            exposures: Vec::new(),
        })
    }

    /// Append one observed exposure only at the next fixed schedule position.
    /// The parent must identify the previous actual exposure, preserving the
    /// chain instead of accepting a caller-declared ancestry summary.
    pub fn record_exposure(
        &mut self,
        exposure_id: impl Into<String>,
        task_id: &str,
        parent_exposure_id: Option<&str>,
        success: bool,
    ) -> Result<(), CurriculumError> {
        if self.exposures.len() >= self.replay_limit || self.exposures.len() >= MAX_EXPOSURES {
            return Err(CurriculumError::ExposureLimit);
        }
        let position = self.exposures.len() % self.schedule.len();
        let task = &self.schedule[position];
        if task.task_id != task_id {
            return Err(CurriculumError::WrongScheduledTask);
        }
        let expected_parent = self.exposures.last().map(|item| item.exposure_id.as_str());
        if expected_parent != parent_exposure_id {
            return Err(CurriculumError::WrongParent);
        }
        let exposure_id = exposure_id.into();
        if exposure_id.trim().is_empty()
            || exposure_id.len() > MAX_ID_BYTES
            || self.exposures.iter().any(|e| e.exposure_id == exposure_id)
        {
            return Err(CurriculumError::DuplicateExposure);
        }
        if self
            .exposures
            .iter()
            .filter(|e| e.task_id == task.task_id)
            .count()
            >= MAX_EXPOSURES_PER_TASK
        {
            return Err(CurriculumError::TaskReplayLimit);
        }
        self.exposures.push(CurriculumExposure {
            exposure_id,
            task_id: task.task_id.clone(),
            family_id: task.family_id.clone(),
            schedule_position: position,
            parent_exposure_id: parent_exposure_id.map(str::to_owned),
            synthetic: task.synthetic,
            success,
        });
        self.snapshot_id = exposure_digest(
            &self.snapshot_id,
            self.exposures.last().expect("just appended exposure"),
        );
        Ok(())
    }

    /// Candidate outcomes are intentionally insufficient to qualify real-family
    /// improvement. This type has no acceptance or promotion authority.
    pub fn can_claim_real_family_improvement(&self) -> bool {
        false
    }

    pub fn synthetic_successes(&self) -> usize {
        self.exposures
            .iter()
            .filter(|e| e.synthetic && e.success)
            .count()
    }
}

fn snapshot_digest(
    schedule_id: &str,
    policy: &str,
    replay_limit: usize,
    schedule: &[PlantedTask],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"jopedime.selfplay-curriculum.v1\0");
    put_string(&mut hasher, schedule_id);
    put_string(&mut hasher, policy);
    hasher.update((replay_limit as u64).to_le_bytes());
    hasher.update((schedule.len() as u64).to_le_bytes());
    for task in schedule {
        put_string(&mut hasher, &task.task_id);
        put_string(&mut hasher, &task.family_id);
        hasher.update([u8::from(task.synthetic)]);
    }
    hex::encode(hasher.finalize())
}

fn put_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn exposure_digest(previous: &str, exposure: &CurriculumExposure) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"jopedime.selfplay-curriculum-exposure.v1\0");
    put_string(&mut hasher, previous);
    put_string(&mut hasher, &exposure.exposure_id);
    put_string(&mut hasher, &exposure.task_id);
    put_string(&mut hasher, &exposure.family_id);
    hasher.update((exposure.schedule_position as u64).to_le_bytes());
    match &exposure.parent_exposure_id {
        Some(parent) => {
            hasher.update([1]);
            put_string(&mut hasher, parent);
        }
        None => hasher.update([0]),
    }
    hasher.update([u8::from(exposure.synthetic), u8::from(exposure.success)]);
    hex::encode(hasher.finalize())
}
